# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.pyomo — a Pyomo model → a Quicopt ``Program``.

Reads the model at Pyomo's expression-tree level: the single active objective and
each constraint are walked into Quicopt's expression graph, variable bounds and
domain into ``VarDecl``, and each constraint into a ``Zero``/``Nonneg`` row. The
result is a *flat* ``Program`` (one scalar ``VarDecl`` per Pyomo ``VarData``; Pyomo
has already expanded indexed components).

An operator the service does not know is a gap to close in the service, never
papered over here.

Example:
    Maximize ``3x + 5y`` subject to ``x + 2y ≤ 5``, with ``0 ≤ x ≤ 4`` and ``y``
    binary:

    ```python
    import pyomo.environ as pyo
    from quicopt import Client

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.y = pyo.Var(domain=pyo.Binary)
    m.cap = pyo.Constraint(expr=m.x + 2 * m.y <= 5)
    m.obj = pyo.Objective(expr=3 * m.x + 5 * m.y, sense=pyo.maximize)

    result = Client().solve(m)                # the import below happens inside
    print(result.status, result.objective)    # optimal 14.0
    print(result.solution)                    # {'x1': 3.0, 'x2': 1.0}
    ```

    [`Client.solve`][quicopt.client.Client.solve] takes the Pyomo model as it
    stands and calls this module on the way out, so the import is not a step to
    perform — reach for `import_model` only to hold the ``Program`` itself.

    Note what the solution is keyed by: ``x1``/``x2``, not ``m.x``/``m.y``.
    Variables are named positionally, because Pyomo has already expanded indexed
    components and declaration order is what is left to identify them by. Inside
    the ``Program``, the ``≤`` row has become ``5 − (x1 + 2·x2) ≥ 0``: an
    inequality has *one* sign convention in a ``Program``, so ``≤`` is carried by
    negating.
"""
from __future__ import annotations

from math import inf

import pyomo.environ as pyo

from ._terms import _emit, _sum
from .ir import (Const, Var, Apply, VarDecl, Program, SourceRef,
                 Parametric, Empirical, CONTINUOUS, INTEGER, BINARY)
from .stochastic import _declarations

# The ops the service knows. A head outside this set raises — resolve it in the
# service's catalog, do not paper over it here. The last four are the scenario
# aggregators (see ``quicopt.stochastic``); ``max``/``min`` are Pyomo's own nodes.
_CATALOG = {"+", "-", "*", "/", "^", "sin", "cos", "exp", "log", "sqrt", "abs",
            "max", "min", "smean", "scvar", "sfreq_leq", "sfreq_geq"}


# ── Pyomo expression node → Quicopt Expression ──────────────────────────────

def _expr(node, var):
    """Walk a Pyomo expression node into a Quicopt ``Expression``.

    Recurses the Pyomo tree, dispatching on each node's ``getname()`` head: ``sum``
    → n-ary ``+``, ``prod``/``mon`` → ``*``, ``div``/``pow`` → ``/``/``^``, ``neg``
    → ``0 − a`` (the catalog has only binary ``−``), and a unary math head passes
    through under its own catalog name. Variable leaves are mapped by ``var``;
    numeric and parameter leaves are baked to ``Const`` data.

    Args:
        node: A Pyomo expression node, variable, parameter, or Python number.
        var: A callable mapping a Pyomo ``VarData`` to its
            :class:`~quicopt.ir.Var`.

    Returns:
        Expression: The corresponding Quicopt expression node.
    """
    if isinstance(node, (int, float)):
        return Const(float(node))
    if node.is_expression_type():
        head = node.getname()
        args = [_expr(a, var) for a in node.args]
        if head == "sum":                       # SumExpression and flattened LinearExpression
            return _sum(args)
        if head in ("prod", "mon"):             # a*b ; and coef*var (both binary)
            return _apply("*", args)
        if head == "div":
            return _apply("/", args)
        if head == "pow":
            return _apply("^", args)
        if head == "neg":                       # unary minus → 0 − a (catalog has binary −)
            return Apply("-", [Const(0.0), args[0]])
        if head in ("max", "min"):              # Pyomo's are n-ary, the catalog's binary
            return _nest(head, args)
        return _apply(head, args)               # a unary math fn reports its catalog name
    if node.is_variable_type():
        return var(node)
    return Const(float(pyo.value(node)))        # constant / parameter leaf → baked to data


def _apply(op, args):
    """Build an ``Apply`` node, guarding the head against the operator catalog.

    Args:
        op: The operator head; must be one of :data:`_CATALOG`.
        args: The argument expressions.

    Returns:
        Apply: The ``Apply(op, args)`` node.

    Raises:
        ValueError: If ``op`` is not in the service's operator catalog — a coverage
            gap to resolve in the service, not to paper over here.
    """
    if op not in _CATALOG:
        raise ValueError(f"operator {op!r} is not in the Quicopt operator catalog — "
                         "it must be added to the service before importing")
    return Apply(op, args)


def _nest(op, args):
    """Rewrite an n-ary comparison into nested binary ones.

    ``max(a, b, c)`` becomes ``max(max(a, b), c)``: Pyomo's ``max``/``min`` take
    any number of arguments, the catalog's take two, and the two agree because
    both are associative.

    Args:
        op: ``"max"`` or ``"min"``.
        args: The argument expressions, at least one.

    Returns:
        Expression: The nested application, or the sole argument if there is one.
    """
    nested = args[0]
    for a in args[1:]:
        nested = _apply(op, [nested, a])
    return nested


def _check_source(v, name):
    """Reject a random variable that was also declared like a decision variable.

    Bounds, a domain and a fixed value all say what a variable may be — which for
    a random one is already said by its distribution. Accepting both would leave
    two answers to one question, so this is an error rather than a precedence rule.

    Args:
        v: The Pyomo ``VarData`` carrying a distribution.
        name: The name it was declared under, for the message.

    Returns:
        None.

    Raises:
        ValueError: If ``v`` carries bounds, an integrality domain, or a pin.
    """
    if v.fixed or v.lb is not None or v.ub is not None or v.is_integer() or v.is_binary():
        raise ValueError(f"the random variable '{name}' also carries bounds, a domain or a "
                         "fixed value — what it can be is said by its distribution; "
                         "declare it as pyo.Var() and nothing more")


def _source_decls(sources, scenarios, var):
    """Build the ``Program``'s source declarations from the model's registry.

    A parametric distribution's parameters are ordinary deterministic expressions
    and are walked like any other; an empirical column is already data, and only
    has to line up with the scenario count.

    Args:
        sources: The registry ``id(vardata) → (vardata, name, spec)``.
        scenarios: The model's scenario count, which every column must match.
        var: The variable-mapping closure, for walking distribution parameters.

    Returns:
        dict: ``name → Parametric | Empirical``, the ``Program.sources`` table.

    Raises:
        ValueError: If an empirical column's length is not the scenario count.
    """
    decls = {}
    for (_, name, spec) in sources.values():
        if isinstance(spec, Empirical):
            if len(spec.data) != scenarios:
                raise ValueError(f"the scenario column for '{name}' has {len(spec.data)} "
                                 f"values, but the model is solved over {scenarios} "
                                 "scenarios — a column carries one value per scenario")
            decls[name] = Empirical([float(x) for x in spec.data])
        else:
            decls[name] = Parametric(spec.head, [_expr(p, var) for p in spec.params])
    return decls


def import_model(m):
    """Convert a Pyomo ``ConcreteModel`` into a Quicopt ``Program``.

    Each ``VarData`` becomes a scalar ``VarDecl`` (``x{i}`` in declaration order)
    carrying its bounds and domain (``Binary``→``BINARY``, integer domains→
    ``INTEGER``, else ``CONTINUOUS``); the active objective and every constraint
    become Quicopt expressions. Requires exactly one active objective; an absent
    Pyomo variable bound means unbounded in that direction (±Inf). A variable that
    is fixed but carries no value **raises**: its pin has no value to pin *to*, and the ±Inf
    reading of an absent bound would turn it into a free variable instead.

    A variable given a distribution (see ``quicopt.stochastic``) is not a decision
    at all: it leaves the variable list, becomes a named declaration in
    ``Program.sources``, and its every use becomes a reference to that name. The
    scenario count and seed ride along as model data. Names are assigned across
    *all* variables before the random ones are set aside, so which variables are
    random never renumbers the rest. A model that declares no uncertainty imports
    to exactly the ``Program`` it always did.
    """
    declared = _declarations(m)
    sources = {} if declared is None else declared.sources
    scenarios = 1 if declared is None else declared.scenarios
    seed = 1 if declared is None else declared.seed

    vis = list(m.component_data_objects(pyo.Var))
    name = {id(v): f"x{i + 1}" for i, v in enumerate(vis)}
    var = lambda v: (SourceRef(sources[id(v)][1]) if id(v) in sources
                     else Var(name[id(v)], ()))

    vars = []
    for v in vis:
        if id(v) in sources:                   # a random variable is declared, never decided
            _check_source(v, sources[id(v)][1])
            continue
        domain = BINARY if v.is_binary() else INTEGER if v.is_integer() else CONTINUOUS
        if v.fixed and v.value is None:            # else the pin below is (None, None) ⇒ ±Inf ⇒ silently free
            raise ValueError(f"variable '{v.name}' is fixed but has no value")
        lb, ub = (v.value, v.value) if v.fixed else (v.lb, v.ub)   # a fixed var ⇒ a [val, val] pin
        lb = -inf if lb is None else lb                            # an absent Pyomo bound ⇒ ±Inf (free)
        ub =  inf if ub is None else ub
        start = v.value if v.value is not None else min(max(0.0, lb), ub)   # clamp 0 into [lb, ub]
        vars.append(VarDecl(name[id(v)], [], domain, float(lb), float(ub), float(start)))

    objs = list(m.component_data_objects(pyo.Objective, active=True))
    if len(objs) != 1:
        raise ValueError(f"expected exactly one active objective, found {len(objs)}")
    sense = "min" if objs[0].sense == pyo.minimize else "max"
    objective = _expr(objs[0].expr, var)

    cons = []
    for c in m.component_data_objects(pyo.Constraint, active=True):
        lb = pyo.value(c.lower) if c.has_lb() else -inf   # an absent side ⇒ free, emits no row
        ub = pyo.value(c.upper) if c.has_ub() else inf
        _emit(cons, _expr(c.body, var), lb, ub)

    return Program(vars=vars, objective=objective, sense=sense, constraints=cons,
                   scenarios=scenarios, scenario_seed=seed,
                   sources=_source_decls(sources, scenarios, var))
