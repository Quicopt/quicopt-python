# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.pyomo — a Pyomo model → a Quicopt ``Program``.

Reads the model at Pyomo's expression-tree level: the single active objective and
each constraint are walked into Quicopt's expression graph, variable bounds and
domain into ``VarDecl``, and each constraint into a ``Zero``/``Nonneg`` row. The
result is a *flat* ``Program`` (one scalar ``VarDecl`` per Pyomo ``VarData``; Pyomo
has already expanded indexed components).

An operator the service does not support raises, rather than being approximated
or dropped — so a model that imports is a model that means what it says.

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
    components and declaration order is what is left to identify them by.
"""
from __future__ import annotations

from math import inf

import pyomo.environ as pyo

from ._terms import _emit, _sum
from .ir import (Const, Var, Apply, VarDecl, Program, SourceRef, StructRef, TableRef,
                 Parametric, Empirical, PermutationDecl, PartitionDecl,
                 CONTINUOUS, INTEGER, BINARY, SEMICONTINUOUS, SEMIINTEGER)
from .stochastic import _declarations
from .structured import _declarations as _structured_declarations, _Part, _Lookup

# The ops the service knows. A head outside this set raises — resolve it in the
# service's catalog, do not paper over it here. ``smean`` to ``sfreq_geq`` are the
# scenario aggregators (see ``quicopt.stochastic``); ``max``/``min`` are Pyomo's
# own nodes; ``item_at``/``slot_of`` read a permutation and ``part_of``/``size_of``
# a partition (``quicopt.structured``); ``indicator`` is ``active``'s head, and
# ``step`` and ``indicator`` are what Pyomo's ``Expr_if`` imports as.
_CATALOG = {"+", "-", "*", "/", "^", "sin", "cos", "exp", "log", "sqrt", "abs",
            "max", "min", "smean", "scvar", "sfreq_leq", "sfreq_geq",
            "item_at", "slot_of", "part_of", "size_of", "indicator", "step"}


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
            [`Var`][quicopt.ir.Var].

    Returns:
        Expression: The corresponding Quicopt expression node.
    """
    if isinstance(node, (int, float)):
        return Const(float(node))
    if isinstance(node, _Part):                 # counted from 0 here, from 1 on the wire
        part = _apply(node._head, [Const(float(node._position + 1)), StructRef(node._structure)])
        return part if node._head == "size_of" else Apply("-", [part, Const(1.0)])   # a size is a count
    if isinstance(node, _Lookup):
        return TableRef(node._table, [_one_based(_expr(a, var)) for a in node.args])
    if node.is_expression_type() and node.getname() == "Expr_if":
        return _expr_if(node, var)
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


def _expr_if(node, var):
    """Import Pyomo's ``Expr_if(IF=c, THEN=a, ELSE=b)`` as ``b + (a − b)·t``.

    ``t`` is 1 where the comparison ``c`` holds, else 0, written with the two 0/1
    operators the service knows: ``step(e) = 1`` for ``e ≥ 0``, and
    ``indicator(e) = 1`` for ``e ≠ 0``. Pyomo stores ``x ≥ y`` as ``y ≤ x``, so
    three cases cover every comparison:

    - ``l ≤ r`` is ``step(r − l)``;
    - ``l < r`` is ``1 − step(l − r)``;
    - ``l = r`` is ``1 − indicator(l − r)``.

    With ``THEN=1, ELSE=0`` the result is ``t`` itself.

    Args:
        node: The Pyomo ``Expr_if`` node.
        var: The variable-mapping callable.

    Returns:
        Expression: The imported expression.

    Raises:
        ValueError: If the condition is not a single comparison (a ranged
            ``lo ≤ x ≤ hi``, a constant, a logical expression).
    """
    cond, then, other = node.args
    name = type(cond).__name__
    if name == "InequalityExpression":
        lhs, rhs = (_expr(a, var) for a in cond.args)
        t = (Apply("-", [Const(1.0), _apply("step", [Apply("-", [lhs, rhs])])]) if cond.strict
             else _apply("step", [Apply("-", [rhs, lhs])]))
    elif name == "EqualityExpression":
        lhs, rhs = (_expr(a, var) for a in cond.args)
        t = Apply("-", [Const(1.0), _apply("indicator", [Apply("-", [lhs, rhs])])])
    else:
        raise ValueError(f"Expr_if: the condition must be one comparison (<=, <, >=, >, ==), "
                         f"got {cond}; write a range as two nested Expr_if")
    a, b = _expr(then, var), _expr(other, var)
    if isinstance(a, Const) and isinstance(b, Const):
        if (a.value, b.value) == (1.0, 0.0):
            return t
        return _sum([b, Apply("*", [Const(a.value - b.value), t])])
    return _sum([b, Apply("*", [Apply("-", [a, b]), t])])


def _apply(op, args):
    """Build an ``Apply`` node, guarding the head against the operator catalog.

    Args:
        op: The operator head; must be one of ``_CATALOG``.
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


def _one_based(index):
    """Shift a position counted from 0 to the wire's count from 1: ``index + 1``.

    Undoes a ``− 1`` rather than stacking a ``+ 1`` on it, so a lookup at a
    permutation's part, which reads ``item_at(…) − 1`` here, reaches the service
    as ``item_at(…)``. The rewrite holds for any ``e − 1``, not only a part's.

    Args:
        index: The position expression, counted from 0.

    Returns:
        Expression: The same position, counted from 1.
    """
    if isinstance(index, Const):
        return Const(index.value + 1.0)
    if isinstance(index, Apply) and index.op == "-" and index.args[1] == Const(1.0):
        return index.args[0]
    return Apply("+", [index, Const(1.0)])


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


def _source_decls(sources, n_scenarios, var):
    """Build the ``Program``'s source declarations from the model's registry.

    A parametric distribution's parameters are ordinary deterministic expressions
    and are walked like any other; an empirical column is already data, and only
    has to line up with the scenario count.

    Args:
        sources: The registry ``id(vardata) → (vardata, name, spec)``.
        n_scenarios: The model's scenario count, which every column must match.
        var: The variable-mapping closure, for walking distribution parameters.

    Returns:
        dict: ``name → Parametric | Empirical``, the ``Program.sources`` table.

    Raises:
        ValueError: If an empirical column's length is not the scenario count.
    """
    decls = {}
    for (_, name, spec) in sources.values():
        if isinstance(spec, Empirical):
            if len(spec.data) != n_scenarios:
                raise ValueError(f"the scenario column for '{name}' has {len(spec.data)} "
                                 f"values, but the model is solved over {n_scenarios} "
                                 "scenarios — a column carries one value per scenario")
            decls[name] = Empirical([float(x) for x in spec.data])
        else:
            decls[name] = Parametric(spec.head, [_expr(p, var) for p in spec.params])
    return decls


def _structure_decls(structured):
    """Build the ``Program``'s structure declarations from the model's registry,
    shifting items, slots and parts from the surface's count from 0 to the wire's
    from 1.

    Args:
        structured: The model's ``_Structured`` record.

    Returns:
        dict: ``name → PermutationDecl | PartitionDecl``, the ``Program.structures``
        table.

    Raises:
        ValueError: If a start places an item after one it must follow.
    """
    decls = {}
    for (name, perm) in structured.permutations.items():
        for (before, after) in perm.precede:
            if perm.start and perm.start[before] >= perm.start[after]:
                raise ValueError(f"'{name}': the start places item {before} after item "
                                 f"{after}, which it must precede")
        decls[name] = PermutationDecl(perm.n_items, start=[s + 1 for s in perm.start],
                                      precede=[(a + 1, b + 1) for (a, b) in perm.precede])
    for (name, part) in structured.partitions.items():
        decls[name] = PartitionDecl(part.n_items, part.n_parts,
                                    start=[b + 1 for b in part.start])
    return decls


def _table_params(structured):
    """Build a parameter table for each lookup table, keyed by its positions
    counted from 1, as the service reads them.

    Args:
        structured: The model's ``_Structured`` record.

    Returns:
        dict: ``name → {(i,): value}`` or ``name → {(i, j): value}``.
    """
    params = {}
    for (name, values) in structured.tables.items():
        if isinstance(values[0], list):
            params[name] = {(i + 1, j + 1): x for (i, row) in enumerate(values)
                            for (j, x) in enumerate(row)}
        else:
            params[name] = {(i + 1,): x for (i, x) in enumerate(values)}
    return params


def _check_names(structured, taken):
    """Refuse a permutation, partition or table whose name is already in use in
    the model.

    The service reads every name as one thing — a decision variable, a random
    variable, a structure or a table — so a clash would be refused there; here the
    message can say which declaration to rename.

    Args:
        structured: The model's ``_Structured`` record.
        taken: The names of the decision and random variables.

    Raises:
        ValueError: On the first clash.
    """
    for name in list(structured.permutations) + list(structured.partitions) + list(structured.tables):
        if name in taken:
            raise ValueError(f"the name '{name}' of a permutation, partition or lookup table is "
                             "also the name of a variable in this model; rename one of them")


def _domain(v, holed):
    """The ``Program`` domain of the Pyomo variable ``v``.

    A variable marked as 0 or within its bounds (``holed``) is ``SEMIINTEGER`` or
    ``SEMICONTINUOUS`` by its integrality, unless it is fixed: a pin takes one
    value, which is no hole. Every other variable is ``BINARY``, ``INTEGER`` or
    ``CONTINUOUS``.

    Args:
        v: The Pyomo ``VarData``.
        holed: Whether ``v`` was marked by ``semicontinuous``/``semiinteger``.

    Returns:
        Domain: The domain.

    Raises:
        ValueError: If a marked variable lacks a finite bound.
    """
    if holed and not v.fixed:
        if v.lb is None or v.ub is None:
            raise ValueError(f"variable '{v.name}' is 0 or within its bounds, so both bounds "
                             f"must be finite; it has [{v.lb}, {v.ub}]")
        return SEMIINTEGER if v.is_integer() else SEMICONTINUOUS
    return BINARY if v.is_binary() else INTEGER if v.is_integer() else CONTINUOUS


def _blocks(structured, name, sources):
    """Build the ``Program``'s blocks from the model's registry.

    Args:
        structured: The model's ``_Structured`` record.
        name: The map ``id(vardata) → Program variable name``.
        sources: The model's random variables, ``id(vardata) → (…)``.

    Returns:
        dict: ``block name → [Var, …]``, members in the order given.

    Raises:
        ValueError: If a member is a random variable or has an infinite bound.
    """
    blocks = {}
    for (bname, members) in structured.blocks.items():
        for v in members:
            if id(v) in sources:
                raise ValueError(f"block '{bname}': '{v.name}' is a random variable, not a "
                                 "decision; a block holds decision variables")
            if not v.fixed and (v.lb is None or v.ub is None):
                raise ValueError(f"block '{bname}': '{v.name}' has bounds [{v.lb}, {v.ub}]; "
                                 "a block's variables need finite bounds")
        blocks[bname] = [Var(name[id(v)], ()) for v in members]
    return blocks


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

    A permutation, partition or lookup table (see ``quicopt.structured``) becomes a
    named declaration in ``Program.structures`` or ``Program.params``. Its items,
    slots, parts and positions count from 0 in the model and from 1 in the
    ``Program``, which is how the service counts them; the shift is made here. A
    variable marked as 0 or within its bounds becomes ``SEMICONTINUOUS`` or
    ``SEMIINTEGER``, and the model's blocks become ``Program.blocks``.

    Pyomo's ``Expr_if(IF=c, THEN=a, ELSE=b)`` is read when ``c`` is one
    comparison (``<=``, ``<``, ``>=``, ``>`` or ``==``); a range such as
    ``pyo.inequality(1, x, 3)`` raises, and is written as two nested ``Expr_if``.
    """
    declared = _declarations(m)
    structured = _structured_declarations(m)
    holes = {} if structured is None else structured.holes
    sources = {} if declared is None else declared.sources
    n_scenarios = 1 if declared is None else declared.n_scenarios
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
        if v.fixed and v.value is None:            # else the pin below is (None, None) ⇒ ±Inf ⇒ silently free
            raise ValueError(f"variable '{v.name}' is fixed but has no value")
        domain = _domain(v, id(v) in holes)
        lb, ub = (v.value, v.value) if v.fixed else (v.lb, v.ub)   # a fixed var ⇒ a [val, val] pin
        lb = -inf if lb is None else lb                            # an absent Pyomo bound ⇒ ±Inf (free)
        ub =  inf if ub is None else ub
        if v.value is not None:
            start = v.value
        elif domain in (SEMICONTINUOUS, SEMIINTEGER):
            start = 0.0                                            # 0 is in a holed domain: start off
        else:
            start = min(max(0.0, lb), ub)                          # clamp 0 into [lb, ub]
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

    structures, params, blocks = {}, {}, {}
    if structured is not None:
        _check_names(structured, set(name.values())
                     | {source_name for (_, source_name, _) in sources.values()})
        structures, params = _structure_decls(structured), _table_params(structured)
        blocks = _blocks(structured, name, sources)

    return Program(params=params, vars=vars, objective=objective, sense=sense,
                   constraints=cons, scenarios=n_scenarios, scenario_seed=seed,
                   sources=_source_decls(sources, n_scenarios, var), structures=structures,
                   blocks=blocks)
