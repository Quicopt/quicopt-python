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
    inequality is *one* sign convention in the IR, so ``≤`` is carried by
    negating.
"""
from __future__ import annotations

from math import inf

import pyomo.environ as pyo

from ._terms import _emit, _sum
from .ir import (Const, Var, Apply, VarDecl, Program,
                 CONTINUOUS, INTEGER, BINARY)

# The ops the service knows. A head outside this set raises — resolve it in the
# service's catalog, do not paper over it here.
_CATALOG = {"+", "-", "*", "/", "^", "sin", "cos", "exp", "log", "sqrt", "abs"}


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


def import_model(m):
    """Convert a Pyomo ``ConcreteModel`` into a Quicopt ``Program``.

    Each ``VarData`` becomes a scalar ``VarDecl`` (``x{i}`` in declaration order)
    carrying its bounds and domain (``Binary``→``BINARY``, integer domains→
    ``INTEGER``, else ``CONTINUOUS``); the active objective and every constraint
    become Quicopt expressions. Requires exactly one active objective; an absent
    Pyomo variable bound means unbounded in that direction (±Inf). A variable that
    is fixed but carries no value **raises**: its pin has no value to pin *to*, and the ±Inf
    reading of an absent bound would turn it into a free variable instead.
    """
    vis = list(m.component_data_objects(pyo.Var))
    name = {id(v): f"x{i + 1}" for i, v in enumerate(vis)}
    var = lambda v: Var(name[id(v)], ())

    vars = []
    for v in vis:
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

    return Program(vars=vars, objective=objective, sense=sense, constraints=cons)
