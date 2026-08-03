# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.ir — a model as plain data.

The name is short for *intermediate representation*: the form a model takes
between the library that wrote it and the service that solves it. A ``Program``
is exactly that form — variables, expressions and constraints, with no trace of
the library it came from. An importer (``quicopt.pyomo``, ``quicopt.mathopt``,
``quicopt.pulp``) builds one; ``quicopt.wire`` encodes it to the bytes the
service reads. The shape of a ``Program`` is the service's published contract — these types track
it, they never fork it. An index tuple entry is either an ``int``, a concrete
coordinate, or a ``str``, the name of an index to bind.

Example:
    Maximize ``3x + 5y`` subject to ``x + 2y ≤ 5``, with ``0 ≤ x ≤ 4`` and ``y``
    binary — written directly, with no modeling library in sight:

    ```python
    from quicopt import Client
    from quicopt.ir import (Program, VarDecl, Constraint, Nonneg,
                            Var, Const, Apply, CONTINUOUS, BINARY)

    p = Program(
        vars=[VarDecl("x", [], CONTINUOUS, 0.0, 4.0, 0.0),
              VarDecl("y", [], BINARY, 0.0, 1.0, 0.0)],
        objective=Apply("+", [Apply("*", [Const(3.0), Var("x")]),
                              Apply("*", [Const(5.0), Var("y")])]),
        sense="max",
        constraints=[Constraint(
            f=Apply("-", [Const(5.0),
                          Apply("+", [Var("x"),
                                      Apply("*", [Const(2.0), Var("y")])])]),
            set=Nonneg())],
    )

    result = Client().solve(p)                # a Program solves like a model
    print(result.status, result.objective)    # optimal 14.0
    print(result.solution)                    # {'x': 3.0, 'y': 1.0}
    ```

    A constraint is a *set membership*: the expression ``f`` must land in ``set``,
    so ``x + 2y ≤ 5`` is written as ``5 − (x + 2y) ∈ Nonneg`` — one sign
    convention rather than two.

    This is byte-for-byte the ``Program`` that ``quicopt.pulp`` builds from the
    same model written in PuLP, and it solves to the same answer. So writing a
    ``Program`` by hand is a real option — for a model no front-end expresses, or
    for generating one programmatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

__all__ = [
    "Const", "Param", "Var", "Apply", "Reduce", "SetRef", "SourceRef",
    "Zero", "Nonneg", "Indicator",
    "Parametric", "Empirical",
    "VarDecl", "IndexSet", "Constraint", "Program",
    "Domain", "CONTINUOUS", "INTEGER", "BINARY",
]


class Domain(IntEnum):
    """The domain a variable ranges over. Values are the codes the service reads."""
    CONTINUOUS = 1
    INTEGER = 2
    BINARY = 3


CONTINUOUS = Domain.CONTINUOUS
INTEGER = Domain.INTEGER
BINARY = Domain.BINARY


# ── Expression = Const | Param | Var | Apply | Reduce | SourceRef ───────────

class Expression:
    """Base of the expression grammar."""


@dataclass
class Const(Expression):
    """A literal numeric constant."""
    value: float


@dataclass
class Param(Expression):
    """A reference to a parameter-table entry — data bound at instance time, by
    ``name`` and an index tuple (``()`` for a scalar)."""
    name: str
    index: tuple = ()


@dataclass
class Var(Expression):
    """A reference to a decision variable, by ``name`` and an index tuple
    (``()`` for a scalar)."""
    name: str
    index: tuple = ()


@dataclass
class Apply(Expression):
    """A catalog operator ``op`` applied to its argument subexpressions."""
    op: str
    args: list                  # list[Expression]


@dataclass
class SetRef:
    """A reference to an index set: ``args=()`` is the flat set; ``args=("i",)``
    references the set indexed by enclosing bound indices."""
    name: str
    args: tuple = ()


@dataclass
class Reduce(Expression):
    """A fold of ``body`` over ``idx`` ranging across ``over`` — e.g. a Σ or Π —
    keeping a term only where ``cond`` is non-zero (``None`` ⇒ keep every term)."""
    op: str                     # the fold operator key (+, *, …)
    idx: str                    # the bound dummy index
    over: SetRef
    body: Expression
    cond: "Expression | None" = None    # keep a term only where cond ≠ 0


@dataclass
class SourceRef(Expression):
    """A reference to a random variable declared in ``Program.sources``.

    Identity lives in the name: every ``SourceRef("demand")`` denotes the *same*
    draw, however many times the subtree containing it is copied, so a model's
    correlation structure survives substitution and tree copying. Two independent
    random variables are two declarations under two names."""
    name: str


# ── ConSet = Zero | Nonneg | Indicator ──────────────────────────────────────

@dataclass
class Zero:
    """f = 0"""


@dataclass
class Nonneg:
    """f ≥ 0"""


@dataclass
class Indicator:
    """``bin`` active (= 1) implies the body satisfies the ``inner`` ConSet."""
    bin: Var                    # the binary being active implies f ∈ inner
    inner: object               # ConSet


# ── Source = Parametric | Empirical ─────────────────────────────────────────

@dataclass
class Parametric:
    """A random variable drawn from a distribution ``head`` with ``params``.

    ``head`` is a catalog operator (``"normal"``, …) and each parameter is an
    ordinary deterministic expression — so a distribution whose mean is itself a
    decision (an *endogenous* distribution) needs nothing the grammar does not
    already have. A parameter containing a ``SourceRef`` is rejected by the
    service: a distribution's parameters are data, not draws."""
    head: str
    params: list               # list[Expression]


@dataclass
class Empirical:
    """A random variable given as a fixed scenario column of ``data``.

    Exactly ``Program.scenarios`` values, one per scenario. Several empirical
    columns are read at the same scenario index, so columns drawn jointly stay
    correlated — which is how a joint distribution is expressed."""
    data: list                 # list[float], length == Program.scenarios


# ── variables, constraints, container ───────────────────────────────────────

@dataclass
class VarDecl:
    """A variable declaration: ``name`` over the product of the index sets in
    ``axes`` (``[]`` ⇒ scalar), ranging over ``domain``, with ``lower``/``upper``
    bounds and an initial ``start``. A bound is a float, or the ``str`` name of a
    Param table when it varies by index."""
    name: str
    axes: list                  # list[str] of set names; [] ⇒ scalar
    domain: Domain
    lower: object               # float scalar, or str = name of a Param table
    upper: object
    start: float


@dataclass
class IndexSet:
    """A named index set with concrete ``elements`` (each ``int`` or ``str``)."""
    name: str
    elements: list              # list[int | str]


@dataclass
class Constraint:
    """A constraint row: the expression ``f`` lies in the ConSet ``set``, for every
    binding in ``over`` (``[(dummy, SetRef)]``; empty ⇒ a single scalar row)."""
    f: Expression
    set: object                 # ConSet
    over: list = field(default_factory=list)    # list[(str, SetRef)] quantifier bindings


@dataclass
class Program:
    """A complete optimization model: index sets and data tables, variable
    declarations, the objective and its ``sense``, the constraint rows, and any
    per-index variable pins (``fix``).

    A model under uncertainty adds three more: the random variables it draws
    (``sources``), how many scenarios are drawn (``scenarios``) and the seed they
    are drawn from (``scenario_seed``). The last two are model *data* — they pin
    the sampled instance, so the same Program always sees the same draws. Left at
    their defaults they say nothing, and the encoded bytes are those of a
    deterministic model."""
    sets: list = field(default_factory=list)          # list[IndexSet]
    indexed_sets: dict = field(default_factory=dict)  # {name: {tuple: list}}  a[(i,)] -> [j…]
    params: dict = field(default_factory=dict)        # {name: {tuple: float}} p[(i,)], A[(i,j)]
    vars: list = field(default_factory=list)          # list[VarDecl]
    objective: Expression = None
    sense: str = "min"                                # "min" | "max"
    constraints: list = field(default_factory=list)   # list[Constraint]
    fix: dict = field(default_factory=dict)           # {(varname, tuple): float} per-index pins
    scenarios: int = 1                                # R — how many are drawn
    scenario_seed: int = 1                            # the draw seed
    sources: dict = field(default_factory=dict)       # {name: Parametric | Empirical}
