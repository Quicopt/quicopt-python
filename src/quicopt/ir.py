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
    "StructRef", "TableRef",
    "Zero", "Nonneg", "Indicator",
    "Parametric", "Empirical", "PermutationDecl", "PartitionDecl",
    "VarDecl", "IndexSet", "Constraint", "Program",
    "Domain", "CONTINUOUS", "INTEGER", "BINARY", "SEMICONTINUOUS", "SEMIINTEGER",
]


class Domain(IntEnum):
    """The domain a variable ranges over. Values are the codes the service reads.

    ``SEMICONTINUOUS`` and ``SEMIINTEGER`` are the two domains with a hole: the
    variable is 0 or lies within its bounds, never in between — any value there
    for ``SEMICONTINUOUS``, a whole number for ``SEMIINTEGER``. Both bounds must
    be finite. When 0 already lies within the bounds there is no hole, and the
    variable is an ordinary continuous or integer one.
    """
    CONTINUOUS = 1
    INTEGER = 2
    BINARY = 3
    SEMICONTINUOUS = 4
    SEMIINTEGER = 5


CONTINUOUS = Domain.CONTINUOUS
INTEGER = Domain.INTEGER
BINARY = Domain.BINARY
SEMICONTINUOUS = Domain.SEMICONTINUOUS
SEMIINTEGER = Domain.SEMIINTEGER


# ── Expression = Const | Param | Var | Apply | Reduce | SourceRef | StructRef | TableRef

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


@dataclass
class StructRef(Expression):
    """A reference to a structured decision variable declared in
    ``Program.structures``, by name.

    It appears only as the second argument of the operator that reads one of the
    structure's parts: ``Apply("item_at", [Const(3.0), StructRef("tour")])`` is the
    item in slot 3 of the permutation ``tour``, and ``Apply("slot_of", [Const(2.0),
    StructRef("tour")])`` the slot of item 2. A partition's parts are read with
    ``"part_of"`` (the part an item is in) and ``"size_of"`` (how many items a
    part holds). Items, slots and parts count from 1 here, as in the service's
    schema."""
    name: str


@dataclass
class TableRef(Expression):
    """The entry of the parameter table ``param`` at positions given by
    expressions: one index for a table of one dimension, two for a table of two.

    Unlike a [`Param`][quicopt.ir.Param], whose index is fixed data, an index here may involve
    decisions — a structure's part, an integer variable — so the entry is read
    when the model is solved. The table must hold every entry on the positions
    ``1..a`` (or ``1..a`` by ``1..b``). An index value that is not a whole number
    is rounded, and one outside the table is moved to its nearest end."""
    param: str
    index: list                 # list[Expression], one or two


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


# ── Structure = PermutationDecl | PartitionDecl ─────────────────────────────

@dataclass
class PermutationDecl:
    """A permutation of ``size`` items over ``size`` slots, as the service's schema
    states it: items and slots count from 1.

    ``start`` is empty, or lists the starting slot of each item (``start[i - 1]``
    is the slot of item ``i``), so it holds each of ``1..size`` once. ``fixed``
    pins the permutation at ``start``, which must then be given. ``precede`` lists
    ``(before, after)`` item pairs: ``before`` is placed in an earlier slot than
    ``after`` in every arrangement the service considers."""
    size: int
    start: list = field(default_factory=list)      # list[int], empty or a permutation of 1..size
    fixed: bool = False
    precede: list = field(default_factory=list)    # list[(int, int)]


@dataclass
class PartitionDecl:
    """An assignment of ``size`` items to ``parts`` parts, as the service's schema
    states it: items and parts count from 1, and a part may stay empty.

    ``start`` is empty, or lists the starting part of each item (``start[i - 1]``
    is the part of item ``i``). ``fixed`` pins the assignment at ``start``, which
    must then be given."""
    size: int
    parts: int
    start: list = field(default_factory=list)      # list[int], empty or one part in 1..parts per item
    fixed: bool = False


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
    deterministic model.

    Structured decision variables, a permutation or a partition, are declared by
    name in ``structures`` and read in expressions through [`StructRef`][quicopt.ir.StructRef].

    ``blocks`` names groups of variables, each the variables of one subproblem of
    the model (one customer's supply, say), as ``{name: [Var, …]}``. A variable is
    in at most one block, and a block's members need finite bounds. A block
    changes nothing about what the model means; the service keeps each block's
    own constraints, those over its members alone, satisfied in every solution
    it returns."""
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
    structures: dict = field(default_factory=dict)    # {name: PermutationDecl | PartitionDecl}
    blocks: dict = field(default_factory=dict)        # {name: list[Var]}
