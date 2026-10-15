# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.structured — orderings, groups, minimum loads and fees, written in Pyomo.

Some decisions are an ordering, such as the stops of a courier's route or the
jobs on a machine, or a one-to-one assignment, such as which department moves
into which office. Each is a *permutation*: things, called *items*, placed in
numbered places, called *slots*, one item per slot and one slot per item. For a
route the items are the stops and the slots are the visits, so slot 0 holds the
first stop visited.

[`permutation`][quicopt.structured.permutation] declares one on a Pyomo model.
Its [`item_at`][quicopt.structured.Permutation.item_at] is the item in a slot and
its [`slot_of`][quicopt.structured.Permutation.slot_of] the slot of an item; both
are expressions whose values the service chooses, like decision variables. Items
and slots count from 0, like the positions of a Python list.

Data that depends on the arrangement, such as the distance between consecutive
stops, is read from a [`lookup_table`][quicopt.structured.lookup_table]: indexing
it with expressions gives the entry at the positions the service chooses.

Both are declared on a Pyomo model. PuLP and OR-Tools MathOpt models hold only
linear and quadratic terms, which cannot express either.

Example:
    Five stops along a road. Find the shortest route through all of them that
    visits stop 3 before stop 0:

    ```python
    import pyomo.environ as pyo
    from quicopt import Client
    from quicopt.structured import permutation, lookup_table

    position = [0, 3, 1, 4, 2]                    # km along the road, stops 0 to 4
    m = pyo.ConcreteModel()
    dist = lookup_table(m, "dist", [[abs(a - b) for b in position] for a in position])
    route = permutation(m, "route", 5)
    route.precede(3, 0)                           # stop 3 is visited before stop 0
    m.length = pyo.Objective(
        expr=sum(dist[route.item_at(k), route.item_at(k + 1)] for k in range(4)),
        sense=pyo.minimize)

    result = Client().solve(m)
    print(result.objective)                       # 4.0
    print(result.structures["route"].item_at)     # [3, 1, 4, 2, 0]
    ```

    The route drives from one end of the road to the other. Starting at the far
    end is what visits stop 3 before stop 0.

    A model with a permutation or a lookup table is solved by a search, so its
    result has the status ``heuristic``. Permutations combine with random
    variables (``quicopt.stochastic``), for example a route whose travel times are
    uncertain, minimized in expectation.

Minimum loads and fees:
    A delivery is often either nothing or at least a minimum load: a truck does
    not go out for two units. [`semiinteger`][quicopt.structured.semiinteger]
    marks an integer variable so that it is 0 or a whole number within its
    bounds, and nothing in between;
    [`semicontinuous`][quicopt.structured.semicontinuous] does the same for a
    continuous one. [`active`][quicopt.structured.active] is 1 when such a
    variable is not 0, so ``5 * active(m.ship[w, c])`` charges 5 for every
    delivery made.

    A fee that depends on a total, such as a warehouse's opening fee due as soon
    as it delivers anything, is Pyomo's own ``Expr_if`` over a comparison:
    ``pyo.Expr_if(IF=total >= 1, THEN=60, ELSE=0)``. Any of ``<=``, ``<``,
    ``>=``, ``>`` and ``==`` works; [`minimum`][quicopt.structured.minimum] and
    [`maximum`][quicopt.structured.maximum] cap or floor an expression in the
    same spirit.

    A [`block`][quicopt.structured.block] names the variables of one subproblem,
    such as one customer's deliveries. Every solution the service returns
    satisfies the constraints that involve only one block's variables, such as
    that customer's demand.

    Three warehouses supply three customers. Each delivery is none or 10 to 25
    units and costs 5 plus a price per unit; a warehouse that delivers anything
    pays its opening fee:

    ```python
    import pyomo.environ as pyo
    from quicopt import Client
    from quicopt.structured import semiinteger, active, block

    demand = [40, 30, 20]                         # units, customers 0 to 2
    price = [[1, 3, 4], [3, 1, 3], [4, 2, 1]]     # per unit, price[warehouse][customer]
    opening = [60, 50, 40]                        # fee of warehouses 0 to 2

    m = pyo.ConcreteModel()
    m.ship = pyo.Var(range(3), range(3), within=pyo.NonNegativeIntegers, bounds=(10, 25))
    semiinteger(m.ship)                           # every delivery: none, or 10 to 25 units
    for c in range(3):
        block(m, f"customer_{c}", [m.ship[w, c] for w in range(3)])
    m.served = pyo.Constraint(range(3), rule=lambda m, c:
                              sum(m.ship[w, c] for w in range(3)) >= demand[c])
    m.cost = pyo.Objective(expr=
        sum(price[w][c] * m.ship[w, c] for w in range(3) for c in range(3))
        + sum(5 * active(m.ship[w, c]) for w in range(3) for c in range(3))
        + sum(pyo.Expr_if(IF=sum(m.ship[w, c] for c in range(3)) >= 1,
                          THEN=opening[w], ELSE=0) for w in range(3)))

    result = Client().solve(m)
    print(result.objective)                       # 300.0
    # the deliveries ship[0, 0], ship[0, 1], …, ship[2, 2] are x1 to x9 in the solution
    print([result.solution[f"x{k}"] for k in range(1, 10)])
    # [25.0, 10.0, 0.0, 0.0, 0.0, 0.0, 15.0, 20.0, 20.0]
    ```

    Warehouse 1 stays closed. Warehouse 0 sends its full 25 units to customer 0
    and the minimum of 10 to customer 1; warehouse 2 covers the rest. Without
    the minimum load, the cheapest plan would cost 295 and send customer 1 a
    delivery of only 5 units from warehouse 0.

Splitting items into groups:
    [`partition`][quicopt.structured.partition] declares a split of items into
    parts as a decision: people into teams, jobs onto machines. Its
    [`part_of`][quicopt.structured.Partition.part_of] is the part an item is in
    and its [`size_of`][quicopt.structured.Partition.size_of] how many items a
    part holds. Items and parts count from 0.

    Six players of strengths 7, 5, 4, 3, 2 and 1 form a team of four and a team
    of two, as evenly matched as possible:

    ```python
    import pyomo.environ as pyo
    from quicopt import Client
    from quicopt.structured import partition, maximum

    strength = [7, 5, 4, 3, 2, 1]                 # of players 0 to 5
    m = pyo.ConcreteModel()
    teams = partition(m, "teams", 6, 2)           # 6 players into teams 0 and 1
    total = [sum(pyo.Expr_if(IF=teams.part_of(p) == t, THEN=strength[p], ELSE=0)
                 for p in range(6)) for t in range(2)]
    m.four = pyo.Constraint(expr=teams.size_of(0) == 4)
    m.stronger = pyo.Objective(expr=maximum(total[0], total[1]))

    result = Client().solve(m)
    print(result.objective)                       # 11.0
    print(result.structures["teams"].part_of)     # [1, 0, 1, 0, 0, 0]
    ```

    Players 0 and 2 (7 + 4) make up the team of two, and the other four also add
    up to 11.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

from pyomo.core import Model, PyomoObject, Var
from pyomo.core.expr import numeric_expr as _ne

from .stochastic import maximum, minimum

__all__ = ["permutation", "Permutation", "partition", "Partition", "lookup_table", "LookupTable",
           "semicontinuous", "semiinteger", "active", "block", "minimum", "maximum"]


# ── what a model has declared ───────────────────────────────────────────────

@dataclass
class _PermutationRecord:
    """One declared permutation, counted from 0 as the user wrote it.

    Attributes:
        n_items: How many items, and so how many slots.
        start: The starting slot of each item, or empty for the default (item
            ``i`` in slot ``i``).
        precede: ``(before, after)`` item pairs, in the order they were given.
    """
    n_items: int
    start: list
    precede: list = field(default_factory=list)


@dataclass
class _PartitionRecord:
    """One declared partition, counted from 0 as the user wrote it.

    Attributes:
        n_items: How many items.
        n_parts: How many parts.
        start: The starting part of each item, or empty for the service's default.
    """
    n_items: int
    n_parts: int
    start: list


@dataclass
class _Structured:
    """What a model has declared through this module.

    Held on the model itself, under a private attribute, and read by
    ``quicopt.pyomo.import_model`` on the way out. Pyomo ``VarData`` objects are
    unhashable, so the variable registries key on ``id`` and keep the variable
    beside it, which also keeps that id valid.

    Attributes:
        permutations: ``name → _PermutationRecord``.
        partitions: ``name → _PartitionRecord``.
        tables: ``name → values``, a list of numbers (one dimension) or a list of
            equally long lists of numbers (two dimensions).
        holes: ``id(vardata) → vardata``, the variables that are 0 or within
            their bounds.
        blocks: ``name → [vardata, …]``, in the order given.
    """
    permutations: dict = field(default_factory=dict)
    partitions: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    holes: dict = field(default_factory=dict)
    blocks: dict = field(default_factory=dict)


_SLOT = "_quicopt_structured"


def _declarations(model):
    """Read a model's structured declarations.

    Args:
        model: A Pyomo model.

    Returns:
        _Structured | None: The record, or ``None`` if the model declared no
        permutation and no lookup table.
    """
    return getattr(model, _SLOT, None)


def _record(model):
    """Read a model's structured declarations, creating the record on first use.

    Args:
        model: A Pyomo model.

    Returns:
        _Structured: The record.

    Raises:
        TypeError: If ``model`` is not a whole Pyomo model.
    """
    _check_model(model)
    record = _declarations(model)
    if record is None:
        record = _Structured()
        setattr(model, _SLOT, record)
    return record


def _declare(model, name):
    """Read a model's structured declarations, creating the record on first use,
    and check that ``name`` is free among its permutations, partitions and tables.

    Args:
        model: A Pyomo model.
        name: The name about to be declared.

    Returns:
        _Structured: The record.

    Raises:
        TypeError: If ``model`` is not a Pyomo model, or ``name`` is not a string.
        ValueError: If ``name`` is empty or already declared.
    """
    record = _record(model)
    _check_name(name)
    if name in record.permutations or name in record.partitions or name in record.tables:
        raise ValueError(f"the name '{name}' is already declared on this model")
    return record


def _check_name(name):
    """Check that ``name`` is a non-empty string.

    Raises:
        TypeError: If it is not.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(f"a name is a non-empty string, got {name!r}")


def _check_model(model):
    """Refuse a declaration on anything but a whole Pyomo model.

    Args:
        model: What the declaration is about to be recorded on.

    Raises:
        TypeError: If ``model`` is a block or another part of a Pyomo model, or
            not a Pyomo object at all (a PuLP or MathOpt model, say).
    """
    # The importer reads declarations off the model it is handed and nowhere else,
    # so one recorded on a block of that model would be silently lost.
    if isinstance(model, Model):
        return
    if isinstance(model, PyomoObject):
        raise TypeError("permutations, partitions, lookup tables and blocks are declared on the "
                        "Pyomo model itself, the one you solve, not on "
                        f"{type(model).__name__} '{getattr(model, 'name', model)}' inside it")
    raise TypeError("permutations, partitions, lookup tables and blocks are declared on a Pyomo "
                    "model, such as a pyo.ConcreteModel, not on "
                    f"{type(model).__module__}.{type(model).__name__}: PuLP and OR-Tools MathOpt "
                    "models hold only linear and quadratic terms, which cannot express them")


# ── expression nodes the service evaluates ──────────────────────────────────

class _Decided(_ne.NumericExpression):
    """A Pyomo expression node whose value only the service knows.

    Its value is a decision, so it must never look fixed to Pyomo: a node with no
    variable inside would otherwise read as a constant, and Pyomo would move it
    to a constraint's bound and try to evaluate it. It has no polynomial degree,
    and evaluating it locally is an error.
    """

    __slots__ = ()

    def nargs(self):
        """The number of child arguments."""
        return len(self._args_)

    def is_fixed(self):
        """Never fixed: the value is a decision."""
        return False

    def _is_fixed(self, values):
        """Never fixed, whatever the children are."""
        return False

    def _compute_polynomial_degree(self, result):
        """No polynomial degree: the value is read off a structure or a table."""
        return None

    def _apply_operation(self, result):
        """Refuse to evaluate: the value is the service's to choose.

        Raises:
            ValueError: Always.
        """
        raise ValueError(f"'{self}' has no value here — the service chooses it, so this "
                         "model solves through quicopt, not through a locally attached solver")


class _Part(_Decided):
    """One part of a declared structure: the item in a slot or the slot of an
    item of a permutation, the part of an item or the size of a part of a
    partition.

    Has no children: the position it reads (``position``, counted from 0) is
    data, and the structure is named, so neither is a Pyomo expression.
    """

    __slots__ = ("_head", "_structure", "_position")

    def __init__(self, head, structure, position, args=()):
        """Build the node.

        Args:
            head: ``"item_at"``, ``"slot_of"``, ``"part_of"`` or ``"size_of"``.
            structure: The structure's name.
            position: The slot, item or part read, counted from 0.
            args: Unused; present so Pyomo can rebuild the node.
        """
        super().__init__(args)
        self._head = head
        self._structure = structure
        self._position = position

    def getname(self, *args, **kwargs):
        """The part's name, which is how a walker dispatches on this node."""
        return self._head

    def create_node_with_local_data(self, args, classtype=None):
        """Rebuild this node, keeping what it reads.

        Pyomo clones expression trees while building constraints, and the default
        would call the constructor with the children alone, losing the head, the
        permutation and the position, which are all this node carries.

        Args:
            args: The new child arguments (none).
            classtype: A class to build instead, when a walker asks for one.

        Returns:
            The rebuilt node.
        """
        if classtype is not None:
            return classtype(args)
        return _Part(self._head, self._structure, self._position, args)

    def _to_string(self, values, verbose, smap):
        """Render as a method call, e.g. ``route.item_at(0)``."""
        return f"{self._structure}.{self._head}({self._position})"


class _Lookup(_Decided):
    """The entry of a lookup table at positions given by its children, counted
    from 0: one child for a table of one dimension, two for two."""

    __slots__ = ("_table",)

    def __init__(self, table, args):
        """Build the node.

        Args:
            table: The table's name.
            args: The index expressions, or plain whole numbers.
        """
        super().__init__(args)
        self._table = table

    def getname(self, *args, **kwargs):
        """``lookup``, the same for every table; the table is in ``_table``."""
        return "lookup"

    def create_node_with_local_data(self, args, classtype=None):
        """Rebuild this node around new children, keeping the table it reads.

        Args:
            args: The new index expressions.
            classtype: A class to build instead, when a walker asks for one.

        Returns:
            The rebuilt node.
        """
        return _Lookup(self._table, args) if classtype is None else classtype(args)

    def _to_string(self, values, verbose, smap):
        """Render as indexing, e.g. ``dist[route.item_at(0), route.item_at(1)]``."""
        return f"{self._table}[{', '.join(values)}]"


class _Active(_Decided):
    """1 when its one child, a variable, is not 0, else 0: the variable's
    indicator, ``indicator`` in the service's catalog."""

    __slots__ = ()

    def getname(self, *args, **kwargs):
        """``indicator``, the catalog operator this node is imported as."""
        return "indicator"

    def create_node_with_local_data(self, args, classtype=None):
        """Rebuild this node around new children.

        Args:
            args: The new child (the variable).
            classtype: A class to build instead, when a walker asks for one.

        Returns:
            The rebuilt node.
        """
        return _Active(args) if classtype is None else classtype(args)

    def _to_string(self, values, verbose, smap):
        """Render as a call, e.g. ``active(ship[1,2])``."""
        return f"active({values[0]})"


# ── permutations ────────────────────────────────────────────────────────────

class Permutation:
    """A permutation declared by [`permutation`][quicopt.structured.permutation]: items in as many slots,
    both counted from 0."""

    def __init__(self, name, record):
        """Wrap the declaration ``record`` of the permutation ``name``; use
        [`permutation`][quicopt.structured.permutation] to declare one."""
        self._name = name
        self._record = record

    @property
    def name(self):
        """The name the permutation was declared under."""
        return self._name

    @property
    def n_items(self):
        """How many items, and so how many slots."""
        return self._record.n_items

    def item_at(self, slot):
        """The item in slot ``slot``, an expression whose value the service chooses.

        For a route whose items are stops, ``route.item_at(0)`` is the first stop
        visited. Index a [`LookupTable`][quicopt.structured.LookupTable] with it to read data about that item.

        ``slot`` is a whole number from 0 to ``n_items - 1``.

        Raises:
            TypeError: If ``slot`` is not a whole number.
            ValueError: If ``slot`` is outside ``0..n_items - 1``.
        """
        return _Part("item_at", self._name, self._position(slot, "slot", "item_at"))

    def slot_of(self, item):
        """The slot item ``item`` is placed in, an expression whose value the
        service chooses.

        For a route whose items are stops, ``route.slot_of(2)`` is when stop 2 is
        visited.

        ``item`` is a whole number from 0 to ``n_items - 1``.

        Raises:
            TypeError: If ``item`` is not a whole number.
            ValueError: If ``item`` is outside ``0..n_items - 1``.
        """
        return _Part("slot_of", self._name, self._position(item, "item", "slot_of"))

    def precede(self, before, after):
        """Require item ``before`` to be in an earlier slot than item ``after``.

        A pickup before its delivery, or a job before the job that needs its
        output. The service only considers arrangements that meet every such
        requirement, so every solution meets it. Returns the permutation, so calls
        can be chained.

        Raises:
            TypeError: If an item is not a whole number.
            ValueError: If an item is outside ``0..n_items - 1``, the two are the
                same item, or the requirement contradicts the ones already given.
        """
        a = self._position(before, "item", "precede")
        b = self._position(after, "item", "precede")
        if a == b:
            raise ValueError("an item cannot precede itself")
        edges = self._record.precede + [(a, b)]
        if _has_cycle(edges, self.n_items):
            raise ValueError(f"requiring {a} before {b} contradicts the requirements already "
                             f"given for '{self._name}'")
        self._record.precede.append((a, b))
        return self

    def _position(self, k, what, caller):
        """Check a slot or item number against this permutation.

        Args:
            k: The number given.
            what: ``"slot"`` or ``"item"``, for the message.
            caller: The method's name, for the message.

        Returns:
            int: ``k``.

        Raises:
            TypeError: If ``k`` is not a whole number.
            ValueError: If ``k`` is outside ``0..n_items - 1``.
        """
        if not _is_whole(k):
            raise TypeError(f"{caller}: a {what} is a whole number, got {k!r}")
        if not 0 <= k < self.n_items:
            raise ValueError(f"{caller}: '{self._name}' has {what}s 0 to {self.n_items - 1}, "
                             f"got {k}")
        return int(k)

    def __repr__(self):
        """A short description: name, size and how many precedences."""
        k = len(self._record.precede)
        extra = f", {k} precedence{'s' if k != 1 else ''}" if k else ""
        return f"<permutation '{self._name}': {self.n_items} items in {self.n_items} slots{extra}>"


def permutation(model, name, n_items, start=None):
    """Declare an ordering as a decision: a permutation of ``n_items`` items over
    as many slots, on the Pyomo model ``model``.

    ``name`` identifies the permutation in the result, as
    ``result.structures[name]``. ``n_items`` is at least 2.

    ``start`` is the arrangement the service's search starts from: ``start[i]`` is
    the slot of item ``i``, so it holds each of ``0..n_items - 1`` once. Left out,
    item ``i`` starts in slot ``i``.

    Returns the [`Permutation`][quicopt.structured.Permutation], to read with ``item_at`` and ``slot_of`` and
    to constrain with ``precede``.

    Raises:
        TypeError: If ``model`` is not a Pyomo model (a PuLP or MathOpt model, or
            a block inside a Pyomo model, is refused), ``name`` is not a string or
            ``n_items`` not a whole number.
        ValueError: If ``n_items`` is below 2, ``name`` is already declared on the
            model, or ``start`` is not an arrangement of ``0..n_items - 1``.
    """
    if not _is_whole(n_items):
        raise TypeError(f"a permutation has a whole number of items, got {n_items!r}")
    if n_items < 2:
        raise ValueError(f"a permutation has at least 2 items, got {n_items}")
    n_items = int(n_items)
    start = [] if start is None else list(start)
    if start and not (all(_is_whole(s) for s in start) and sorted(start) == list(range(n_items))):
        raise ValueError(f"'{name}': start must hold each of 0 to {n_items - 1} once "
                         "(start[i] is the slot item i starts in)")
    record = _declare(model, name)
    perm = _PermutationRecord(n_items, [int(s) for s in start])
    record.permutations[name] = perm
    return Permutation(name, perm)


# ── partitions ──────────────────────────────────────────────────────────────

class Partition:
    """A partition declared by [`partition`][quicopt.structured.partition]: items split into parts, both
    counted from 0."""

    def __init__(self, name, record):
        """Wrap the declaration ``record`` of the partition ``name``; use
        [`partition`][quicopt.structured.partition] to declare one."""
        self._name = name
        self._record = record

    @property
    def name(self):
        """The name the partition was declared under."""
        return self._name

    @property
    def n_items(self):
        """How many items."""
        return self._record.n_items

    @property
    def n_parts(self):
        """How many parts."""
        return self._record.n_parts

    def part_of(self, item):
        """The part item ``item`` is in, an expression whose value the service
        chooses, from 0 to ``n_parts - 1``.

        For people split into teams, ``teams.part_of(2)`` is the team of person 2.
        Index a [`LookupTable`][quicopt.structured.LookupTable] with it to read data about that part.

        Raises:
            TypeError: If ``item`` is not a whole number.
            ValueError: If ``item`` is outside ``0..n_items - 1``.
        """
        return _Part("part_of", self._name,
                     _position(self, item, self.n_items, "item", "part_of"))

    def size_of(self, part):
        """How many items part ``part`` holds, an expression whose value the
        service chooses, from 0 to ``n_items``.

        Raises:
            TypeError: If ``part`` is not a whole number.
            ValueError: If ``part`` is outside ``0..n_parts - 1``.
        """
        return _Part("size_of", self._name,
                     _position(self, part, self.n_parts, "part", "size_of"))

    def __repr__(self):
        """A short description: name, items and parts."""
        return f"<partition '{self._name}': {self.n_items} items in {self.n_parts} parts>"


def _position(structure, k, extent, what, caller):
    """Check a position read from a structure.

    Args:
        structure: The structure, for the message.
        k: The number given.
        extent: How many positions there are.
        what: ``"item"`` or ``"part"``, for the message.
        caller: The method's name, for the message.

    Returns:
        int: ``k``.

    Raises:
        TypeError: If ``k`` is not a whole number.
        ValueError: If ``k`` is outside ``0..extent - 1``.
    """
    if not _is_whole(k):
        raise TypeError(f"{caller}: a {what} is a whole number, got {k!r}")
    if not 0 <= k < extent:
        raise ValueError(f"{caller}: '{structure.name}' has {what}s 0 to {extent - 1}, got {k}")
    return int(k)


def partition(model, name, n_items, n_parts, start=None):
    """Declare a split of ``n_items`` items into ``n_parts`` parts as a decision,
    on the Pyomo model ``model``: every item goes into exactly one part, and a part
    may stay empty.

    People into teams, jobs onto machines, products into warehouses: in each, the
    model reads the part an item is in through [`Partition.part_of`][quicopt.structured.Partition.part_of] and how
    many items a part holds through [`Partition.size_of`][quicopt.structured.Partition.size_of]. Items and parts
    count from 0.

    ``name`` identifies the partition in the result, as
    ``result.structures[name]``, a [`Grouping`][quicopt.client.Grouping]. ``n_items``
    is at least 1 and ``n_parts`` at least 2. ``start`` is the split the service's
    search starts from: ``start[i]`` is the part of item ``i``. Left out, the
    service deals the items out over the parts in turn.

    Raises:
        TypeError: If ``model`` is not a Pyomo model, ``name`` is not a string, or
            ``n_items`` or ``n_parts`` is not a whole number.
        ValueError: If ``n_items`` is below 1 or ``n_parts`` below 2, ``name`` is
            already declared on the model, or ``start`` does not give each item a
            part.
    """
    for (what, k, least) in (("items", n_items, 1), ("parts", n_parts, 2)):
        if not _is_whole(k):
            raise TypeError(f"a partition has a whole number of {what}, got {k!r}")
        if k < least:
            raise ValueError(f"a partition has at least {least} {what[:-1] if least == 1 else what}, got {k}")
    n_items, n_parts = int(n_items), int(n_parts)
    start = [] if start is None else list(start)
    if start and not (len(start) == n_items
                      and all(_is_whole(b) and 0 <= b < n_parts for b in start)):
        raise ValueError(f"'{name}': start gives each of the {n_items} items a part from 0 to "
                         f"{n_parts - 1} (start[i] is the part item i starts in)")
    record = _declare(model, name)
    decl = _PartitionRecord(n_items, n_parts, [int(b) for b in start])
    record.partitions[name] = decl
    return Partition(name, decl)


def _has_cycle(edges, n_items):
    """Whether ``(before, after)`` requirements over items ``0..n_items - 1``
    contradict each other: Kahn's algorithm leaves an item unplaced exactly when
    they do.

    Args:
        edges: The requirements, as ``(before, after)`` pairs.
        n_items: How many items.

    Returns:
        bool: True if the requirements contain a cycle.
    """
    indegree = [0] * n_items
    for (_, b) in edges:
        indegree[b] += 1
    ready = [i for i in range(n_items) if indegree[i] == 0]
    placed = 0
    while ready:
        x = ready.pop()
        placed += 1
        for (a, b) in edges:
            if a == x:
                indegree[b] -= 1
                if indegree[b] == 0:
                    ready.append(b)
    return placed < n_items


# ── lookup tables ───────────────────────────────────────────────────────────

class LookupTable:
    """A table of data declared by [`lookup_table`][quicopt.structured.lookup_table], to be indexed with
    expressions: ``table[i]`` for a table of one dimension, ``table[i, j]`` for
    two, positions counted from 0."""

    def __init__(self, name, values):
        """Wrap the stored entries ``values`` of the table ``name``; use
        [`lookup_table`][quicopt.structured.lookup_table] to declare one."""
        self._name = name
        self._values = values

    @property
    def name(self):
        """The name the table was declared under."""
        return self._name

    @property
    def shape(self):
        """The table's extent: ``(a,)`` for one dimension, ``(a, b)`` for two."""
        v = self._values
        return (len(v), len(v[0])) if isinstance(v[0], list) else (len(v),)

    def __getitem__(self, key):
        """The entry at a position the service chooses.

        Each index is a whole number or an expression, such as a permutation's
        ``item_at`` or an integer variable. With only whole numbers it is the
        entry itself, a plain number. Otherwise it is an expression, whose value
        is the entry at the positions the service chooses: an index value that
        is not a whole number is rounded, and one outside the table is moved to
        its nearest end.

        Raises:
            TypeError: If the number of indices does not match the table, or an
                index is neither a whole number nor a Pyomo expression.
            ValueError: If a whole-number index is outside the table.
        """
        shape = self.shape
        key = key if isinstance(key, tuple) else (key,)
        if len(key) != len(shape):
            usage = f"{self._name}[i]" if len(shape) == 1 else f"{self._name}[i, j]"
            raise TypeError(f"'{self._name}' is read at {len(shape)} "
                            f"position{'s' if len(shape) != 1 else ''}: {usage}")
        for (k, extent) in zip(key, shape):
            if _is_whole(k):
                if not 0 <= k < extent:
                    raise ValueError(f"'{self._name}': a whole-number index runs from 0 to "
                                     f"{extent - 1}, got {k}")
            elif isinstance(k, (bool, Real)) or not hasattr(k, "is_potentially_variable"):
                raise TypeError(f"'{self._name}': an index is a whole number or a Pyomo "
                                f"expression, got {type(k).__name__} {k!r}")
        if all(_is_whole(k) for k in key):
            entry = self._values
            for k in key:
                entry = entry[int(k)]
            return entry
        return _Lookup(self._name, [int(k) if _is_whole(k) else k for k in key])

    def __repr__(self):
        """A short description: name and extent."""
        return f"<lookup table '{self._name}': {' x '.join(str(a) for a in self.shape)}>"


def lookup_table(model, name, values):
    """Declare a table of data that can be indexed by decisions.

    A model sometimes needs a number from a table at a position that is not known
    yet: the distance between the first and the second stop of a route whose
    ordering the service is still choosing, or the price of an option an integer
    variable picks. A Python list cannot be indexed this way; the table returned
    here can.

    ``values`` is a sequence of numbers (a table of one dimension) or a sequence of
    equally long sequences of numbers (two dimensions), such as a list of lists or
    a NumPy array. ``name`` must not already be declared on the model.

    The whole table is sent with the model, so a very large table makes the
    request large.

    Raises:
        TypeError: If ``model`` is not a Pyomo model (a PuLP or MathOpt model, or
            a block inside a Pyomo model, is refused), or ``values`` is not one of
            the shapes above.
        ValueError: If it is empty, ragged or holds a NaN, or ``name`` is already
            declared.
    """
    table = _as_table(name, values)
    record = _declare(model, name)
    record.tables[name] = table
    return LookupTable(name, table)


def _as_table(name, values):
    """Copy table data into plain lists of floats, checking its shape.

    Args:
        name: The table's name, for the message.
        values: The data as given.

    Returns:
        list: A list of floats, or a list of equally long lists of floats.

    Raises:
        TypeError: If the data is not one or two dimensions of numbers.
        ValueError: If it is empty, ragged or holds a NaN.
    """
    rows = values.tolist() if hasattr(values, "tolist") else values
    try:
        rows = list(rows)
    except TypeError:
        raise TypeError(f"'{name}': a lookup table is a sequence of numbers or of "
                        f"rows of numbers, got {type(values).__name__}") from None
    if not rows:
        raise ValueError(f"'{name}': a lookup table has at least one entry")
    if all(_is_number(x) for x in rows):
        table = [float(x) for x in rows]
    elif all(not _is_number(r) and hasattr(r, "__iter__") for r in rows):
        table = [[float(x) if _is_number(x) else _not_a_number(name, x) for x in r] for r in rows]
        if not table[0] or any(len(r) != len(table[0]) for r in table):
            raise ValueError(f"'{name}': the rows of a lookup table are non-empty and "
                             "equally long")
    else:
        raise TypeError(f"'{name}': a lookup table is a sequence of numbers or of rows of "
                        "numbers, not a mix")
    flat = table if not isinstance(table[0], list) else [x for r in table for x in r]
    if any(x != x for x in flat):
        raise ValueError(f"'{name}': a lookup table cannot hold NaN")
    return table


def _not_a_number(name, x):
    """Refuse an entry that is not a number.

    Args:
        name: The table's name, for the message.
        x: The offending entry.

    Raises:
        TypeError: Always.
    """
    raise TypeError(f"'{name}': a lookup table holds numbers, got {x!r}")


def _is_number(x):
    """Whether ``x`` is a plain real number (and not a bool, which is not data here).

    Args:
        x: Any object.

    Returns:
        bool: True for an ``int``/``float`` (or NumPy scalar) other than a bool.
    """
    return isinstance(x, Real) and not isinstance(x, bool)


def _is_whole(x):
    """Whether ``x`` is a whole number given as an integer (not a bool, not a float).

    Args:
        x: Any object.

    Returns:
        bool: True for an ``int`` (or NumPy integer) other than a bool.
    """
    return isinstance(x, Integral) and not isinstance(x, bool)


# ── variables that are 0 or within their bounds; "in use"; blocks ───────────

def semicontinuous(var):
    """Let a continuous Pyomo variable be 0 as well as any value within its bounds,
    and nothing in between: a delivery of nothing, or of at least a minimum load.

    ``var`` is a variable (``m.x``) or an indexed one (``m.ship``, every element).
    Its bounds are the range it takes when it is not 0; both must be finite, and
    when 0 already lies within them nothing changes. Use [`semiinteger`][quicopt.structured.semiinteger] for an
    integer variable, and [`active`][quicopt.structured.active] to charge a fee whenever the variable is
    not 0.

    Raises:
        TypeError: If ``var`` is not a Pyomo variable, or is integer.
    """
    _mark_holes(var, integer=False, caller="semicontinuous")


def semiinteger(var):
    """Let an integer Pyomo variable be 0 as well as any whole number within its
    bounds, and nothing in between: a batch of none, or of 10 to 25 units.

    ``var`` is a variable (``m.x``) or an indexed one (``m.ship``, every element),
    declared with ``within=pyo.Integers`` (or ``NonNegativeIntegers``, …). Its
    bounds are the range it takes when it is not 0; both must be finite. See
    [`semicontinuous`][quicopt.structured.semicontinuous] for a continuous variable.

    Raises:
        TypeError: If ``var`` is not a Pyomo variable, or is not integer, or is
            binary.
    """
    _mark_holes(var, integer=True, caller="semiinteger")


def _mark_holes(var, integer, caller):
    """Record every variable in ``var`` as 0 or within its bounds, on its model.

    Args:
        var: A Pyomo ``VarData`` or indexed ``Var``.
        integer: Whether the variables must be integer (else continuous).
        caller: The public function's name, for the message.

    Raises:
        TypeError: If a member is not a Pyomo variable or has the wrong domain.
    """
    for v in _members(var, caller):
        if v.is_binary():
            raise TypeError(f"{caller}: '{v.name}' is binary, which is 0 or 1 already")
        if v.is_integer() != integer:
            want = "an integer" if integer else "a continuous"
            other = "semicontinuous" if integer else "semiinteger"
            raise TypeError(f"{caller}: '{v.name}' is not {want} variable; use {other} for it")
        _record(v.model()).holes[id(v)] = v


def _members(var, caller):
    """The variables in ``var``: itself, or every element of an indexed ``Var``.

    Args:
        var: A Pyomo ``VarData`` or indexed ``Var``.
        caller: The public function's name, for the message.

    Returns:
        list: The ``VarData`` objects.

    Raises:
        TypeError: If ``var`` is not a Pyomo variable.
    """
    if not _is_pyomo_var(var):
        raise TypeError(f"{caller}: expected a Pyomo variable, got {var!r}")
    return list(var.values()) if var.is_indexed() else [var]


def _is_pyomo_var(x):
    """Whether ``x`` is a Pyomo variable: a single one (``m.x``, ``m.ship[1]``) or
    an indexed family (``m.ship``).

    Args:
        x: Anything.

    Returns:
        bool: True for a Pyomo ``Var`` component or one of its elements.
    """
    return getattr(x, "ctype", None) is Var


def active(var):
    """Whether the variable ``var`` is in use: an expression that is 1 when
    ``var`` is not 0, and 0 when it is.

    A fee due whenever a delivery is made is ``fee * active(m.ship[w, c])``. On a
    variable that is 0 or within its bounds ([`semicontinuous`][quicopt.structured.semicontinuous],
    [`semiinteger`][quicopt.structured.semiinteger]), such a fee keeps a linear model linear, so the service
    solves it exactly.

    Raises:
        TypeError: If ``var`` is not a single Pyomo variable.
    """
    if not _is_pyomo_var(var) or var.is_indexed():
        raise TypeError(f"active: expected a single Pyomo variable such as m.x or m.ship[1], "
                        f"got {var!r}")
    return _Active([var])


def block(model, name, vars):
    """Declare a block: the variables of one subproblem of the model, such as one
    customer's supply or one project's staffing.

    The constraints over a block's own variables alone, such as that customer's
    demand, hold in every solution the service returns. A block changes nothing
    else about the model; it tells the service which variables belong together.

    ``vars`` is a list of variables, indexed ones included (every element).
    ``name`` must be new among the model's blocks. A variable belongs to at most
    one block, and a block's variables need finite bounds.

    Raises:
        TypeError: If ``model`` is not a Pyomo model, ``name`` not a string, or a
            member not a Pyomo variable.
        ValueError: If the block is empty, its name is taken, a variable is listed
            twice or is in another block, or a variable belongs to another model.
    """
    record = _record(model)
    _check_name(name)
    if name in record.blocks:
        raise ValueError(f"block '{name}' is already declared on this model")
    members = [v for x in vars for v in _members(x, "block")]
    if not members:
        raise ValueError(f"block '{name}' has no variables")
    owner = {id(v): other for (other, vs) in record.blocks.items() for v in vs}
    seen = set()
    for v in members:
        if v.model() is not model:
            raise ValueError(f"block '{name}': '{v.name}' belongs to another model")
        if id(v) in seen:
            raise ValueError(f"block '{name}' lists '{v.name}' twice")
        if id(v) in owner:
            raise ValueError(f"'{v.name}' is already in block '{owner[id(v)]}'; a variable "
                             "belongs to at most one block")
        seen.add(id(v))
    record.blocks[name] = members
