# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
The structured authoring surface: permutations and lookup tables in Pyomo.

The surface counts items, slots and table positions from 0, like Python; the
service counts them from 1. Every test that inspects a built ``Program`` is
therefore also a test of that shift, which is where an off-by-one would hide.

Self-contained: no network, no solver, no reference codec. What the service makes
of these bytes is checked elsewhere; what is checked here is that the right bytes
are built in the first place.

Skips unless the ``[pyomo]`` extra is installed:
    pip install -e '.[pyomo]'
Run with:
    python3 tests/test_structured.py         # or: pytest tests/
"""
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                       # importable without an install

from quicopt import (Apply, Const, Var, StructRef, TableRef, PermutationDecl,   # noqa: E402
                     Parametric, encode)

try:
    import pyomo.environ as pyo
    from quicopt.pyomo import import_model
    from quicopt.structured import permutation, lookup_table
    from quicopt.stochastic import Normal, set_distribution, set_scenarios, expectation
except ImportError:                                         # the extra is absent
    pyo = None


def _route():
    """The running example: five stops on a road, stop 3 visited before stop 0."""
    position = [0, 3, 1, 4, 2]
    m = pyo.ConcreteModel()
    dist = lookup_table(m, "dist", [[abs(a - b) for b in position] for a in position])
    route = permutation(m, "route", 5)
    route.precede(3, 0)
    m.length = pyo.Objective(
        expr=sum(dist[route.item_at(k), route.item_at(k + 1)] for k in range(4)),
        sense=pyo.minimize)
    return m, route, dist


def _raises(exc, fn, *args, **kwargs):
    """Run ``fn`` and return the message of the exception it must raise.

    Args:
        exc: The exception type expected.
        fn: The callable under test.
        *args: Its positional arguments.
        **kwargs: Its keyword arguments.

    Returns:
        str: The exception's message, for asserting on what it explains.

    Raises:
        AssertionError: If ``fn`` raises nothing, or raises something else.
    """
    try:
        fn(*args, **kwargs)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__} from {getattr(fn, '__name__', fn)}")


def _skip(name):
    """Report a skipped test when Pyomo is absent; True if it is."""
    if pyo is None:
        print(f"skip  {name} (needs the [pyomo] extra)")
    return pyo is None


def _item_at(slot, name):
    """``item_at`` as the service reads it: slot and result counted from 1."""
    return Apply("item_at", [Const(float(slot)), StructRef(name)])


# ── what gets built ─────────────────────────────────────────────────────────

def test_route_counts_from_one_on_the_wire():
    """The route's lookups read item_at at slots 1..5, with no shift left over."""
    if _skip("test_route_counts_from_one_on_the_wire"):
        return
    m, _, _ = _route()
    p = import_model(m)
    legs = [TableRef("dist", [_item_at(k, "route"), _item_at(k + 1, "route")]) for k in range(1, 5)]
    assert p.objective == Apply("+", legs), p.objective
    assert p.structures == {"route": PermutationDecl(5, precede=[(4, 1)])}, p.structures
    assert p.vars == [], p.vars
    assert p.params["dist"][(1, 2)] == 3.0 and len(p.params["dist"]) == 25


def test_part_outside_a_lookup_is_counted_from_zero():
    """Read on its own, a part is the service's value less one."""
    if _skip("test_part_outside_a_lookup_is_counted_from_zero"):
        return
    m = pyo.ConcreteModel()
    office = permutation(m, "office", 3)
    m.c = pyo.Constraint(expr=office.slot_of(2) <= 0)       # department 2 gets office 0
    m.o = pyo.Objective(expr=office.item_at(1))
    p = import_model(m)
    slot_of_2 = Apply("slot_of", [Const(3.0), StructRef("office")])
    assert p.objective == Apply("-", [_item_at(2, "office"), Const(1.0)]), p.objective
    assert p.constraints[0].f == Apply("-", [Const(0.0), Apply("-", [slot_of_2, Const(1.0)])]), \
        p.constraints[0].f


def test_start_and_precedence_shift_to_one():
    if _skip("test_start_and_precedence_shift_to_one"):
        return
    m = pyo.ConcreteModel()
    jobs = permutation(m, "jobs", 4, start=[1, 0, 3, 2])
    jobs.precede(1, 0).precede(0, 3)
    m.o = pyo.Objective(expr=jobs.slot_of(0))
    assert import_model(m).structures == {
        "jobs": PermutationDecl(4, start=[2, 1, 4, 3], precede=[(2, 1), (1, 4)])}


def test_lookup_at_an_integer_variable():
    """A variable index is shifted by adding one; a whole-number index reads the data."""
    if _skip("test_lookup_at_an_integer_variable"):
        return
    m = pyo.ConcreteModel()
    price = lookup_table(m, "price", [3.0, 1.0, 4.0, 1.5])
    m.choice = pyo.Var(domain=pyo.Integers, bounds=(0, 3))
    m.o = pyo.Objective(expr=price[m.choice])
    p = import_model(m)
    assert p.objective == TableRef("price", [Apply("+", [Var("x1"), Const(1.0)])]), p.objective
    assert p.params == {"price": {(1,): 3.0, (2,): 1.0, (3,): 4.0, (4,): 1.5}}
    assert price[2] == 4.0                                  # plain data, no expression
    assert price.shape == (4,)


def test_structures_combine_with_random_variables():
    """A route whose legs are scaled by a random delay carries both declarations."""
    if _skip("test_structures_combine_with_random_variables"):
        return
    m, route, dist = _route()
    m.delay = pyo.Var()
    set_distribution(m, m.delay, Normal(1.0, 0.1))
    set_scenarios(m, 64, seed=4)
    m.length.deactivate()
    m.expected = pyo.Objective(expr=expectation(m.delay * dist[route.item_at(0), route.item_at(1)]))
    p = import_model(m)
    assert set(p.structures) == {"route"} and p.sources == {
        "delay": Parametric("normal", [Const(1.0), Const(0.1)])}
    assert p.scenarios == 64
    encode(p)                                               # every node has a wire form


def test_parts_have_no_local_value():
    """Evaluating a part in Python is an error, not a silent number."""
    if _skip("test_parts_have_no_local_value"):
        return
    m, route, dist = _route()
    quiet = logging.getLogger("pyomo.core")                 # Pyomo logs the error it raises
    level = quiet.level
    quiet.setLevel(logging.CRITICAL)
    try:
        _raises(ValueError, pyo.value, route.item_at(0))
        _raises(ValueError, pyo.value, dist[route.item_at(0), 1])
    finally:
        quiet.setLevel(level)


def test_a_model_without_structures_is_unchanged():
    if _skip("test_a_model_without_structures_is_unchanged"):
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.o = pyo.Objective(expr=m.x)
    p = import_model(m)
    assert p.structures == {} and p.params == {}


# ── what gets refused ───────────────────────────────────────────────────────

def test_permutation_arguments_are_checked():
    if _skip("test_permutation_arguments_are_checked"):
        return
    m = pyo.ConcreteModel()
    _raises(ValueError, permutation, m, "p", 1)
    _raises(TypeError, permutation, m, "p", 2.5)
    _raises(ValueError, permutation, m, "p", 3, start=[0, 1, 1])
    _raises(ValueError, permutation, m, "p", 3, start=[1, 2, 3])        # counted from 0
    p = permutation(m, "p", 3)
    _raises(ValueError, permutation, m, "p", 4)                          # name taken
    _raises(ValueError, p.item_at, 3)
    _raises(TypeError, p.slot_of, 1.0)
    _raises(ValueError, p.precede, 1, 1)
    p.precede(0, 1).precede(1, 2)
    assert "contradicts" in _raises(ValueError, p.precede, 2, 0)


def test_start_must_respect_the_precedences():
    if _skip("test_start_must_respect_the_precedences"):
        return
    m = pyo.ConcreteModel()
    p = permutation(m, "p", 3, start=[0, 1, 2])
    p.precede(2, 0)
    m.o = pyo.Objective(expr=p.slot_of(0))
    assert "precede" in _raises(ValueError, import_model, m)


def test_table_arguments_are_checked():
    if _skip("test_table_arguments_are_checked"):
        return
    m = pyo.ConcreteModel()
    _raises(ValueError, lookup_table, m, "t", [])
    _raises(ValueError, lookup_table, m, "t", [[1.0, 2.0], [3.0]])
    _raises(ValueError, lookup_table, m, "t", [1.0, float("nan")])
    _raises(TypeError, lookup_table, m, "t", [1.0, [2.0]])
    t = lookup_table(m, "t", [[1.0, 2.0], [3.0, 4.0]])
    _raises(TypeError, t.__getitem__, 0)                                # two indices needed
    _raises(ValueError, t.__getitem__, (0, 2))
    _raises(TypeError, t.__getitem__, (0, 0.5))
    assert t[1, 0] == 3.0


def test_names_must_not_clash_with_variables():
    """A permutation named like a variable would mean two things on the wire."""
    if _skip("test_names_must_not_clash_with_variables"):
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))                                        # named x1 on the wire
    p = permutation(m, "x1", 2)
    m.o = pyo.Objective(expr=m.x + p.item_at(0))
    assert "rename" in _raises(ValueError, import_model, m)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all structured tests passed")
