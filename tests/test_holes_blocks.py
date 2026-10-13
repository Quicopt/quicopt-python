# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
Variables that are 0 or within their bounds, ``active``, blocks, partitions and
Pyomo's ``Expr_if``, as ``quicopt.pyomo.import_model`` turns them into a
``Program``.

A partition counts items and parts from 0 here and from 1 in the ``Program``,
except a part's size, which is a count and is never shifted; the partition tests
are where an off-by-one would hide.

Self-contained: no network, no solver, no reference codec. What the service makes
of these models is checked elsewhere.

Skips unless the ``[pyomo]`` extra is installed:
    pip install -e '.[pyomo]'
Run with:
    python3 tests/test_holes_blocks.py         # or: pytest tests/
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                       # importable without an install

from quicopt import (Apply, Const, Var, StructRef, PartitionDecl,          # noqa: E402
                     INTEGER, SEMICONTINUOUS, SEMIINTEGER, encode)

try:
    import pyomo.environ as pyo
    from quicopt.pyomo import import_model
    from quicopt.structured import (semicontinuous, semiinteger, active, block, partition,
                                    minimum, maximum, permutation)
    from quicopt import stochastic
    from quicopt.stochastic import Normal, set_distribution
except ImportError:                                         # the extra is absent
    pyo = None


def _raises(exc, fn, *args, **kwargs):
    """Call ``fn`` and return the message of the ``exc`` it must raise.

    Raises:
        AssertionError: If ``fn`` returns, or raises something else.
    """
    try:
        fn(*args, **kwargs)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def _skip(name):
    """Report and skip a test when the ``[pyomo]`` extra is absent.

    Returns:
        bool: True when the test should be skipped.
    """
    if pyo is None:
        print(f"skip {name}: pyomo not installed")
        return True
    return False


def _warehouses():
    """The docs' example: three warehouses deliver to three customers, each
    delivery none or 10 to 25 units, 5 per delivery made, and an opening fee for
    each warehouse that delivers at all."""
    demand = [40, 30, 20]
    cost = [[1, 3, 4], [3, 1, 3], [4, 2, 1]]
    opening = [60, 50, 40]
    m = pyo.ConcreteModel()
    m.ship = pyo.Var(range(3), range(3), within=pyo.NonNegativeIntegers, bounds=(10, 25))
    semiinteger(m.ship)
    for c in range(3):
        block(m, f"customer_{c}", [m.ship[w, c] for w in range(3)])
    m.served = pyo.Constraint(range(3), rule=lambda m, c: sum(m.ship[w, c] for w in range(3)) >= demand[c])
    m.cost = pyo.Objective(expr=
        sum(cost[w][c] * m.ship[w, c] for w in range(3) for c in range(3))
        + sum(5 * active(m.ship[w, c]) for w in range(3) for c in range(3))
        + sum(pyo.Expr_if(IF=sum(m.ship[w, c] for c in range(3)) >= 1, THEN=opening[w], ELSE=0)
              for w in range(3)))
    return m


def _heads(e, out=None):
    """Every operator name in the expression ``e``, depth first."""
    out = [] if out is None else out
    if isinstance(e, Apply):
        out.append(e.op)
        for a in e.args:
            _heads(a, out)
    return out


def test_a_hole_becomes_its_domain_and_starts_at_zero():
    if _skip("test_a_hole_becomes_its_domain_and_starts_at_zero"):
        return
    m = pyo.ConcreteModel()
    m.q = pyo.Var(within=pyo.Integers, bounds=(10, 25))
    m.r = pyo.Var(bounds=(2.5, 8))
    m.s = pyo.Var(bounds=(1, 4), initialize=3)
    semiinteger(m.q)
    semicontinuous(m.r)
    semicontinuous(m.s)
    m.o = pyo.Objective(expr=m.q + m.r + m.s)
    d = {v.name: v for v in import_model(m).vars}
    assert (d["x1"].domain, d["x1"].lower, d["x1"].upper, d["x1"].start) == (SEMIINTEGER, 10.0, 25.0, 0.0)
    assert (d["x2"].domain, d["x2"].lower, d["x2"].upper, d["x2"].start) == (SEMICONTINUOUS, 2.5, 8.0, 0.0)
    assert d["x3"].start == 3.0                                 # a value the model gives is kept


def test_an_indexed_variable_is_marked_element_by_element():
    if _skip("test_an_indexed_variable_is_marked_element_by_element"):
        return
    m = _warehouses()
    prog = import_model(m)
    assert [v.domain for v in prog.vars] == [SEMIINTEGER] * 9


def test_a_fixed_hole_is_an_ordinary_pin():
    if _skip("test_a_fixed_hole_is_an_ordinary_pin"):
        return
    m = pyo.ConcreteModel()
    m.q = pyo.Var(within=pyo.Integers, bounds=(10, 25))
    semiinteger(m.q)
    m.q.fix(12)
    m.o = pyo.Objective(expr=m.q)
    v = import_model(m).vars[0]
    assert (v.domain, v.lower, v.upper) == (INTEGER, 12.0, 12.0)


def test_hole_arguments_are_checked():
    if _skip("test_hole_arguments_are_checked"):
        return
    m = pyo.ConcreteModel()
    m.i = pyo.Var(within=pyo.Integers, bounds=(1, 5))
    m.c = pyo.Var(bounds=(1, 5))
    m.b = pyo.Var(within=pyo.Binary)
    m.free = pyo.Var(bounds=(1, None))
    assert "semiinteger" in _raises(TypeError, semicontinuous, m.i)
    assert "semicontinuous" in _raises(TypeError, semiinteger, m.c)
    assert "binary" in _raises(TypeError, semiinteger, m.b)
    assert "Pyomo variable" in _raises(TypeError, semiinteger, 3)
    semicontinuous(m.free)
    m.o = pyo.Objective(expr=m.free)
    assert "finite" in _raises(ValueError, import_model, m)


def test_active_is_the_indicator():
    if _skip("test_active_is_the_indicator"):
        return
    m = pyo.ConcreteModel()
    m.q = pyo.Var(within=pyo.Integers, bounds=(10, 25))
    semiinteger(m.q)
    m.o = pyo.Objective(expr=m.q + 30 * active(m.q))
    obj = import_model(m).objective
    assert Apply("indicator", [Var("x1")]) in obj.args[1].args, obj
    assert "single Pyomo variable" in _raises(TypeError, active, m.q + 1)
    m.ship = pyo.Var([1, 2])
    assert "single Pyomo variable" in _raises(TypeError, active, m.ship)


def test_blocks_keep_their_order_and_name_wire_variables():
    if _skip("test_blocks_keep_their_order_and_name_wire_variables"):
        return
    prog = import_model(_warehouses())
    # ship[w, c] is variable 3w + c + 1 in Pyomo's order; customer c's block
    # lists warehouses 0, 1, 2 in the order given
    assert prog.blocks == {f"customer_{c}": [Var(f"x{3 * w + c + 1}") for w in range(3)] for c in range(3)}
    assert encode(prog) == encode(import_model(_warehouses()))


def test_block_arguments_are_checked():
    if _skip("test_block_arguments_are_checked"):
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], bounds=(0, 4))
    block(m, "a", [m.x[1]])
    assert "already declared" in _raises(ValueError, block, m, "a", [m.x[2]])
    assert "no variables" in _raises(ValueError, block, m, "b", [])
    assert "twice" in _raises(ValueError, block, m, "b", [m.x[2], m.x[2]])
    assert "already in block 'a'" in _raises(ValueError, block, m, "b", [m.x])
    other = pyo.ConcreteModel()
    other.y = pyo.Var(bounds=(0, 1))
    assert "another model" in _raises(ValueError, block, m, "b", [other.y])
    assert "non-empty string" in _raises(TypeError, block, m, 3, [m.x[2]])
    m.free = pyo.Var(bounds=(0, None))
    block(m, "c", [m.free])
    m.o = pyo.Objective(expr=m.free)
    assert "finite bounds" in _raises(ValueError, import_model, m)


def test_a_random_variable_is_not_a_block_member():
    if _skip("test_a_random_variable_is_not_a_block_member"):
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.d = pyo.Var(bounds=(0, 10))
    set_distribution(m, m.d, Normal(5, 1))
    block(m, "b", [m.x, m.d])
    m.o = pyo.Objective(expr=m.x + stochastic.expectation(m.d))
    assert "random variable" in _raises(ValueError, import_model, m)


def test_partition_counts_items_and_parts_from_one_on_the_wire():
    if _skip("test_partition_counts_items_and_parts_from_one_on_the_wire"):
        return
    m = pyo.ConcreteModel()
    teams = partition(m, "teams", 5, 2, start=[0, 1, 0, 1, 0])
    m.o = pyo.Objective(expr=teams.part_of(3) + teams.size_of(1))
    prog = import_model(m)
    assert prog.structures == {"teams": PartitionDecl(5, 2, start=[1, 2, 1, 2, 1])}
    part, size = prog.objective.args
    # the part of item 3 is the wire's part of item 4, less 1; the size of part 1
    # is the wire's size of part 2, unshifted
    assert part == Apply("-", [Apply("part_of", [Const(4.0), StructRef("teams")]), Const(1.0)])
    assert size == Apply("size_of", [Const(2.0), StructRef("teams")])


def test_partition_arguments_are_checked():
    if _skip("test_partition_arguments_are_checked"):
        return
    m = pyo.ConcreteModel()
    assert "at least 1 item" in _raises(ValueError, partition, m, "p", 0, 2)
    assert "at least 2 parts" in _raises(ValueError, partition, m, "p", 3, 1)
    assert "whole number" in _raises(TypeError, partition, m, "p", 2.5, 2)
    assert "start" in _raises(ValueError, partition, m, "p", 3, 2, start=[0, 2, 1])
    assert "start" in _raises(ValueError, partition, m, "p", 3, 2, start=[0, 1])
    p = partition(m, "p", 3, 2)
    assert "already declared" in _raises(ValueError, permutation, m, "p", 3)
    assert "0 to 2" in _raises(ValueError, p.part_of, 3)
    assert "0 to 1" in _raises(ValueError, p.size_of, 2)
    assert "whole number" in _raises(TypeError, p.part_of, 1.0)


def test_expr_if_becomes_a_step_or_an_indicator():
    if _skip("test_expr_if_becomes_a_step_or_an_indicator"):
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 9))
    m.y = pyo.Var(bounds=(0, 9))
    x, y = Var("x1"), Var("x2")
    one = Const(1.0)

    def imported(cond, then=1, other=0):
        m.del_component("o")
        m.o = pyo.Objective(expr=pyo.Expr_if(IF=cond, THEN=then, ELSE=other))
        return import_model(m).objective

    step = lambda e: Apply("step", [e])
    assert imported(m.x >= 3) == step(Apply("-", [x, Const(3.0)]))           # 1 when x − 3 ≥ 0
    assert imported(m.x <= m.y) == step(Apply("-", [y, x]))
    assert imported(m.x > 3) == Apply("-", [one, step(Apply("-", [Const(3.0), x]))])
    assert imported(m.x < m.y) == Apply("-", [one, step(Apply("-", [x, y]))])
    assert imported(m.x == 3) == Apply("-", [one, Apply("indicator", [Apply("-", [x, Const(3.0)])])])
    fee = imported(m.x >= 1, then=60, other=10)                              # 10 + 50·[x ≥ 1]
    assert "step" in _heads(fee) and Const(50.0) in _heads_and_consts(fee)
    assert "one comparison" in _raises(ValueError, imported, pyo.inequality(1, m.x, 3))


def _heads_and_consts(e, out=None):
    """Every operator name and constant in the expression ``e``, depth first."""
    out = [] if out is None else out
    if isinstance(e, Const):
        out.append(e)
    if isinstance(e, Apply):
        out.append(e.op)
        for a in e.args:
            _heads_and_consts(a, out)
    return out


def test_minimum_and_maximum_are_the_stochastic_ones():
    if _skip("test_minimum_and_maximum_are_the_stochastic_ones"):
        return
    assert minimum is stochastic.minimum and maximum is stochastic.maximum


def test_the_warehouse_model_reads_as_written():
    if _skip("test_the_warehouse_model_reads_as_written"):
        return
    obj = import_model(_warehouses()).objective
    heads = _heads(obj)
    assert heads.count("indicator") == 9 and heads.count("step") == 3


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all hole, block and partition tests passed")
