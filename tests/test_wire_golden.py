# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
The byte-equality test — the encoder's correctness property.

For each fixture, ``quicopt.wire.encode`` must reproduce, byte for byte, the
committed golden bytes the Quicopt service expects (``tests/goldens/<name>.hex``).
Byte-equality is the sharpest check: identical bytes ⇒ identical decoded ``Program``
(the codec is injective), so a match proves structural fidelity. The goldens are
committed fixtures; the service's own codec is what they were captured from.

Run with no dependencies installed:
    python3 tests/test_wire_golden.py
or under pytest:
    pytest tests/
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                       # importable without an install

from quicopt import (Const, Param, Var, Apply, Reduce, SetRef, SourceRef,   # noqa: E402
                     StructRef, TableRef, PermutationDecl, PartitionDecl,
                     Zero, Nonneg, Indicator, Parametric, Empirical,
                     VarDecl, IndexSet, Constraint, Program,
                     CONTINUOUS, INTEGER, BINARY, SEMICONTINUOUS, SEMIINTEGER, encode)

GOLD = ROOT / "tests" / "goldens"


# Fixtures — the same Programs the service's codec encodes for the goldens.

def _scalar_nlp():                                          # min x1² + 1/x1, x1 ∈ [0.1, 10]
    x = Var("x1")
    return Program(
        vars=[VarDecl("x1", [], CONTINUOUS, 0.1, 10.0, 0.1)],
        objective=Apply("+", [Apply("^", [x, Const(2.0)]), Apply("/", [Const(1.0), x])]),
        sense="min")


def _bounds_zero_start():                                   # start=0, fix=0, max, Nonneg+Zero
    y = Var("y1")
    return Program(
        vars=[VarDecl("y1", [], CONTINUOUS, -1.0, 1.0, 0.0)],
        objective=y, sense="max",
        constraints=[Constraint(y, Nonneg(), []),
                     Constraint(Apply("-", [y, Const(0.5)]), Zero(), [])],
        fix={("y1", ()): 0.0})


def _indexed():                                             # families, indexing, canonical sort
    # Names are deliberately abstract: this fixture exercises the encoder's handling
    # of index sets, dependent sets, param tables and reductions — not any domain.
    sets = [IndexSet("S", [1, 2, 3])]
    indexed = {"a": {(1,): [10, 11], (3,): [30], (2,): [20]}}
    params = {"p": {(2,): 2.0, (1,): 0.0, (3,): 3.0}}
    vars = [
        VarDecl("α", ["S"], CONTINUOUS, -3.15, 3.15, 0.0),
        VarDecl("y", ["S"], CONTINUOUS, "p", 10.0, 1.0),        # lower = Param name ⇒ Bound::param
        VarDecl("z", ["S"], BINARY, 0.0, 1.0, 0.0)]
    obj = Reduce("+", "i", SetRef("S", ()),
                 Apply("*", [Param("p", ("i",)), Var("y", ("i",))]), None)
    # The filter is data-determined — it decides which terms exist, so it cannot
    # depend on a decision variable (the server rejects that at the boundary;
    # a variable-gated sum is written Σ z[j]·y[j] instead). p[1] = 0.0, so this
    # one genuinely drops a fibre.
    cons = [Constraint(
        Reduce("+", "j", SetRef("a", ("i",)), Var("y", ("j",)), Param("p", ("i",))),
        Nonneg(),
        [("i", SetRef("S", ()))])]
    fix = {("α", (1,)): 0.0}
    return Program(sets, indexed, params, vars, obj, "min", cons, fix)


def _indicator():                                           # u ⟹ x ≥ 0
    x = Var("x")
    u = Var("u")
    return Program(
        vars=[VarDecl("x", [], CONTINUOUS, -5.0, 5.0, 0.0),
              VarDecl("u", [], BINARY, 0.0, 1.0, 0.0)],
        objective=x, sense="min",
        constraints=[Constraint(x, Indicator(u, Nonneg()), [])])


def _stoch_parametric():                                    # sources, both scalars, aggregators
    # One random variable and a decision taken before it is seen: buy x, pay for
    # the mismatch, and hold the shortfall under 10% of scenarios.
    x, d = Var("x"), SourceRef("demand")
    return Program(
        vars=[VarDecl("x", [], CONTINUOUS, 0.0, 200.0, 0.0)],
        objective=Apply("+", [Apply("*", [Const(3.0), x]),
                              Apply("smean", [Apply("abs", [Apply("-", [x, d])])])]),
        sense="min",
        constraints=[Constraint(
            Apply("-", [Apply("sfreq_leq", [Apply("-", [d, x]), Const(0.0)]), Const(0.9)]),
            Nonneg(), [])],
        scenarios=512, scenario_seed=42,
        sources={"demand": Parametric("normal", [Const(100.0), Const(15.0)])})


def _stoch_empirical():                                     # both source kinds at once
    # A scenario column beside a distribution whose mean is read from data, and a
    # seed left at its default — so this fixture also pins the omitted field.
    # Sorted emission (price before shock) is not the insertion order, on purpose.
    x, price, shock = Var("x"), SourceRef("price"), SourceRef("shock")
    return Program(
        params={"μ": {(): 50.0}},
        vars=[VarDecl("x", [], CONTINUOUS, 0.0, 100.0, 0.0)],
        objective=Apply("+", [
            Apply("smean", [Apply("*", [shock, Apply("-", [price, x])])]),
            Apply("scvar", [Apply("-", [price, x]), Const(0.95)])]),
        sense="min",
        scenarios=4,
        sources={"shock": Empirical([0.9, 1.0, 1.1, 1.3]),
                 "price": Parametric("normal", [Param("μ"), Const(2.0)])})


def _distances(coords):
    """The dense distance table between points on a line, keyed from 1."""
    n_points = len(coords)
    return {(i + 1, j + 1): float(abs(coords[i] - coords[j]))
            for i in range(n_points) for j in range(n_points)}


def _struct_tour():                                         # every StructureDecl field, 2-D lookup
    # A lookup over item_at, a precedence and an explicit start. Like the two
    # stochastic fixtures, transcribed verbatim from the service's own corpus.
    tour = StructRef("tour")
    leg = lambda k: TableRef("dist", [Apply("item_at", [Const(float(k)), tour]),
                                      Apply("item_at", [Const(float(k + 1)), tour])])
    return Program(
        params={"dist": _distances([0, 3, 1, 4])},
        vars=[VarDecl("x", [], CONTINUOUS, 0.0, 1.0, 0.0)],
        objective=Apply("+", [leg(1), leg(2), leg(3), Var("x")]),
        sense="min",
        structures={"tour": PermutationDecl(4, start=[2, 1, 4, 3], precede=[(2, 1)])})


def _struct_assignment():                                   # slot_of, no optional fields
    # The same params table read two ways in one Program: through Param leaves and
    # through a lookup at decided positions.
    place = StructRef("place")
    at = lambda f: Apply("slot_of", [Const(float(f)), place])
    term = lambda f, g: Apply("*", [Param("flow", (f, g)), TableRef("dist", [at(f), at(g)])])
    return Program(
        params={"dist": _distances([0, 1, 3]),
                "flow": {(1, 2): 2.0, (1, 3): 5.0, (2, 3): 1.0}},
        vars=[VarDecl("y", [], CONTINUOUS, 0.0, 1.0, 0.0)],
        objective=Apply("+", [term(1, 2), term(1, 3), term(2, 3), Var("y")]),
        sense="min",
        structures={"place": PermutationDecl(3)})


def _struct_fixed():                                        # a pinned permutation, 1-D lookup
    order = StructRef("order")
    return Program(
        params={"cost": {(1,): 1.5, (2,): 0.5, (3,): 2.0}},
        vars=[VarDecl("z", [], CONTINUOUS, 0.0, 2.0, 1.0)],
        objective=Apply("+", [TableRef("cost", [Apply("slot_of", [Const(1.0), order])]),
                              Var("z")]),
        sense="min",
        constraints=[Constraint(Apply("-", [Var("z"), Apply("item_at", [Const(2.0), order])]),
                                Nonneg(), [])],
        fix={("z", ()): 1.0},
        structures={"order": PermutationDecl(3, start=[3, 1, 2], fixed=True)})


def _struct_stoch():                                        # a structure in a stochastic model
    tour = StructRef("tour")
    leg = TableRef("dist", [Apply("item_at", [Const(1.0), tour]),
                            Apply("item_at", [Const(2.0), tour])])
    return Program(
        params={"dist": _distances([0, 2, 5])},
        vars=[VarDecl("x", [], CONTINUOUS, 0.0, 1.0, 0.0)],
        objective=Apply("+", [Apply("smean", [Apply("*", [leg, SourceRef("f")])]), Var("x")]),
        sense="min",
        scenarios=8, scenario_seed=3,
        sources={"f": Parametric("normal", [Const(1.0), Const(0.1)])},
        structures={"tour": PermutationDecl(3)})


def _holes_blocks():                                        # holes, an indicator, a step and a min, two blocks
    # A semi-integer and a semicontinuous variable, a fixed charge through the
    # first one's indicator, a step and a min over sums, a binary, and two blocks:
    # one of scalars, one of the elements of an indexed family.
    q, r, s = Var("q"), Var("r"), Var("s")
    x = lambda i: Var("x", (i,))
    ind = lambda e: Apply("indicator", [e])
    return Program(
        sets=[IndexSet("I", [1, 2])],
        vars=[VarDecl("q", [], SEMIINTEGER, 10.0, 25.0, 0.0),
              VarDecl("r", [], SEMICONTINUOUS, 2.5, 8.0, 0.0),
              VarDecl("s", [], BINARY, 0.0, 1.0, 0.0),
              VarDecl("x", ["I"], INTEGER, 0.0, 4.0, 0.0)],
        objective=Apply("+", [
            Apply("*", [Const(1.5), q]), Apply("*", [Const(2.0), r]),
            Apply("*", [Const(30.0), ind(q)]),
            Apply("*", [Const(40.0), Apply("step", [
                Apply("-", [Apply("+", [x(1), x(2)]), Const(1.0)])])]),
            Apply("*", [Const(-5.0), Apply("min", [Const(1.0), Apply("+", [ind(q), ind(r)])])])]),
        sense="min",
        constraints=[Constraint(Apply("-", [Apply("+", [q, r]), Const(12.0)]), Nonneg(), []),
                     Constraint(Apply("-", [Apply("*", [Const(8.0), s]), r]), Nonneg(), []),
                     Constraint(Apply("-", [Apply("+", [x(1), x(2)]), Const(3.0)]), Nonneg(), [])],
        blocks={"supply": [q, r, s], "pair": [x(1), x(2)]})


def _partition():                                           # a live and a fixed partition
    teams, split = StructRef("teams"), StructRef("split")
    part = lambda i: Apply("part_of", [Const(float(i)), teams])
    size = lambda b, P: Apply("size_of", [Const(float(b)), P])
    return Program(
        params={"cost": {(1,): 2.0, (2,): 0.5}},
        vars=[VarDecl("y", [], CONTINUOUS, 0.0, 5.0, 0.0)],
        objective=Apply("+", [TableRef("cost", [part(1)]), TableRef("cost", [part(3)]),
                              Apply("^", [Apply("-", [size(1, teams), size(2, teams)]), Const(2.0)]),
                              Var("y")]),
        sense="min",
        constraints=[Constraint(Apply("-", [Var("y"), size(3, split)]), Nonneg(), [])],
        structures={"teams": PartitionDecl(5, 2, start=[1, 2, 1, 2, 1]),
                    "split": PartitionDecl(4, 3, start=[3, 1, 3, 2], fixed=True)})


FIXTURES = {
    "scalar_nlp": _scalar_nlp,
    "bounds_zero_start": _bounds_zero_start,
    "indexed": _indexed,
    "indicator": _indicator,
    "stoch_parametric": _stoch_parametric,
    "stoch_empirical": _stoch_empirical,
    "struct_tour": _struct_tour,
    "struct_assignment": _struct_assignment,
    "struct_fixed": _struct_fixed,
    "struct_stoch": _struct_stoch,
    "holes_blocks": _holes_blocks,
    "partition": _partition,
}


def _golden(name):
    return bytes.fromhex((GOLD / f"{name}.hex").read_text().strip())


def test_goldens():
    for name, build in FIXTURES.items():
        got, want = encode(build()), _golden(name)
        assert got == want, f"{name}: {got.hex()} != {want.hex()}"


if __name__ == "__main__":
    ok = True
    for name, build in FIXTURES.items():
        got, want = encode(build()), _golden(name)
        if got == want:
            print(f"PASS  {name:<20} {len(got):>4} bytes")
        else:
            ok = False
            print(f"FAIL  {name}\n   got  {got.hex()}\n   want {want.hex()}")
    sys.exit(0 if ok else 1)
