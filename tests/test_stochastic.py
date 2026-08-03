# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
The stochastic authoring surface: what it builds, and what it refuses to build.

A model under uncertainty can go wrong in a way a deterministic one cannot — it can
be *well-formed and mean something else*. A random variable that also carries bounds,
a scenario column of the wrong length, a probability read off the wrong side of a
comparison: each imports cleanly and answers a question nobody asked. These tests pin
that boundary, and the shape of what is built when it is respected.

Self-contained: no network, no solver, no reference codec. What the service makes of
these bytes is checked elsewhere; what is checked here is that the right bytes are
built in the first place.

Skips unless the ``[pyomo]`` extra is installed:
    pip install -e '.[pyomo]'
Run with:
    python3 tests/test_stochastic.py         # or: pytest tests/
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                       # importable without an install

from quicopt import Apply, Const, SourceRef, Parametric, Empirical    # noqa: E402

try:
    import pyomo.environ as pyo
    from quicopt.pyomo import import_model
    from quicopt.stochastic import (Normal, Distribution, set_distribution, set_scenarios,
                                    expectation, cvar, prob, maximum, minimum)
except ImportError:                                         # the extra is absent
    pyo = None


def _newsvendor():
    """Build the running example: one random demand, one order-up-to decision."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 200))
    m.demand = pyo.Var()
    set_distribution(m, m.demand, Normal(100.0, 15.0))
    set_scenarios(m, 512, seed=42)
    m.cost = pyo.Objective(expr=3 * m.x + 10 * expectation(maximum(m.demand - m.x, 0)),
                           sense=pyo.minimize)
    return m


def _raises(exc, fn, *args, **kwargs):
    """Run ``fn`` and return the stringified exception it must raise.

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


# ── what gets built ─────────────────────────────────────────────────────────

def test_source_leaves_the_decision_set():
    """A random variable is declared, not decided: it becomes a source, not a VarDecl."""
    if pyo is None:
        print("skip  test_source_leaves_the_decision_set (needs the [pyomo] extra)")
        return
    p = import_model(_newsvendor())
    assert [v.name for v in p.vars] == ["x1"], f"the source stayed a decision: {p.vars}"
    assert p.sources == {"demand": Parametric("normal", [Const(100.0), Const(15.0)])}, p.sources
    assert (p.scenarios, p.scenario_seed) == (512, 42), (p.scenarios, p.scenario_seed)


def test_source_uses_become_references_by_name():
    """Every use of a random variable becomes a reference to its declared name."""
    if pyo is None:
        print("skip  test_source_uses_become_references_by_name (needs the [pyomo] extra)")
        return
    p = import_model(_newsvendor())
    refs = []

    def walk(e):
        if isinstance(e, SourceRef):
            refs.append(e.name)
        elif isinstance(e, Apply):
            for a in e.args:
                walk(a)

    walk(p.objective)
    assert refs == ["demand"], f"expected one reference to the declared name, got {refs}"


def test_sources_do_not_renumber_the_decisions():
    """Which variables are random must not change what the others are called.

    Names are assigned across every variable before the random ones are set aside, so
    a source declared first leaves a gap (``x2``) rather than shifting ``x2`` down to
    ``x1``. A solution comes back keyed by these names, so a rule that depended on
    which variables happened to be random would silently mislabel the answer — and
    the Julia client numbers them the same way, so the two must agree.
    """
    if pyo is None:
        print("skip  test_sources_do_not_renumber_the_decisions (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.demand = pyo.Var()                                    # declared FIRST, and random
    m.x = pyo.Var(bounds=(0, 200))
    set_distribution(m, m.demand, Normal(100.0, 15.0))
    m.cost = pyo.Objective(expr=m.x + expectation(m.demand), sense=pyo.minimize)
    assert [v.name for v in import_model(m).vars] == ["x2"], "the decision was renumbered"


def test_empirical_column_travels_as_data():
    """An empirical column is already data, and is carried across unchanged."""
    if pyo is None:
        print("skip  test_empirical_column_travels_as_data (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 100))
    m.shock = pyo.Var()
    set_scenarios(m, 4)
    set_distribution(m, m.shock, Empirical([0.9, 1.0, 1.1, 1.3]))
    m.o = pyo.Objective(expr=expectation(m.shock * m.x), sense=pyo.minimize)
    p = import_model(m)
    assert p.sources == {"shock": Empirical([0.9, 1.0, 1.1, 1.3])}, p.sources
    assert p.scenario_seed == 1, "an unset seed must stay at the default, which is not sent"


def test_distribution_parameters_may_be_decisions():
    """A distribution whose mean is itself a decision needs no special machinery."""
    if pyo is None:
        print("skip  test_distribution_parameters_may_be_decisions (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.base = pyo.Var(bounds=(1, 5))
    m.price = pyo.Var()
    set_distribution(m, m.price, Normal(10 * m.base, 2.0))
    m.o = pyo.Objective(expr=expectation(m.price), sense=pyo.minimize)
    mean = import_model(m).sources["price"].params[0]
    assert isinstance(mean, Apply) and mean.op == "*", f"mean did not stay an expression: {mean}"


def test_any_catalog_head_is_reachable():
    """A distribution the service knows and this client does not is still writable."""
    if pyo is None:
        print("skip  test_any_catalog_head_is_reachable (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.u = pyo.Var()
    set_distribution(m, m.u, Distribution("uniform", 0.0, 1.0))
    m.o = pyo.Objective(expr=expectation(m.u * m.x), sense=pyo.minimize)
    assert import_model(m).sources["u"] == Parametric("uniform", [Const(0.0), Const(1.0)])


def test_deterministic_models_are_untouched():
    """A model that declares no uncertainty carries the stochastic defaults, i.e. nothing."""
    if pyo is None:
        print("skip  test_deterministic_models_are_untouched (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.o = pyo.Objective(expr=m.x, sense=pyo.minimize)
    p = import_model(m)
    assert (p.scenarios, p.scenario_seed, p.sources) == (1, 1, {}), "a plain model gained a layer"


# ── the aggregators ─────────────────────────────────────────────────────────

def test_prob_reads_the_comparison_off_either_side():
    """``a <= c``, ``c >= a`` and a comparison of two quantities all mean the right thing."""
    if pyo is None:
        print("skip  test_prob_reads_the_comparison_off_either_side (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 200))
    m.d = pyo.Var()
    set_distribution(m, m.d, Normal(100.0, 15.0))

    assert prob(m.d - m.x <= 0).getname() == "sfreq_leq"
    assert prob(m.d <= 120).args[1] == 120.0, "the threshold must be read off as a constant"

    # Pyomo normalizes every inequality to `lower <= upper`, so a ≥ arrives with the
    # threshold on the left — which is the one case that is a ≥ on the wire too.
    atleast = prob(m.d >= 120)
    assert (atleast.getname(), atleast.args[1]) == ("sfreq_geq", 120.0), atleast.args

    # Neither side constant ⇒ P(a ≤ b) = P(a − b ≤ 0), the form the wire can carry.
    both = prob(m.d <= m.x)
    assert both.getname() == "sfreq_leq" and both.args[1] == 0.0, both.args


def test_prob_refuses_what_has_no_scenario_meaning():
    """A strict comparison and an equality are refused, not quietly reinterpreted."""
    if pyo is None:
        print("skip  test_prob_refuses_what_has_no_scenario_meaning (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 200))
    m.d = pyo.Var()
    set_distribution(m, m.d, Normal(100.0, 15.0))
    assert "strict" in _raises(ValueError, prob, m.d - m.x < 0)
    _raises(TypeError, prob, m.d - m.x)                     # not a comparison at all


def test_cvar_level_must_be_a_plain_number_in_range():
    """The tail level is carried as a constant, so it cannot be a decision or nonsense."""
    if pyo is None:
        print("skip  test_cvar_level_must_be_a_plain_number_in_range (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 200))
    m.d = pyo.Var()
    set_distribution(m, m.d, Normal(100.0, 15.0))
    _raises(ValueError, cvar, m.d, 1.0)                     # not strictly inside (0, 1)
    _raises(ValueError, cvar, m.d, 0.0)
    _raises(TypeError, cvar, m.d, m.x)                      # a decision, not a level
    assert cvar(m.d, 0.95).args[1] == 0.95


def test_an_aggregator_refuses_to_be_evaluated_locally():
    """Asking Pyomo for the value of an expectation must fail, and say why."""
    if pyo is None:
        print("skip  test_an_aggregator_refuses_to_be_evaluated_locally (needs the [pyomo] extra)")
        return
    import logging

    m = pyo.ConcreteModel()
    m.d = pyo.Var(initialize=1.0)
    quiet = logging.getLogger("pyomo")                      # it logs the failure before raising
    was = quiet.level
    quiet.setLevel(logging.CRITICAL)
    try:
        msg = _raises(ValueError, pyo.value, expectation(m.d))
    finally:
        quiet.setLevel(was)
    assert "service" in msg, f"the message must point at where the value comes from: {msg}"


def test_maximum_folds_to_the_binary_catalog_operator():
    """Pyomo's max is n-ary and the catalog's is binary; the import nests them."""
    if pyo is None:
        print("skip  test_maximum_folds_to_the_binary_catalog_operator (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.y = pyo.Var(bounds=(0, 4))
    m.o = pyo.Objective(expr=maximum(m.x, m.y, 3), sense=pyo.minimize)
    obj = import_model(m).objective
    assert obj.op == "max" and len(obj.args) == 2, f"not folded to binary: {obj}"
    assert obj.args[0].op == "max", f"the fold nested the wrong way: {obj}"
    _raises(ValueError, maximum, m.x)                       # one thing has no maximum
    _raises(ValueError, minimum, m.x)


# ── what must not import ────────────────────────────────────────────────────

def test_a_source_may_not_also_be_declared_like_a_decision():
    """Bounds or a domain on a random variable is a contradiction, not a refinement."""
    if pyo is None:
        print("skip  test_a_source_may_not_also_be_declared_like_a_decision (needs the [pyomo] extra)")
        return
    for declared in ({"bounds": (0, 1)}, {"domain": pyo.Binary}):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 200))
        m.d = pyo.Var(**declared)
        set_distribution(m, m.d, Normal(100.0, 15.0))
        m.o = pyo.Objective(expr=m.x + expectation(m.d), sense=pyo.minimize)
        assert "'d'" in _raises(ValueError, import_model, m), f"not caught for {declared}"


def test_a_column_must_have_one_value_per_scenario():
    """A column of the wrong length is a mismatch the author has to resolve."""
    if pyo is None:
        print("skip  test_a_column_must_have_one_value_per_scenario (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 100))
    m.shock = pyo.Var()
    set_scenarios(m, 10)
    set_distribution(m, m.shock, Empirical([0.9, 1.0, 1.1, 1.3]))       # 4 values, 10 scenarios
    m.o = pyo.Objective(expr=expectation(m.shock * m.x), sense=pyo.minimize)
    msg = _raises(ValueError, import_model, m)
    assert "4" in msg and "10" in msg, f"the message must give both counts: {msg}"


def test_one_name_is_one_random_variable():
    """Two random variables under one name would silently become one."""
    if pyo is None:
        print("skip  test_one_name_is_one_random_variable (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.a = pyo.Var()
    m.b = pyo.Var()
    set_distribution(m, m.a, Normal(0.0, 1.0), name="shared")
    _raises(ValueError, set_distribution, m, m.b, Normal(0.0, 1.0), name="shared")


def test_a_distribution_attaches_only_to_a_variable():
    """The declaration is about a variable; anything else is a mistake worth naming."""
    if pyo is None:
        print("skip  test_a_distribution_attaches_only_to_a_variable (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    _raises(TypeError, set_distribution, m, 2 * m.x, Normal(0.0, 1.0))  # an expression
    _raises(TypeError, set_distribution, m, m.x, "normal")              # not a Distribution


def test_scenario_counts_below_one_are_refused():
    """Zero is not merely invalid: the wire cannot tell it from an absent field."""
    if pyo is None:
        print("skip  test_scenario_counts_below_one_are_refused (needs the [pyomo] extra)")
        return
    m = pyo.ConcreteModel()
    _raises(ValueError, set_scenarios, m, 0)
    _raises(ValueError, set_scenarios, m, 8, 0)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all stochastic tests passed")
