# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.stochastic — optimization under uncertainty, written in Pyomo.

Part of a model's data is often unknown when the decision has to be made: demand,
prices, yields, arrival times. Declare that data as random variables carrying
distributions, and the model is solved over a sample of scenarios drawn from them.

Two rules describe the whole surface:

- A Pyomo variable given a distribution by [`set_distribution`][quicopt.stochastic.set_distribution]
  is a random variable, not a decision variable. Every use of the variable references the same sample.
- An expression containing a random variable is itself random, and cannot serve as an
  objective or a constraint until an aggregator reduces it over the scenarios:
  [`expectation`][quicopt.stochastic.expectation] for the mean,
  [`cvar`][quicopt.stochastic.cvar] for the tail,
  [`prob`][quicopt.stochastic.prob] for a chance constraint.

[`set_scenarios`][quicopt.stochastic.set_scenarios] sets how many scenarios are drawn
and from which seed. Both belong to the model, so repeated solves see the same sample.

Example:
    Order ``x`` units at 3 apiece against demand you will only learn later, pay
    10 per unit of mismatch, and meet demand in at least 90% of scenarios:

    ```python
    import pyomo.environ as pyo
    from quicopt import Client
    from quicopt.stochastic import (Normal, set_distribution, set_scenarios,
                                    expectation, prob, maximum)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 200))              # decide now
    m.demand = pyo.Var()                        # learn later
    set_distribution(m, m.demand, Normal(100.0, 15.0))
    set_scenarios(m, 512, seed=42)

    m.cost = pyo.Objective(
        expr=3 * m.x + 10 * expectation(maximum(m.demand - m.x, 0)),
        sense=pyo.minimize)
    m.service = pyo.Constraint(expr=prob(m.demand - m.x <= 0) >= 0.9)

    result = Client().solve(m)
    print(result.solution)                      # {'x1': 107.9…}
    ```

    Every mention of ``m.demand`` refers to the same sample. Two independent random
    variables are two Pyomo variables under two names.

    ``maximum`` and ``minimum`` build the kinked expressions that price recourse:
    ``maximum(demand - stock, 0)`` is the shortfall, and zero when there is none.

    Models written this way are solved by the service. Attaching a local solver to
    one fails — an expectation has no value until the scenarios are drawn.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyomo.core.expr import numeric_expr as _ne
from pyomo.core.expr.relational_expr import InequalityExpression

from .ir import Empirical

__all__ = [
    "Distribution", "Normal", "Empirical",
    "set_distribution", "set_scenarios",
    "expectation", "cvar", "prob",
    "maximum", "minimum",
]


# ── distributions ───────────────────────────────────────────────────────────

class Distribution:
    """Any distribution the service supports, named by ``head``.

    Use :class:`Normal` where it fits; use this for a distribution that has no
    named class here yet, as in ``Distribution("uniform", 0.0, 1.0)``.

    A parameter may be a number or a deterministic Pyomo expression. An expression
    gives an *endogenous* distribution — one whose parameters depend on the
    decision, such as a demand whose mean rises with the price you set.
    """

    def __init__(self, head, *params):
        """Draw from the distribution ``head`` with the given parameters, each a
        number or a deterministic Pyomo expression."""
        self.head = str(head)
        self.params = params

    def __repr__(self):
        """A readable rendering: the head applied to its parameters."""
        return f"{self.head}({', '.join(str(p) for p in self.params)})"


class Normal(Distribution):
    """The normal distribution with mean ``mu`` and standard deviation ``sigma``."""

    def __init__(self, mu, sigma):
        """Declare a draw from ``N(mu, sigma)`` — the mean and the standard
        deviation, each a number or a deterministic Pyomo expression."""
        super().__init__("normal", mu, sigma)


# ── declaring a model's random variables ────────────────────────────────────

@dataclass
class _Stochastic:
    """What a model has declared about the uncertainty it faces.

    Held on the model itself, under a private attribute, and read by
    ``quicopt.pyomo.import_model`` on the way out.

    Attributes:
        sources: ``id(vardata) → (vardata, name, spec)``. Pyomo variables are
            unhashable (``==`` builds an expression rather than answering), so
            identity is the key; the variable is kept beside it both to hold that
            identity valid and to let the importer read its bounds.
        scenarios: How many scenarios are drawn.
        seed: The seed they are drawn from.
    """
    sources: dict
    scenarios: int
    seed: int


_SLOT = "_quicopt_stochastic"


def _declarations(model):
    """Read a model's stochastic declarations.

    Args:
        model: A Pyomo model.

    Returns:
        _Stochastic | None: The record, or ``None`` if the model declared no
        uncertainty — in which case it is an ordinary deterministic model.
    """
    return getattr(model, _SLOT, None)


def _declare(model):
    """Read a model's stochastic declarations, creating the record on first use.

    Args:
        model: A Pyomo model.

    Returns:
        _Stochastic: The record, defaulting to one scenario drawn from seed 1 —
        the degenerate instance, which is what "no uncertainty" means.
    """
    record = _declarations(model)
    if record is None:
        record = _Stochastic({}, 1, 1)
        setattr(model, _SLOT, record)
    return record


def set_distribution(model, v, dist, name=None):
    """Turn the Pyomo variable ``v`` into a random variable drawn from ``dist``.

    ``v`` is no longer a decision variable: the solver is handed its value rather
    than choosing it, and every use of ``v`` means the same sample within a scenario.

    ``dist`` is a :class:`Distribution` such as ``Normal(100.0, 15.0)``, or an
    :class:`~quicopt.ir.Empirical` column holding one observed value per scenario.
    ``name`` defaults to the variable's own name and is what identifies the random
    variable, so two independent ones need two names. Returns ``v``.

    Give ``v`` no bounds and no domain. Its distribution already says what values
    it takes, and a variable carrying both is rejected on import.

    Raises:
        TypeError: If ``v`` is not a Pyomo variable, or ``dist`` is neither a
            ``Distribution`` nor an ``Empirical`` column.
        ValueError: If ``name`` is already taken by a different random variable.
    """
    if not (hasattr(v, "is_variable_type") and v.is_variable_type()):
        raise TypeError(f"a distribution attaches to a Pyomo variable, not to {type(v).__name__}")
    if not isinstance(dist, (Distribution, Empirical)):
        raise TypeError("a distribution must be a Distribution (e.g. Normal(mu, sigma)) "
                        f"or an Empirical column, got {type(dist).__name__}")

    record = _declare(model)
    declared = str(v.name if name is None else name)
    for (other, other_name, _) in record.sources.values():
        if other is not v and other_name == declared:
            raise ValueError(f"the name '{declared}' already belongs to another random variable — "
                             "one name is one random variable, so independent ones need "
                             "distinct names")
    record.sources[id(v)] = (v, declared, dist)
    return v


def set_scenarios(model, n, seed=None):
    """Draw ``n`` scenarios, optionally from a given ``seed``.

    More scenarios estimate the true problem more closely and cost more to solve.
    Both settings belong to the model, not to the solve, so the same model always
    faces the same sample and two solves of it are comparable. Left unset, a model
    is solved over one scenario, where every random variable takes a single value.

    ``n`` and ``seed`` are both at least 1. Leaving ``seed`` out keeps the current
    one.

    Raises:
        ValueError: If ``n`` or ``seed`` is below 1.
    """
    # 0 is rejected rather than defaulted: protobuf cannot tell a zero from an absent
    # field, so a 0 sent here would arrive at the service as "use your default".
    if n < 1:
        raise ValueError(f"a model is solved over at least 1 scenario, got {n}")
    record = _declare(model)
    record.scenarios = int(n)
    if seed is not None:
        if seed < 1:
            raise ValueError(f"the scenario seed must be at least 1, got {seed}")
        record.seed = int(seed)


# ── aggregators: where a random quantity becomes a number ───────────────────

class _Aggregate(_ne.NumericExpression):
    """A scenario aggregator applied to a random quantity.

    A Pyomo expression node whose head is the service's name for the aggregator
    (``smean``, ``scvar``, ``sfreq_leq``, ``sfreq_geq``). It composes with Pyomo
    arithmetic like any other node, and ``quicopt.pyomo`` reads it through the
    same head dispatch it uses for every other operator — so nothing about the
    walk is special-cased.

    What *is* special is that it never evaluates. Pyomo cannot compute an
    expectation, and is never asked to: the node travels to the service as data.
    Attaching a local solver to a model containing one fails, which is correct.

    A level or threshold (CVaR's ``α``, a probability's ``τ``) is carried as an
    ordinary trailing argument, so the importer emits it as the constant the wire
    expects with no special handling.
    """

    __slots__ = ("_head",)

    def __init__(self, head, args):
        """Build the node.

        Args:
            head: The aggregator's catalog name.
            args: Its arguments — the aggregated quantity, then any level.
        """
        super().__init__(args)
        self._head = head

    def nargs(self):
        """The number of child arguments."""
        return len(self._args_)

    def getname(self, *args, **kwargs):
        """The catalog name, which is how every walker dispatches on this node."""
        return self._head

    def create_node_with_local_data(self, args, classtype=None):
        """Rebuild this node around new children, preserving the head.

        Pyomo clones expression trees while building constraints, and the default
        implementation would call the constructor with the arguments alone —
        losing the head, which is the only thing this node carries.

        Args:
            args: The new child arguments.
            classtype: A class to build instead, when a walker asks for one.

        Returns:
            The rebuilt node.
        """
        return _Aggregate(self._head, args) if classtype is None else classtype(args)

    def _apply_operation(self, result):
        """Refuse to evaluate: an aggregator's value is the service's to compute.

        Raises:
            ValueError: Always.
        """
        raise ValueError(f"'{self._head}' has no value here — it is computed over the "
                         "scenarios by the service, so this model solves through "
                         "quicopt, not through a locally attached solver")

    def _to_string(self, values, verbose, smap):
        """Render as a call, e.g. ``scvar(cost, 0.95)``."""
        return f"{self._head}({', '.join(values)})"


def expectation(x):
    """The expected value of ``x`` over the scenarios.

    Minimizing an expectation optimizes the average case and says nothing about
    the bad ones; use :func:`cvar` when the bad ones are what matter.

    ``x`` is any expression containing a random variable. The result is
    deterministic, and can be used anywhere a number can.
    """
    return _Aggregate("smean", [x])


def cvar(x, alpha):
    """The conditional value at risk of ``x`` at level ``alpha``.

    The mean of ``x`` over its worst ``1 − alpha`` fraction of scenarios — at
    ``alpha = 0.95``, the average of the worst 5%. Minimizing it optimizes the
    tail instead of the average, and is the usual way to ask for a solution that
    holds up in bad scenarios rather than merely on average.

    ``x`` is any expression containing a random variable. ``alpha`` is the tail
    level, a plain number strictly between 0 and 1; it cannot depend on a decision.

    Raises:
        TypeError: If ``alpha`` is not a plain number.
        ValueError: If ``alpha`` does not lie strictly between 0 and 1.
    """
    if not _is_number(alpha):
        raise TypeError(f"the tail level must be a plain number, got {type(alpha).__name__}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"the tail level must lie strictly between 0 and 1, got {alpha}")
    return _Aggregate("scvar", [x, float(alpha)])


def prob(relation):
    """The probability that ``relation`` holds — the fraction of scenarios it does.

    This is what a chance constraint is built from::

        m.service = pyo.Constraint(expr=prob(m.demand - m.x <= 0) >= 0.9)

    which reads as *demand is met in at least 90% of scenarios*. Note that the
    line holds two comparisons: the one inside ``prob`` is the event being
    measured, the outer one is the service level demanded of it.

    ``relation`` is a comparison, ``a <= b`` or ``a >= b``, with at least one side
    containing a random variable. The result is a probability between 0 and 1.

    Raises:
        TypeError: If the argument is not a comparison. An equality is refused
            too: for a continuous quantity its probability is zero.
        ValueError: If the comparison is strict. Use ``<=`` or ``>=``, which for a
            continuous quantity mean the same thing anyway.
    """
    if not isinstance(relation, InequalityExpression):
        raise TypeError("prob takes a comparison, as in prob(demand - x <= 0), "
                        f"not {type(relation).__name__}")
    if relation.strict:
        raise ValueError("prob takes <= or >=; a strict comparison has no scenario "
                         "counterpart (and for a continuous quantity means the same thing)")

    lower, upper = relation.args              # Pyomo normalizes any inequality to lower <= upper
    if _is_number(upper):
        return _Aggregate("sfreq_leq", [lower, float(upper)])
    if _is_number(lower):
        return _Aggregate("sfreq_geq", [upper, float(lower)])
    return _Aggregate("sfreq_leq", [lower - upper, 0.0])   # P(a ≤ b) = P(a − b ≤ 0)


# ── the kinked pair Pyomo has but does not publish ──────────────────────────

def maximum(*args):
    """The largest of two or more expressions.

    This is how recourse is priced: ``maximum(demand - stock, 0)`` is the
    shortfall, and zero whenever stock covers demand. Python's built-in ``max``
    cannot do this — it would have to compare the arguments, and their values are
    not known yet.

    Raises:
        ValueError: If given fewer than two arguments.
    """
    return _ne.MaxExpression(_at_least_two("maximum", args))


def minimum(*args):
    """The smallest of two or more expressions.

    The counterpart of :func:`maximum` — ``minimum(demand, stock)`` is what you
    actually sell.

    Raises:
        ValueError: If given fewer than two arguments.
    """
    return _ne.MinExpression(_at_least_two("minimum", args))


def _at_least_two(fname, args):
    """Check an arity and hand the arguments back as a list.

    Args:
        fname: The calling function's name, for the message.
        args: The argument tuple.

    Returns:
        list: The arguments.

    Raises:
        ValueError: If there are fewer than two.
    """
    if len(args) < 2:
        raise ValueError(f"{fname} compares at least two quantities, got {len(args)}")
    return list(args)


def _is_number(x):
    """Whether ``x`` is a plain Python number (and not a bool, which is not data here).

    Args:
        x: Any object.

    Returns:
        bool: True for ``int``/``float`` other than ``True``/``False``.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool)
