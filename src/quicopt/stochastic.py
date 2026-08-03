# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.stochastic — models whose data is not known when the decision is made.

Some decisions have to be taken before the numbers are in: how much to stock
before demand arrives, what to commit before prices move. Such a model is an
ordinary Pyomo model with two additions. A variable can be given a
*distribution*, which makes it a random variable rather than something to
choose. And a quantity that depends on one is random too, so it has to be closed
with an *aggregator* — an expected value, a tail risk, a probability — before it
can be an objective or a constraint, because only a number can be minimized.

Nothing is evaluated here. A distribution and an aggregator travel to the service
as data, which is why they can be written at all: Pyomo has no evaluator for
them, and needs none.

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

    Using ``m.demand`` twice refers to the *same* draw — one name is one random
    variable, so two independent ones are two variables with two names.

    These constructors need Pyomo, so they live here rather than in the top-level
    package, which stays dependency-free. ``maximum`` and ``minimum`` are here for
    a duller reason: Pyomo has the expressions but publishes no way to build them,
    and ``max(a, b)`` cannot work — Python would have to decide which is larger,
    which is exactly what is not yet known.
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
    """A distribution to draw from: a service catalog ``head`` and its parameters.

    :class:`Normal` and its siblings are the named forms; this base class is the
    general one, and takes any head the service knows — so a distribution that
    the service has learned and this client has not yet been taught is still
    reachable, without waiting for a release.

    A parameter is a number or an ordinary deterministic Pyomo expression. The
    latter is what makes a distribution whose mean is itself a decision — an
    *endogenous* one — nothing special: it is the same grammar.
    """

    def __init__(self, head, *params):
        """Declare a draw from ``head``, the distribution's name in the service's
        operator catalog, with the given parameters — each a number or a
        deterministic Pyomo expression."""
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
    """Make the Pyomo variable ``v`` random, drawn from ``dist``.

    ``v`` stops being a variable the solver chooses and becomes one it is handed:
    every use of it refers to the same draw, per scenario. Independent random
    variables are declared separately, under distinct names.

    Do not give ``v`` bounds or a domain — it carries a distribution, which is
    what says where it lives. The import rejects a source that has them.

    ``v`` is a scalar ``Var`` or one element of an indexed one, belonging to
    ``model``. ``dist`` says how it is drawn: a :class:`Distribution` such as
    :class:`Normal`, or an :class:`~quicopt.ir.Empirical` column carrying one
    value per scenario. ``name`` is what it is declared and referenced under,
    defaulting to the variable's own name — and it is the identity that matters,
    since two references under one name are one random variable. The variable
    itself is returned, so the call can be chained onto its declaration.

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
    """Set how many scenarios the model is solved over, and optionally the seed.

    Both are part of the model rather than of the solve: they pin which sample is
    drawn, so the same model always faces the same scenarios and two solves are
    comparable. Left unset, the model is solved over a single scenario — the
    degenerate case, where every random variable takes one value.

    ``n`` is the number of scenarios and ``seed`` the seed they are drawn from,
    both at least 1. Leaving ``seed`` out keeps the previous one, which is 1
    until it is set.

    Raises:
        ValueError: If ``n`` or ``seed`` is below 1. Zero is not merely invalid
            but unrepresentable: the wire cannot distinguish it from an absent
            field, so it would arrive as "use the default".
    """
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

    The plainest way to close a random quantity: average it. What a solution
    optimizes is then the average case, which says nothing about the bad ones —
    see :func:`cvar` when those are the point.

    ``x`` is a quantity depending on at least one random variable. What comes
    back is an ordinary number as far as the rest of the model is concerned.
    """
    return _Aggregate("smean", [x])


def cvar(x, alpha):
    """The conditional value at risk of ``x`` at level ``alpha``.

    The mean of ``x`` over its worst ``1 − alpha`` fraction of scenarios: at
    ``alpha = 0.95``, the average outcome across the worst 5%. Minimizing it
    optimizes the bad tail rather than the average, which is what "robust"
    usually means in practice.

    ``x`` is a quantity depending on at least one random variable, and ``alpha``
    the tail level, strictly between 0 and 1. The level has to be a plain number:
    the wire carries it as a constant, so it cannot depend on a decision.

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

    which reads as *demand is met in at least 90% of scenarios*. The comparison
    is written inside the call rather than named by it (``prob_atmost(x, τ)``)
    for one reason: there are two comparisons in that line, and only this way is
    it unambiguous which one is being made a probability of.

    ``relation`` is a non-strict comparison, ``a <= b`` or ``a >= b``, with at
    least one side depending on a random variable. What comes back is a
    probability, between 0 and 1.

    Raises:
        TypeError: If the argument is not a comparison — an equality in
            particular, whose probability is zero for any continuous quantity.
        ValueError: If the comparison is strict. ``<`` and ``>`` have no
            scenario counterpart, and against a continuous quantity they would
            in any case mean the same as ``<=`` and ``>=``.
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
    """The largest of its arguments, as an expression rather than a decision.

    The recourse idiom is written with it — ``maximum(demand - stock, 0)`` is what
    you are short by, which is 0 when you are not short at all. Takes two or more
    quantities and gives back a Pyomo expression.

    Raises:
        ValueError: If given fewer than two arguments.
    """
    return _ne.MaxExpression(_at_least_two("maximum", args))


def minimum(*args):
    """The smallest of its arguments, as an expression rather than a decision.

    Takes two or more quantities and gives back a Pyomo expression.

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
