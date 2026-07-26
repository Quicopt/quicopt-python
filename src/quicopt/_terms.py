# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt._terms — the shared ``Program`` forms every importer builds.

Private. An importer (``quicopt.pyomo``, ``quicopt.mathopt``, ``quicopt.pulp``)
differs only in the model it reads; the ``Program`` it writes is the same, and these
are the shapes it writes. They live here because that agreement is load-bearing: the
encoded bytes are checked byte-for-byte against what the service produces, so the
same model must come out the same ``Program`` whichever library wrote it, and a
per-importer copy of these forms is free to drift out of that agreement.

The constraint forms::

    f = c   ⇒  f − c = 0   (Zero)
    f ≥ l   ⇒  f − l ≥ 0   (Nonneg)
    f ≤ u   ⇒  u − f ≥ 0   (Nonneg)
    l ≤ f ≤ u  ⇒  both rows

An infinite bound is not a row — it is a free direction, and is dropped.
"""
from __future__ import annotations

from math import isfinite

from .ir import Apply, Const, Constraint, Nonneg, Zero


def _scaled(coef, e):
    """Return ``coef · e`` as a Quicopt expression, dropping a unit coefficient.

    Args:
        coef: The scalar multiplier; ``1.0`` leaves ``e`` unchanged.
        e: The expression being scaled.

    Returns:
        Expression: ``e`` if ``coef == 1.0`` else ``Apply("*", [Const(coef), e])``.
    """
    return e if coef == 1.0 else Apply("*", [Const(float(coef)), e])


def _sum(terms):
    """Fold terms into an n-ary sum, dropping additive-zero constants.

    Args:
        terms: The expression terms being added.

    Returns:
        Expression: ``Const(0.0)`` if every term is a zero constant, the sole term
        if exactly one survives, else an ``Apply("+", …)`` over the survivors.
    """
    nz = [t for t in terms if not (isinstance(t, Const) and t.value == 0.0)]
    if not nz:
        return Const(0.0)
    return nz[0] if len(nz) == 1 else Apply("+", nz)


def _minus(f, c):
    """Return ``f − c`` as a Quicopt expression, dropping the additive zero.

    Args:
        f: The expression on the left.
        c: The numeric constant subtracted (a bound); ``0.0`` leaves ``f`` unchanged.

    Returns:
        Expression: ``f`` if ``c == 0.0`` else ``Apply("-", [f, Const(c)])``.
    """
    return f if c == 0.0 else Apply("-", [f, Const(float(c))])


def _geq(u, f):
    """Return ``u − f`` — the body of ``f ≤ u`` written as ``u − f ≥ 0``.

    Args:
        u: The numeric upper bound.
        f: The constraint-body expression.

    Returns:
        Expression: ``Apply("-", [Const(u), f])``.
    """
    return Apply("-", [Const(float(u)), f])


def _emit(cons, f, lb, ub):
    """Append the ``Zero``/``Nonneg`` rows for one ``lb ≤ f ≤ ub`` range to ``cons``.

    A finite ``lb == ub`` is an equality (``f − lb = 0``, ``Zero``); otherwise each
    finite side emits a ``Nonneg`` row. An infinite side is dropped as a free
    direction.

    Args:
        cons: The list of :class:`~quicopt.ir.Constraint` rows appended to
            (mutated in place).
        f: The expression of the row body.
        lb: The row's lower bound (``-inf`` for none).
        ub: The row's upper bound (``+inf`` for none).

    Returns:
        None. Rows are appended to ``cons``.
    """
    if isfinite(lb) and lb == ub:              # equality: f − c = 0
        cons.append(Constraint(_minus(f, lb), Zero(), []))
        return
    if isfinite(lb):                           # f − lb ≥ 0
        cons.append(Constraint(_minus(f, lb), Nonneg(), []))
    if isfinite(ub):                           # ub − f ≥ 0
        cons.append(Constraint(_geq(ub, f), Nonneg(), []))
