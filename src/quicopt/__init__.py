# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt — the Python client for the Quicopt optimization service.

Takes an optimization model written in a Python modeling library (Pyomo, OR-Tools
MathOpt, PuLP) and reads it into a ``Program`` — a model as plain data, in the
published, language-neutral form the service reads. Nothing is solved here; see
``README.md`` for usage.

Steps:
  ir      — the ``Program``: a model as plain data, whichever library wrote it
            (short for *intermediate representation*)
  wire    — ``Program`` → the bytes the service reads
  pyomo   — a Pyomo model → ``Program``
  mathopt — an OR-Tools MathOpt model → ``Program``
  pulp    — a PuLP model → ``Program``
  client  — POST those bytes to the service and read the result (HTTP, stdlib)
"""
from .ir import (Const, Param, Var, Apply, Reduce, SetRef, SourceRef,
                 StructRef, TableRef,
                 Zero, Nonneg, Indicator, Parametric, Empirical, PermutationDecl,
                 VarDecl, IndexSet, Constraint, Program,
                 Domain, CONTINUOUS, INTEGER, BINARY)
from .wire import encode, encode_params, SCHEMA_VERSION
from .client import Client, Job, Result, QuicoptError, DEFAULT_BASE_URL

__all__ = [
    "Const", "Param", "Var", "Apply", "Reduce", "SetRef", "SourceRef",
    "StructRef", "TableRef",
    "Zero", "Nonneg", "Indicator", "Parametric", "Empirical", "PermutationDecl",
    "VarDecl", "IndexSet", "Constraint", "Program",
    "Domain", "CONTINUOUS", "INTEGER", "BINARY",
    "encode", "encode_params", "SCHEMA_VERSION",
    "Client", "Job", "Result", "QuicoptError", "DEFAULT_BASE_URL",
]
