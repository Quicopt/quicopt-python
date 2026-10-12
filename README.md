# quicopt

[![PyPI](https://img.shields.io/pypi/v/quicopt.svg)](https://pypi.org/project/quicopt/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/quicopt/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://quicopt.github.io/quicopt-python/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/quicopt.svg)](https://github.com/Quicopt/quicopt-python/blob/main/LICENSE)

The Python client for the [Quicopt](https://quicopt.com) optimization service.

You describe a decision: what you get to choose, what has to hold, and what you
want as much (or as little) of as possible. Quicopt finds the best choice there is.

Write the model with the Python modeling library you already use —
[Pyomo](https://www.pyomo.org/),
[OR-Tools MathOpt](https://developers.google.com/optimization/math_opt) or
[PuLP](https://coin-or.github.io/pulp/) — and `solve` encodes it, sends it to the
service and hands back the answer. There is no solver on your machine; the service
does the solving.

Full documentation: **<https://quicopt.github.io/quicopt-python/>**

## Install

```sh
pip install quicopt              # the model and the encoder — standard library only
pip install "quicopt[pyomo]"     # + read models written in Pyomo
pip install "quicopt[mathopt]"   # + read models written in OR-Tools MathOpt
pip install "quicopt[pulp]"      # + read models written in PuLP
```

From source (contributors), an editable install into a virtual environment:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[pyomo,mathopt,pulp]'
```

## Use

```python
import pyomo.environ as pyo
from quicopt import Client

m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(0.1, 10))
m.obj = pyo.Objective(expr=m.x**2 + 1.0 / m.x, sense=pyo.minimize)

client = Client()                             # defaults to the free-tier Quicopt server
result = client.solve(m)                      # read the model, encode it, solve it
print(result.status, result.objective, result.solution)
print(result.display)                         # the service's ready-to-print summary
```

`solve` takes the model as it stands (Pyomo, OR-Tools MathOpt, or PuLP) and reads
it into Quicopt's own form on the way out. The first keyless call mints an API key,
cached at `$XDG_CACHE_HOME/quicopt/free_key` (`~/.cache/…` by default) and replayed
on every later call — including from later runs, so you keep one key without doing
anything.
Pass `Client(api_key=…)` to authenticate with a key you already hold (used as-is,
never cached), or `Client(cache=False)` to keep the key in memory only. Point
`Client(base_url=…)` at another server to override the default. For a long solve,
`client.submit(m)` returns a job handle to poll — `job.result()`.

Where the home directory does not survive the run (CI, containers, Colab), the
cache is wiped between sessions and each run mints a new key. Set
`QUICOPT_KEY_PATH` to a durable location — or `Client(key_path=…)` — to keep one
key across sessions.

Tag a call with `client.solve(m, project="my-project")` to attribute it to a
project — handy when one key serves several projects. Which modeling library you
wrote in (Pyomo/MathOpt/PuLP) is recorded automatically.

The service decides how long a solve may run. `client.solve(m, time_limit=30)`
sets it to 30 seconds instead: a solver that proves its answer sooner still stops
sooner, and a search runs for all of it. The service grants at most its maximum
(120 s on the free tier), and the client warns when a solve got less time than
asked for.

If you want the encoded model yourself — to inspect it, store it, or send it by
another route — the importers and the encoder are public too:

```python
from quicopt import encode
from quicopt.pyomo import import_model

payload = encode(import_model(m))   # Pyomo model → a Quicopt model → bytes
```

## Layout

```
quicopt/ir.py         a model as plain data: variables, expressions, constraints
quicopt/wire.py       a model → the bytes the service reads (standard library only)
quicopt/pyomo.py      a Pyomo model → a Quicopt model
quicopt/mathopt.py    an OR-Tools MathOpt model → a Quicopt model
quicopt/pulp.py       a PuLP model → a Quicopt model
quicopt/stochastic.py writing a Pyomo model whose data is not known yet
quicopt/structured.py deciding an order in a Pyomo model, and lookup tables
quicopt/client.py     POST those bytes to the service, read the result (HTTP, stdlib)
```

Between client and service, a model is a `Program`: variables, expressions and
constraints as data, with a published [protobuf](https://protobuf.dev) schema for
it, which `wire.py` encodes exactly. Each importer is an independent module —
`pyomo.py`, `mathopt.py`, `pulp.py`, and further modeling libraries slot in the
same way — pulls in only its own optional extra, and builds the model through the
shared forms in `_terms.py`, so the same model comes out the same bytes whichever
library wrote it.

## Test

The encoder is checked against committed golden bytes, with **no dependencies**:

```sh
python3 tests/test_wire_golden.py        # or: pytest tests/
```

The importers are pinned to each other by byte equality: the same model written in
Pyomo and in PuLP must encode to identical bytes, which carries the goldens'
authority across (`tests/test_frontend_equivalence.py`; needs the `[pyomo,pulp]`
extras, skips without them).

## Status

- **the model + the encoder** (`ir`, `wire`) — stable; the bytes are exactly what
  the service decodes.
- **pyomo importer** — affine / quadratic / nonlinear (`+ - * / ^ sin cos exp log
  sqrt abs`), variable bounds (incl. unbounded) + integrality, `==` / `<=` / `>=` /
  ranged constraints, `min` / `max`. A fixed variable pins to `[val, val]`; one fixed
  *without* a value raises rather than importing as free.
- **mathopt importer** — OR-Tools MathOpt `ModelProto`: linear / quadratic
  objective, linear constraints (incl. ranged and one-sided), variable bounds
  (incl. unbounded) + integrality, `min` / `max`.
- **pulp importer** — PuLP `LpProblem`: linear objective (with offset) and linear
  `<=` / `==` / `>=` constraints, variable bounds (incl. unbounded) + integrality,
  `min` / `max`. PuLP is linear by construction, so this is exactly LP / MILP; a
  problem with no objective is a feasibility problem (a constant `0`).
- **orders and lookup tables** (`structured`, Pyomo) — a permutation of items
  over slots, read by `item_at` / `slot_of`, with precedences and a starting
  arrangement; tables indexed by decisions. Counted from 0, like Python.
- **sending (HTTP)** — `Client.solve` / `Client.submit` over `/v1/solve` and
  `/v1/jobs`: the encoded model up, result JSON (status / objective / solution /
  framed `display` / the arrangement of each permutation) back; API-key minting on
  the first call, optional gzip, an optional time limit. Standard library only.

## License

Apache License 2.0 — see [`LICENSE`](https://github.com/Quicopt/quicopt-python/blob/main/LICENSE). (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich.
