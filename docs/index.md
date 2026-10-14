# quicopt

The Python client for the [Quicopt](https://quicopt.com) optimization service.

You describe a decision: what you get to choose, what has to hold, and what you
want as much (or as little) of as possible. Quicopt finds the best choice there is.

Write the model with the Python modeling library you already use —
[Pyomo](https://www.pyomo.org/),
[OR-Tools MathOpt](https://developers.google.com/optimization/math_opt) or
[PuLP](https://coin-or.github.io/pulp/) — and `solve` encodes it, sends it to the
service and hands back the answer. There is no solver on your machine.

Reading and encoding a model (`ir` + `wire`) needs nothing outside the standard
library; support for each modeling library is an optional extra.

## Install

```sh
pip install quicopt              # the model and the encoder — standard library only
pip install "quicopt[pyomo]"     # + read models written in Pyomo
pip install "quicopt[mathopt]"   # + read models written in OR-Tools MathOpt
pip install "quicopt[pulp]"      # + read models written in PuLP
```

## Quickstart

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

[`solve`][quicopt.client.Client.solve] takes the model as it stands (Pyomo, OR-Tools
MathOpt, or PuLP) and reads it into Quicopt's own form on the way out. For a long
solve, [`submit`][quicopt.client.Client.submit] returns a [`Job`][quicopt.client.Job]
handle to poll. Pass `project="…"` to tag a call for per-project invoicing; which
modeling library you wrote in is recorded automatically.

If you want the encoded model yourself — to inspect it or send it by another route —
build a [`Program`][quicopt.ir.Program] (directly, or with one of the
`import_model` functions) and call [`encode`][quicopt.wire.encode].

## How it fits together

```
your model ──import_model──▶ Program ──encode──▶ bytes ──▶ service
 (Pyomo,                     the same           what travels
  MathOpt, PuLP)             model as           over the network
                             plain data
```

`Program` is what a model *is* to Quicopt, whichever library wrote it: variables,
expressions and constraints as data. Its schema is published
([protobuf](https://protobuf.dev)), so those bytes mean the same thing to the
Python, Julia and Ruby clients alike.

See the **API reference** for each step:

- [`ir`](api/ir.md) — the *intermediate representation*: the `Program`, a model
  as plain data.
- [`wire`](api/wire.md) — `Program` → the bytes the service reads.
- [`pyomo`](api/pyomo.md) — a Pyomo model → `Program`.
- [`stochastic`](api/stochastic.md) — writing a model whose data is not known when the decision is made.
- [`structured`](api/structured.md) — orderings such as a route, splits into groups, deliveries of none or at least a minimum load, fees, and data read at positions the service chooses.
- [`mathopt`](api/mathopt.md) — an OR-Tools MathOpt model → `Program`.
- [`pulp`](api/pulp.md) — a PuLP model → `Program`.
- [`client`](api/client.md) — POST those bytes and read the result back.
