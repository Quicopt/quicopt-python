# `quicopt.ir`

::: quicopt.ir
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

## The model

A `Program` holds a whole model: its variables, objective and constraints, and
any index sets, data, random variables and structures it uses. Pass it to
`Client.solve` like a model from a modeling library, or turn it into bytes
yourself with `quicopt.wire.encode`.

::: quicopt.ir.Program
    options:
      heading_level: 3

## Variables

Each variable is declared by a `VarDecl`: its name, its index sets, its `Domain`
(continuous, integer, binary, semicontinuous or semi-integer), its bounds and its
starting value.

::: quicopt.ir.VarDecl
    options:
      heading_level: 3

::: quicopt.ir.Domain
    options:
      heading_level: 3

## Expressions

The objective and every constraint are expressions, built as a tree: `Const` is a
number, `Var` a variable, and `Apply` applies an operator such as `"+"`, `"*"` or
`"exp"` to its arguments.

::: quicopt.ir.Expression
    options:
      heading_level: 3

::: quicopt.ir.Const
    options:
      heading_level: 3

::: quicopt.ir.Var
    options:
      heading_level: 3

::: quicopt.ir.Apply
    options:
      heading_level: 3

## Constraints

A `Constraint` requires an expression to lie in a set: `Zero` means it equals 0,
`Nonneg` that it is at least 0. So `x + 2y ≤ 5` is written as `5 − (x + 2y)` in
`Nonneg`. `Indicator` makes a constraint hold only when a binary variable is 1.

::: quicopt.ir.Constraint
    options:
      heading_level: 3

::: quicopt.ir.Zero
    options:
      heading_level: 3

::: quicopt.ir.Nonneg
    options:
      heading_level: 3

::: quicopt.ir.Indicator
    options:
      heading_level: 3

## Sets, data and sums

A model with one variable per warehouse, or one constraint per customer, is
written over index sets. Each set is an `IndexSet`, and the data the model reads
lives in tables in `Program.params`. `Param` is an entry of such a table, `Reduce`
sums (or multiplies) an expression over a set, and `SetRef` names the set it runs
over. A constraint's `over` repeats it for every element of a set.

::: quicopt.ir.IndexSet
    options:
      heading_level: 3

::: quicopt.ir.SetRef
    options:
      heading_level: 3

::: quicopt.ir.Param
    options:
      heading_level: 3

::: quicopt.ir.Reduce
    options:
      heading_level: 3

## Random variables

A model under uncertainty lists its random variables in `Program.sources`, each
either `Parametric` (drawn from a distribution) or `Empirical` (a column of
values, one per scenario), and uses them in expressions through `SourceRef`.
`quicopt.stochastic` builds these for you from a Pyomo model.

::: quicopt.ir.Parametric
    options:
      heading_level: 3

::: quicopt.ir.Empirical
    options:
      heading_level: 3

::: quicopt.ir.SourceRef
    options:
      heading_level: 3

## Orderings, groups and tables

A permutation or a partition is declared in `Program.structures`, as a
`PermutationDecl` or a `PartitionDecl`, and read in expressions through
`StructRef`. `TableRef` reads a data table at positions the service chooses.
`quicopt.structured` builds these for you from a Pyomo model. Note that items,
slots and parts count from 1 here, while `quicopt.structured` counts them from 0.

::: quicopt.ir.PermutationDecl
    options:
      heading_level: 3

::: quicopt.ir.PartitionDecl
    options:
      heading_level: 3

::: quicopt.ir.StructRef
    options:
      heading_level: 3

::: quicopt.ir.TableRef
    options:
      heading_level: 3
