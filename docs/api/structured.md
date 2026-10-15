# `quicopt.structured`

::: quicopt.structured
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

## Orderings

To choose an order, such as the stops of a route or the jobs on a machine,
declare a permutation with `permutation`. It returns a `Permutation`: use its
`item_at` and `slot_of` in the objective and constraints as you would decision
variables, and its `precede` to require one item to come before another.

::: quicopt.structured.permutation
    options:
      heading_level: 3

::: quicopt.structured.Permutation
    options:
      heading_level: 3

## Groups

To split items into groups, such as players into teams or jobs onto machines,
declare a partition with `partition`. It returns a `Partition`: use its `part_of`
and `size_of` in the objective and constraints as you would decision variables.

::: quicopt.structured.partition
    options:
      heading_level: 3

::: quicopt.structured.Partition
    options:
      heading_level: 3

## Lookup tables

To read a number from a table at a position the service chooses, such as the
distance between two consecutive stops of a route, declare the table with
`lookup_table`. It returns a `LookupTable`, which you index with `item_at`,
`part_of` or an integer variable.

::: quicopt.structured.lookup_table
    options:
      heading_level: 3

::: quicopt.structured.LookupTable
    options:
      heading_level: 3

## Minimum loads and fees

For a quantity that is either 0 or at least some minimum, such as a delivery,
mark its variable with `semicontinuous` (continuous) or `semiinteger` (integer).
To charge a fee whenever such a quantity is not 0, add `fee * active(x)` to the
objective.

::: quicopt.structured.semicontinuous
    options:
      heading_level: 3

::: quicopt.structured.semiinteger
    options:
      heading_level: 3

::: quicopt.structured.active
    options:
      heading_level: 3

## Blocks

To tell the service which variables make up one subproblem, such as one
customer's deliveries, declare a block with `block`.

::: quicopt.structured.block
    options:
      heading_level: 3

## Largest and smallest

`maximum` and `minimum` are the largest and smallest of two or more expressions,
such as the larger of two teams' totals. They are the same functions as in
`quicopt.stochastic`.

::: quicopt.structured.maximum
    options:
      heading_level: 3

::: quicopt.structured.minimum
    options:
      heading_level: 3
