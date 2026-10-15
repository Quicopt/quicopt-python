# `quicopt.stochastic`

::: quicopt.stochastic
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

## Random variables

To declare data you only learn after deciding, such as demand, give a Pyomo
variable a distribution with `set_distribution`. `Normal` is a normal
distribution; `Empirical` is a column of values you sampled yourself, one per
scenario.

::: quicopt.stochastic.set_distribution
    options:
      heading_level: 3

::: quicopt.stochastic.Normal
    options:
      heading_level: 3

::: quicopt.stochastic.Empirical
    options:
      heading_level: 3

::: quicopt.stochastic.Distribution
    options:
      heading_level: 3

## Scenarios

To choose how many scenarios the model is solved over, and the seed they are
drawn from, call `set_scenarios`.

::: quicopt.stochastic.set_scenarios
    options:
      heading_level: 3

## Objectives and constraints over scenarios

An expression with a random variable in it takes a different value in each
scenario. To use it in an objective or a constraint, reduce it to one number:
`expectation` is its average, `cvar` its average over the worst scenarios, and
`prob` the fraction of scenarios in which a comparison holds.

::: quicopt.stochastic.expectation
    options:
      heading_level: 3

::: quicopt.stochastic.cvar
    options:
      heading_level: 3

::: quicopt.stochastic.prob
    options:
      heading_level: 3

## Largest and smallest

`maximum` and `minimum` are the largest and smallest of two or more expressions,
such as the shortfall `maximum(demand - stock, 0)`, which is 0 whenever stock
covers demand.

::: quicopt.stochastic.maximum
    options:
      heading_level: 3

::: quicopt.stochastic.minimum
    options:
      heading_level: 3
