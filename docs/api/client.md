# `quicopt.client`

::: quicopt.client
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

## Solving a model

Create a `Client` and pass your model to `solve`, which sends it to the service
and waits for the answer. For a long solve, `submit` sends it and returns at once
with a `Job`.

::: quicopt.client.Client
    options:
      heading_level: 3

## Long solves

A `Job` stands for a model sent with `submit`. Ask it for its `status` while the
service works on it, and for its `result`, which waits until the answer is ready.

::: quicopt.client.Job
    options:
      heading_level: 3

## Results

`solve` and `Job.result` return a `Result`: the status, the objective value and
the value of each variable. For a model that declares a permutation or a partition
(see `quicopt.structured`), it also holds the `Arrangement` or `Grouping` the
service found.

::: quicopt.client.Result
    options:
      heading_level: 3

::: quicopt.client.Arrangement
    options:
      heading_level: 3

::: quicopt.client.Grouping
    options:
      heading_level: 3

## Errors

When the service turns a request down, such as a model too large for the free
tier, the call raises a `QuicoptError`. Its `reason` says why, and its `display`
is a message ready to print.

::: quicopt.client.QuicoptError
    options:
      heading_level: 3

## Server and API key

`Client()` with no arguments uses the free tier at `DEFAULT_BASE_URL`. On its
first call it receives an API key and saves it in a file, so later runs reuse the
same key. The environment variable named by `KEY_PATH_ENV` moves that file.

::: quicopt.client.DEFAULT_BASE_URL
    options:
      heading_level: 3

::: quicopt.client.KEY_PATH_ENV
    options:
      heading_level: 3
