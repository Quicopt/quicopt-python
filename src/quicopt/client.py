# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tim Bode, PGI-12, Forschungszentrum Jülich
"""
quicopt.client — send a model to the Quicopt service and get the answer back.

A [`Client`][quicopt.client.Client] sends your model to the service, and the
answer comes back as a [`Result`][quicopt.client.Result]: the status, the
objective value and the value of each variable. There are two ways to send it:

- [`Client.solve`][quicopt.client.Client.solve] sends the model and waits for
  the answer.
- [`Client.submit`][quicopt.client.Client.submit] sends it and returns at once
  with a [`Job`][quicopt.client.Job], which you ask for the answer later. Use it
  for a long solve.

No account is needed to start. On its first call a client is given an API key,
which it saves in a file, so later runs on the same machine use the same key.

When the service turns a request down, the call raises a
[`QuicoptError`][quicopt.client.QuicoptError] that says why.

Example:
    Maximize ``3x + 5y`` subject to ``x + 2y ≤ 5``, with ``0 ≤ x ≤ 4`` and ``y``
    binary:

    ```python
    import pulp
    from quicopt import Client

    prob = pulp.LpProblem("mix", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0, upBound=4)
    y = pulp.LpVariable("y", cat="Binary")
    prob += 3 * x + 5 * y
    prob += x + 2 * y <= 5

    result = Client().solve(prob)                 # read, encode, POST, wait
    print(result.status, result.objective)        # optimal 14.0
    print(result.solution)                        # {'x': 3.0, 'y': 1.0}
    print(result.model_class)                     # milp
    print(result.display)                         # the service's own summary
    ```

    The model is passed as it stands — Pyomo, MathOpt or PuLP; the matching
    importer is chosen by type, so nothing here changes with the library, and a
    :class:`~quicopt.ir.Program` or already-encoded bytes are accepted too. No
    solver runs locally and none need be installed. Which kind of model it was is
    the service's finding, reported back as ``model_class``, not something to
    declare.

    For a long solve, :meth:`Client.submit` returns immediately with a
    :class:`Job` and :meth:`Job.result` waits on it:

    ```python
    job = Client().submit(prob, project="pricing-study")
    print(job.status()["status"])                 # queued
    result = job.result(timeout=600)              # poll until done
    ```

    The service decides how long a solve may run. ``time_limit`` sets it instead,
    in seconds: a solver that proves its answer sooner still stops sooner, and a
    search, which cannot prove it is done, may stop sooner too, once it stops
    finding better answers:

    ```python
    result = Client().solve(m, time_limit=30)
    print(result.solver_data["time_budget_s"])    # 30.0
    ```

    The service grants at most its maximum (120 s on the free tier). Asked for
    more, it runs for its maximum, and the client warns that the solve got less
    time than requested.
"""
from __future__ import annotations

import contextlib
import gzip as _gzip
import json
import os
import tempfile
import time
# urllib rather than an HTTP library: sending a model adds no dependency, like
# the rest of the package's core.
import urllib.error
import urllib.request
import warnings
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, urlencode

from .ir import Program
from .wire import encode

__all__ = ["Client", "Job", "Result", "Arrangement", "Grouping", "QuicoptError",
           "DEFAULT_BASE_URL", "KEY_PATH_ENV"]

_OCTET = "application/octet-stream"

KEY_PATH_ENV = "QUICOPT_KEY_PATH"
"""Environment variable overriding where the free key is cached. Point it at
durable storage in an environment whose home directory does not survive the run
(CI, containers, Colab), where the default location is wiped between sessions and
every run would otherwise mint a fresh key."""

DEFAULT_BASE_URL = "https://try.quicoptapi.pgi.fz-juelich.de"
"""The public Quicopt free-tier endpoint a :class:`Client` targets when no
``base_url`` is given. Mirrors the Julia client's ``DEFAULT_BASE_URL`` so both
clients reach the same server out of the box."""


def _default_key_path() -> Path:
    """The default location of the cached free key.

    ``$XDG_CACHE_HOME/quicopt/free_key``, falling back to ``~/.cache`` when the
    variable is unset, unless :data:`KEY_PATH_ENV` overrides the whole path.

    Returns:
        Path: Where the free key is read from and written to.
    """
    override = os.environ.get(KEY_PATH_ENV)
    if override:
        return Path(override).expanduser()
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root).expanduser() / "quicopt" / "free_key"


def _read_key(path: Path) -> Optional[str]:
    """Read the cached key, or ``None`` when there is nothing usable to read.

    A missing or unreadable cache is not an error — it just means this caller has
    no key yet and the next call mints one.

    Args:
        path: The cache file.

    Returns:
        The cached key, or ``None`` if the file is absent, unreadable, or empty.
    """
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _write_key(path: Path, key: str) -> None:
    """Persist ``key`` at ``path``, atomically and readable only by its owner.

    The key is a credential, so the file is ``0600``; the write goes to a temp file
    in the same directory and is then ``os.replace``d into position, so a crash or
    a concurrent writer can never leave a truncated key behind for the next run to
    send.

    Args:
        path: The cache file.
        key: The key to persist.

    Raises:
        OSError: If the directory or file cannot be written (callers treat caching
            as best-effort and degrade to a warning).
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".free_key.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# How long a request may take beyond the solve itself — waiting in the queue,
# reading the model, writing the answer — before the client stops waiting on a
# solve whose time limit it set.
_WAIT_MARGIN = 60.0


@dataclass(frozen=True)
class Arrangement:
    """The arrangement the service found for one permutation (see
    ``quicopt.structured``), both ways round and counted from 0: ``item_at[k]`` is
    the item in slot ``k`` — for a route, the stops in the order visited — and
    ``slot_of[i]`` the slot of item ``i``."""
    item_at: List[int]
    slot_of: List[int]


@dataclass(frozen=True)
class Grouping:
    """The assignment the service found for one partition (see
    ``quicopt.structured``): ``part_of[i]`` is the part item ``i`` is in, counted
    from 0, and ``size_of[b]`` the number of items in part ``b``."""
    part_of: List[int]
    size_of: List[int]


@dataclass(frozen=True)
class Result:
    """A finished solve, parsed from the service's result JSON. ``objective`` and
    ``feasible`` are ``None`` where the outcome leaves them undefined — a model
    with nothing to feasibility-check, or a run that found no solution.
    ``display`` is the framed, ready-to-print summary the service renders the same
    way for every kind of model. ``structures`` holds, by name, the
    :class:`Arrangement` of each permutation and the :class:`Grouping` of each
    partition the model declares, and is empty for a model with neither."""
    job_id: str
    status: str
    objective: Optional[float]
    feasible: Optional[bool]
    solution: Dict[str, float]
    solve_time_seconds: float
    solver_data: Dict[str, Any]
    display: str
    structures: Dict[str, Union[Arrangement, Grouping, None]] = field(default_factory=dict)

    @property
    def model_class(self) -> Optional[str]:
        """The kind of model the service recognised (``LP``/``MILP``/``QUBO``/…).

        Returns:
            The ``model_class`` recorded in ``solver_data``, or ``None`` if the
            service did not report one.
        """
        return self.solver_data.get("model_class")

    @classmethod
    def _from_json(cls, d: Dict[str, Any]) -> "Result":
        """Build a ``Result`` from the service's decoded result JSON.

        Missing optional keys default rather than raise, so a partial result (e.g. a
        heuristic backend that reports no ``objective``) still parses.

        Args:
            d: The decoded result JSON object.

        Returns:
            Result: The parsed result.
        """
        return cls(
            job_id=d.get("job_id", ""),
            status=d["status"],
            objective=d.get("objective"),
            feasible=d.get("feasible"),
            solution=dict(d.get("solution") or {}),
            solve_time_seconds=d.get("solve_time_seconds", 0.0),
            solver_data=dict(d.get("solver_data") or {}),
            display=d.get("display", ""),
            structures={name: _structure(s) for (name, s) in (d.get("structures") or {}).items()},
        )


def _structure(s: Optional[Dict[str, Any]]) -> Union[Arrangement, Grouping, None]:
    """Read one structure's value from the result JSON: a permutation's
    arrangement (keys ``item_at``, ``slot_of``) or a partition's grouping (keys
    ``part_of``, ``size_of``).

    The service counts items, slots and parts from 1; Python, and this client's
    ``quicopt.structured``, from 0. A part's size is a count and is not shifted.

    Args:
        s: The JSON object, or ``None`` when no solver returned a value.

    Returns:
        The arrangement or grouping counted from 0, or ``None``.
    """
    if s is None:
        return None
    if "part_of" in s:
        return Grouping(part_of=[int(part) - 1 for part in s["part_of"]],
                        size_of=[int(size) for size in s["size_of"]])
    return Arrangement(item_at=[int(item) - 1 for item in s["item_at"]],
                       slot_of=[int(slot) - 1 for slot in s["slot_of"]])


def _warn_if_lowered(result: Result) -> Result:
    """Warn when the service ran a solve for less time than its ``time_limit``.

    The service grants at most its maximum and reports what it asked for and what
    it granted side by side in ``solver_data``.

    Args:
        result: A finished solve.

    Returns:
        Result: ``result``, unchanged.
    """
    asked = result.solver_data.get("time_limit_s")
    granted = result.solver_data.get("time_budget_s")
    if asked is not None and granted is not None and granted < asked:
        warnings.warn(f"the service ran this solve for {granted:g} s, not the {asked:g} s "
                      f"asked for: {granted:g} s is the most it grants", stacklevel=3)
    return result


def _check_time_limit(time_limit: Optional[float]) -> Optional[float]:
    """Check a time limit before it is sent.

    Args:
        time_limit: Seconds, or ``None`` to leave the choice to the service.

    Returns:
        The time limit as a float, or ``None``.

    Raises:
        TypeError: If it is not a number.
        ValueError: If it is not positive and finite.
    """
    if time_limit is None:
        return None
    if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)):
        raise TypeError(f"time_limit is a number of seconds, got {time_limit!r}")
    if not (math.isfinite(time_limit) and time_limit > 0):
        raise ValueError(f"time_limit is a positive, finite number of seconds, got {time_limit}")
    return float(time_limit)


class QuicoptError(Exception):
    """A non-2xx service response. ``reason`` is the service's stable snake_case
    code (``size_exceeded``, ``unsupported_model``, ``quota_exhausted``, …),
    ``display`` the framed message to print, ``status_code`` the HTTP status."""

    def __init__(self, status_code: int, body: Dict[str, Any]):
        self.status_code = status_code
        self.body = body if isinstance(body, dict) else {"error": str(body)}
        self.reason = self.body.get("reason")
        self.display = self.body.get("display")
        super().__init__(self.body.get("error") or f"HTTP {status_code}")


_Model = Any    # a modeling-library model (Pyomo, MathOpt, PuLP), a Program, or encoded bytes


def _import_frontend(model: Any) -> Program:
    """A modeling-library object → ``Program``, via that library's importer,
    dispatched by type. The importers are loaded lazily so the standard-library-only
    core never pulls in an optional dependency until a model of that kind is
    actually solved."""
    mod = type(model).__module__ or ""
    if mod.startswith("ortools") or hasattr(model, "export_model"):
        from .mathopt import import_model
        return import_model(model)
    if mod.startswith("pyomo"):
        from .pyomo import import_model
        return import_model(model)
    if mod.startswith("pulp"):
        from .pulp import import_model
        return import_model(model)
    raise TypeError(f"cannot solve a {type(model).__name__}: pass a Pyomo, OR-Tools "
                    "MathOpt, or PuLP model, a Program, or encoded bytes")


def _to_wire(model: _Model) -> bytes:
    """Coerce to encoded bytes: bytes pass through, a ``Program`` is encoded, and a
    model from a modeling library (Pyomo, MathOpt, PuLP) is read then encoded — so
    a caller solves the *model*, never a hand-built ``Program``."""
    if isinstance(model, (bytes, bytearray)):
        return bytes(model)
    if isinstance(model, Program):
        return encode(model)
    return encode(_import_frontend(model))


def _query(config: Optional[Dict[str, Any]]) -> str:
    """Render a config mapping as a URL query-string suffix.

    Encodes spaces as ``%20`` (``quote``, not the default ``+``) so a value round-
    trips unambiguously through the service's query parser, matching the Julia
    client.

    Args:
        config: Optional service parameters.

    Returns:
        str: ``"?k=v&…"`` if ``config`` is non-empty, else ``""``.
    """
    return "?" + urlencode(config, quote_via=quote) if config else ""


def _source_language(model: Any) -> Optional[str]:
    """The modeling library that wrote ``model`` (for the ``source_language`` tag),
    or ``None`` for a hand-built ``Program`` / raw bytes, which have none to
    attribute. Mirrors the dispatch in ``_import_frontend``."""
    if isinstance(model, (bytes, bytearray, Program)):
        return None
    mod = type(model).__module__ or ""
    if mod.startswith("ortools") or hasattr(model, "export_model"):
        return "mathopt"
    if mod.startswith("pyomo"):
        return "pyomo"
    if mod.startswith("pulp"):
        return "pulp"
    return None


def _meta_config(model: Any, project: Optional[str], time_limit: Optional[float],
                 config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge the per-call tags into the request config (sent as query params): the
    auto-detected ``source_language`` (unless the caller already set one in
    ``config``), the optional ``project_id`` and the optional ``time_limit``. The
    model is unchanged — these ride the query string, not the wire bytes."""
    meta: Dict[str, Any] = dict(config or {})
    src = _source_language(model)
    if src is not None:
        meta.setdefault("source_language", src)
    if project is not None:
        meta["project_id"] = project
    if time_limit is not None:
        meta["time_limit"] = time_limit
    return meta


class Client:
    """A connection to a Quicopt service at ``base_url`` — the public free tier
    (:data:`DEFAULT_BASE_URL`) unless another URL is given. Holds the API key: pass
    a known one, or let the first keyless call mint one — which is then cached on
    disk and reused by later runs, so one caller keeps one key."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: Optional[str] = None,
                 *, timeout: float = 60.0, key_path: Optional[Union[str, Path]] = None,
                 cache: bool = True):
        """Bind a client to a service endpoint.

        Args:
            base_url: The service base URL; a trailing slash is stripped. Defaults
                to :data:`DEFAULT_BASE_URL`, the public free-tier endpoint; pass
                another URL to target a different server.
            api_key: A known API key, or ``None`` to reuse the cached free key
                (minting one on the first keyless call). A key passed here is used
                as-is and never written to the cache — so authenticating with a
                specific key cannot clobber the free key of the user running you.
            timeout: Per-request socket timeout, in seconds.
            key_path: Where to cache the free key. Defaults to
                :func:`_default_key_path`.
            cache: Set ``False`` to keep the key in memory only, neither reading nor
                writing the cache file.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.key_path = Path(key_path).expanduser() if key_path is not None else _default_key_path()
        self.cache = cache
        self._explicit = api_key is not None
        self.api_key = api_key if self._explicit else (_read_key(self.key_path) if cache else None)
        # Only a key that came off disk may be discarded and re-minted on a 401 —
        # see `_open`, which relies on this to bound re-minting to one per run.
        self._from_cache = self.api_key is not None and not self._explicit

    def solve(self, model: _Model, *, project: Optional[str] = None,
              time_limit: Optional[float] = None,
              config: Optional[Dict[str, Any]] = None, gzip: bool = False) -> Result:
        """Solve ``model`` synchronously, blocking until the result returns.

        Args:
            model: A model from a modeling library (Pyomo, OR-Tools MathOpt, PuLP)
                — read and encoded here — or a ``Program`` / already-encoded bytes
                if you built them yourself.
            project: Optional project tag for the call, so calls on one key can be
                invoiced per project. Sent as a query param, not baked into the model.
            time_limit: Optional number of seconds the solve may run, in place of
                the time the service would choose. A solver that proves its answer
                sooner stops sooner, and a search may too, once it stops finding
                better answers. The service grants at most its maximum, and a
                warning says so when it granted less. The client waits for the
                answer at least this long plus a minute.
            config: Optional service parameters, sent as the query string.
            gzip: If ``True``, gzip-compress the request body.

        Returns:
            Result: The finished solve.

        Raises:
            QuicoptError: On a non-2xx response.
            TypeError: If ``time_limit`` is not a number.
            ValueError: If ``time_limit`` is not positive and finite.
        """
        time_limit = _check_time_limit(time_limit)
        wait = self.timeout if time_limit is None else max(self.timeout, time_limit + _WAIT_MARGIN)
        return _warn_if_lowered(Result._from_json(
            self._request("POST", "/v1/solve", _to_wire(model),
                          config=_meta_config(model, project, time_limit, config),
                          gzip=gzip, timeout=wait)))

    def submit(self, model: _Model, *, project: Optional[str] = None,
               time_limit: Optional[float] = None,
               config: Optional[Dict[str, Any]] = None, gzip: bool = False) -> "Job":
        """Submit ``model`` for asynchronous solving and return a handle to poll.

        Args:
            model: A Pyomo/MathOpt/PuLP model, a ``Program``, or encoded bytes.
            project: Optional project tag for the call (per-project invoicing), sent
                as a query param, not baked into the model.
            time_limit: Optional number of seconds the solve may run, as for
                :meth:`solve`. :meth:`Job.result` then waits at least this long
                plus a minute by default.
            config: Optional service parameters, sent as the query string.
            gzip: If ``True``, gzip-compress the request body.

        Returns:
            Job: A handle to the queued job; call :meth:`Job.result` to await it.

        Raises:
            QuicoptError: On a non-2xx response.
            TypeError: If ``time_limit`` is not a number.
            ValueError: If ``time_limit`` is not positive and finite.
        """
        time_limit = _check_time_limit(time_limit)
        body = self._request("POST", "/v1/jobs", _to_wire(model),
                             config=_meta_config(model, project, time_limit, config), gzip=gzip)
        self._remember(body.get("api_key"))     # /v1/jobs echoes a minted key in the 202 body too
        return Job(self, body["job_id"], time_limit)

    # ── transport ───────────────────────────────────────────────────────────

    def _open(self, method: str, path: str, data: Optional[bytes] = None, **kw) -> bytes:
        """Perform one HTTP request, retrying once if a *cached* key is rejected.

        A cache file can outlive the key it holds (the server was reset, the key was
        revoked, the file was copied from another machine), and a stale key 401s
        every call forever. So a 401 discards the cached key and retries **once**,
        keyless — which mints a fresh key and caches it.

        The retry is deliberately narrow: only a key read from disk is discarded,
        never one minted in this process and never a caller-supplied ``api_key``. A
        run can therefore mint at most one key beyond the stale one, so a server
        that rejected a key it just issued can never turn into a mint loop.

        Args:
            method: The HTTP method.
            path: The path under ``base_url`` (e.g. ``/v1/solve``).
            data: The request body, or ``None`` for a bodyless request.
            **kw: Forwarded to :meth:`_attempt` (``config``, ``gzip``, ``timeout``).

        Returns:
            bytes: The raw response body.

        Raises:
            QuicoptError: On a non-2xx response.
        """
        try:
            return self._attempt(method, path, data, **kw)
        except QuicoptError as e:
            if e.status_code != 401 or not self._from_cache:
                raise
            self._discard_key()
            return self._attempt(method, path, data, **kw)

    def _attempt(self, method: str, path: str, data: Optional[bytes] = None, *,
                 config: Optional[Dict[str, Any]] = None, gzip: bool = False,
                 timeout: Optional[float] = None) -> bytes:
        """Perform one HTTP request and return the raw response body.

        Sets the octet-stream content type (and gzip encoding, if requested) for a
        body, attaches the bearer key when known, and captures a freshly minted key
        off the response — including from an ``HTTPError``, since a 502/504 on a
        first call may still have minted one.

        Args:
            method: The HTTP method.
            path: The path under ``base_url`` (e.g. ``/v1/solve``).
            data: The request body, or ``None`` for a bodyless request.
            config: Optional service parameters, sent as the query string.
            gzip: If ``True``, gzip-compress ``data``.
            timeout: The socket timeout for this request, in seconds; ``None``
                uses the client's.

        Returns:
            bytes: The raw response body.

        Raises:
            QuicoptError: On a non-2xx response.
        """
        headers: Dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = _OCTET
            if gzip:
                data = _gzip.compress(data)
                headers["Content-Encoding"] = "gzip"
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base_url + path + _query(config),
                                     data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                self._capture_key(resp.headers)
                return resp.read()
        except urllib.error.HTTPError as e:
            self._capture_key(e.headers)        # a 504/502 on a first call still minted a key
            raise QuicoptError(e.code, _decode(e.read())) from None

    def _request(self, method: str, path: str, data: Optional[bytes] = None, **kw) -> Dict[str, Any]:
        """Perform a request via :meth:`_open` and parse its JSON body.

        Args:
            method: The HTTP method.
            path: The path under ``base_url``.
            data: The request body, or ``None``.
            **kw: Forwarded to :meth:`_open` (``config``, ``gzip``, ``timeout``).

        Returns:
            dict: The parsed JSON object, or ``{}`` for an empty body.

        Raises:
            QuicoptError: On a non-2xx response.
        """
        raw = self._open(method, path, data, **kw)
        return json.loads(raw) if raw else {}

    def _capture_key(self, headers) -> None:
        """Remember a freshly minted API key from a response, if we have none yet.

        The first keyless call mints a key, returned in the ``X-Quicopt-Api-Key``
        response header. A key we already hold is never overwritten.

        Args:
            headers: The response headers.

        Returns:
            None. ``self.api_key`` is set on first mint.
        """
        self._remember(headers.get("X-Quicopt-Api-Key"))

    def _remember(self, minted: Optional[str]) -> None:
        """Adopt ``minted`` as this client's key and cache it for the next run.

        Persisting is best-effort: an unwritable cache (read-only home, a container
        without ``$HOME``) warns rather than raises, since the solve itself
        succeeded and failing it over a cache miss would be worse than re-minting.

        Args:
            minted: The key the server returned, or ``None``/empty if it minted none.

        Returns:
            None.
        """
        if not minted or self.api_key:
            return
        self.api_key = minted
        if self._explicit or not self.cache:
            return
        try:
            _write_key(self.key_path, minted)
        except OSError as e:
            warnings.warn(f"could not cache the Quicopt free key at {self.key_path} "
                          f"({e}); every run will mint a new key — set "
                          f"${KEY_PATH_ENV} to a writable location", stacklevel=2)

    def _discard_key(self) -> None:
        """Drop the cached key the server just rejected, in memory and on disk.

        Returns:
            None. The next request goes out keyless and mints a replacement.
        """
        self.api_key = None
        self._from_cache = False
        if self.cache:
            with contextlib.suppress(OSError):      # an unremovable cache still gets re-minted past
                self.key_path.unlink(missing_ok=True)


def _decode(raw: bytes) -> Dict[str, Any]:
    """Best-effort decode of an error response body into a dict.

    Args:
        raw: The raw response body (from a non-2xx response).

    Returns:
        dict: The parsed JSON if it parses (or ``{}`` when empty), else the body
        wrapped as ``{"error": <text>}`` so a non-JSON error still surfaces.
    """
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"error": raw.decode("utf-8", "replace")}


@dataclass
class Job:
    """A handle to an async job. :meth:`result` polls until it finishes.
    ``time_limit`` is the one the job was submitted with, if any."""
    client: Client
    job_id: str
    time_limit: Optional[float] = None

    def status(self) -> Dict[str, Any]:
        """Fetch the job's metadata and framed ``log_tail``.

        Returns:
            dict: The job state (``queued``/``running``/``done``/``failed``) and its
            log tail, as returned by the service.
        """
        return self.client._request("GET", f"/v1/jobs/{self.job_id}")

    def result(self, *, wait: bool = True, timeout: Optional[float] = None,
               poll: float = 0.5) -> Result:
        """Fetch the job's result, optionally polling until it is ready.

        Args:
            wait: If ``True``, poll past the service's ``not_done`` reason until the
                worker finishes; if ``False``, fetch once.
            timeout: Maximum time to poll, in seconds, before giving up. Defaults to
                120 s, or the job's ``time_limit`` plus a minute when that is longer.
            poll: Delay between polls, in seconds.

        Returns:
            Result: The finished solve.

        Raises:
            QuicoptError: If ``wait`` is ``False`` and the job is not yet done, on
                any non-``not_done`` error, or once ``timeout`` elapses.
        """
        if timeout is None:
            timeout = 120.0 if self.time_limit is None else max(120.0, self.time_limit + _WAIT_MARGIN)
        deadline = time.monotonic() + timeout
        while True:
            try:
                return _warn_if_lowered(Result._from_json(
                    self.client._request("GET", f"/v1/jobs/{self.job_id}/result")))
            except QuicoptError as e:
                if not wait or e.reason != "not_done" or time.monotonic() > deadline:
                    raise
                time.sleep(poll)

    def log(self) -> str:
        """Fetch the job's plain-text log.

        Returns:
            str: The framed log view on success, or the error text on failure.
        """
        return self.client._open("GET", f"/v1/jobs/{self.job_id}/log").decode("utf-8", "replace")

    def delete(self) -> None:
        """Delete the job and its stored blob/result/log on the server.

        Returns:
            None.
        """
        self.client._open("DELETE", f"/v1/jobs/{self.job_id}")
