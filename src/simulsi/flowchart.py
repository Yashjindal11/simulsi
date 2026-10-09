"""Declarative flowchart models: build a simulation from YAML, no Python needed.

A flowchart describes where entities come from (``sources``), the
``stations`` they visit (a resource and a service time, or a plain delay),
how they move on (``next``: one station, or random routing by
probability) and where they leave (``exit`` or any name listed in
``sinks``). Numbers anywhere can refer to ``parameters`` as ``$name``, so a
flowchart works with experiments, ``simulsi whatif``, the dashboard and the
optimiser like any Python model.

.. code-block:: yaml

    flow:
      name: clinic
      duration: 480
      parameters:
        nurses: 2
        doctors: {default: 3, low: 1, description: doctors on shift}
        arrival_mean: 4.0
      resources:
        nurse: $nurses
        doctor: {capacity: $doctors}
      sources:
        - name: patient
          interarrival: {distribution: exponential, mean: $arrival_mean}
          next: triage
      stations:
        triage: {resource: nurse, service: {distribution: triangular, low: 2, mode: 4, high: 8},
                 next: {consult: 0.8, exit: rest}}
        consult: {resource: doctor, service: {distribution: exponential, mean: 10},
                  patience: 60, next: exit}
      presets:
        extra_doctor: {doctors: 4}

Two more station kinds: ``batch: 4`` (or ``batch: {size: 4, timeout: 30}``)
holds entities until a group is ready and does the step once for the whole
group (an oven, a shuttle); ``parallel: {lab: {...}, xray: {...}}`` splits
an entity into branches that run at the same time and joins when the
slowest finishes.

Load it with :func:`load_flowchart` (or pass the file to any CLI command
that takes a model). Files are data only: they can create registered
distributions and the building blocks below, never run code.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.errors import ConfigError
from simulsi.processes.flow import Router
from simulsi.processes.schedules import PiecewiseRate, arrivals, capacity_schedule
from simulsi.randomness.distributions import Distribution, as_distribution

MAX_VISITS = 1_000

_SOURCE_KEYS = {"name", "entity", "interarrival", "rate", "rate_table", "period", "limit", "next"}
_STEP_KEYS = {"resource", "service", "delay", "units", "patience", "priority"}
_STATION_KEYS = _STEP_KEYS | {"next", "batch", "parallel"}
_BATCH_KEYS = {"size", "timeout"}
_RESOURCE_KEYS = {"capacity", "discipline", "schedule", "period"}
_TOP_KEYS = {
    "name",
    "description",
    "version",
    "duration",
    "warmup",
    "parameters",
    "resources",
    "sources",
    "stations",
    "sinks",
    "presets",
    "outputs",
}


def _sub(value: Any, p: Mapping[str, Any]) -> Any:
    """Replace ``$name`` references with parameter values, recursively."""
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        if key not in p:
            raise ConfigError(f"unknown parameter reference {value!r}")
        return p[key]
    if isinstance(value, Mapping):
        return {k: _sub(v, p) for k, v in value.items()}
    if isinstance(value, list):
        return [_sub(v, p) for v in value]
    return value


def _dist(value: Any, where: str) -> Distribution[Any]:
    try:
        return as_distribution(value)
    except (TypeError, ConfigError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _check_keys(d: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def _parameters(raw: Mapping[str, Any]) -> list[Parameter]:
    out = []
    for name, spec in raw.items():
        if isinstance(spec, Mapping) and "default" in spec:
            extra = {
                k: spec[k] for k in ("kind", "low", "high", "description", "unit") if k in spec
            }
            inferred = Parameter.infer(name, spec["default"])
            out.append(Parameter(name, spec["default"], extra.pop("kind", inferred.kind), **extra))
        else:
            out.append(Parameter.infer(name, spec))
    return out


def _validate(spec: Mapping[str, Any]) -> None:
    _check_keys(spec, _TOP_KEYS, "flow")
    stations = spec.get("stations") or {}
    sinks = set(spec.get("sinks") or []) | {"exit"}
    sources = spec.get("sources") or []
    if not sources:
        raise ConfigError("flow needs at least one source")
    if not isinstance(stations, Mapping):
        raise ConfigError("flow.stations must be a mapping of name -> station")
    clash = sinks & set(stations)
    if clash:
        raise ConfigError(f"names used as both station and sink: {sorted(clash)}")
    resources = spec.get("resources") or {}

    def check_next(nxt: Any, where: str) -> None:
        targets = [nxt] if isinstance(nxt, str) else list(nxt) if isinstance(nxt, Mapping) else None
        if not targets:
            raise ConfigError(f"{where}: 'next' must be a name or a mapping of name -> probability")
        for t in targets:
            if t not in stations and t not in sinks:
                raise ConfigError(f"{where}: unknown destination {t!r}")

    for i, s in enumerate(sources):
        where = f"source {s.get('name', i)!r}"
        _check_keys(s, _SOURCE_KEYS, where)
        if sum(k in s for k in ("interarrival", "rate", "rate_table")) != 1:
            raise ConfigError(f"{where}: give exactly one of interarrival, rate or rate_table")
        check_next(s.get("next"), where)

    def check_step(st: Any, where: str) -> None:
        if not isinstance(st, Mapping):
            raise ConfigError(f"{where} must be a mapping")
        if "resource" in st:
            if st["resource"] not in resources:
                raise ConfigError(f"{where}: unknown resource {st['resource']!r}")
            if "service" not in st:
                raise ConfigError(f"{where}: a resource station needs 'service'")
        elif "delay" not in st:
            raise ConfigError(f"{where}: give 'resource' + 'service', or 'delay'")

    for name, st in stations.items():
        where = f"station {name!r}"
        if not isinstance(st, Mapping):
            raise ConfigError(f"{where} must be a mapping")
        _check_keys(st, _STATION_KEYS, where)
        if "parallel" in st:
            branches = st["parallel"]
            if not isinstance(branches, Mapping) or not branches:
                raise ConfigError(f"{where}: 'parallel' must map branch name -> step")
            if set(st) & _STEP_KEYS:
                raise ConfigError(f"{where}: a parallel station has only 'parallel' and 'next'")
            for b, step in branches.items():
                bw = f"{where} branch {b!r}"
                if isinstance(step, Mapping):
                    _check_keys(step, _STEP_KEYS, bw)
                check_step(step, bw)
        else:
            check_step(st, where)
        if "batch" in st:
            b = st["batch"]
            if isinstance(b, Mapping):
                _check_keys(b, _BATCH_KEYS, f"{where} batch")
            elif not isinstance(b, (int, str)):
                raise ConfigError(f"{where}: 'batch' must be a size or {{size, timeout}}")
        check_next(st.get("next"), where)
    for name, r in resources.items():
        if isinstance(r, Mapping):
            _check_keys(r, _RESOURCE_KEYS, f"resource {name!r}")


def flowchart_model(spec: Mapping[str, Any]) -> Model:
    """Build a :class:`~simulsi.Model` from a flowchart mapping (the content under ``flow:``)."""
    spec = dict(spec)
    _validate(spec)
    params = _parameters(spec.get("parameters") or {})
    sinks = set(spec.get("sinks") or []) | {"exit"}

    def build(sim: Simulation, p: Params) -> None:
        pv = {k: p[k] for k in p}
        res_specs = _sub(spec.get("resources") or {}, pv)
        stations = _sub(spec["stations"], pv)
        sources = _sub(spec["sources"], pv)
        resources = {}
        for name, r in res_specs.items():
            r = r if isinstance(r, Mapping) else {"capacity": r}
            resources[name] = sim.resource(
                name, int(r.get("capacity", 1)), discipline=r.get("discipline", "fifo")
            )
            if "schedule" in r:
                capacity_schedule(
                    sim, resources[name], [tuple(x) for x in r["schedule"]], period=r.get("period")
                )
        routers: dict[str, Router] = {}

        def route(origin: str, nxt: Any) -> str:
            if isinstance(nxt, str):
                return nxt
            if origin not in routers:
                probs = {str(k): v for k, v in nxt.items()}
                rest = [k for k, v in probs.items() if v == "rest"]
                if len(rest) > 1:
                    raise ConfigError(f"{origin}: only one destination can take 'rest'")
                fixed = {k: float(v) for k, v in probs.items() if v != "rest"}
                if rest:
                    fixed[rest[0]] = max(0.0, 1.0 - sum(fixed.values()))
                routers[origin] = Router(sim, origin, fixed)
            return routers[origin].choose()

        steps: dict[str, Mapping[str, Any]] = {}
        for name, st in stations.items():
            if "parallel" in st:
                for b, step in st["parallel"].items():
                    steps[f"{name}.{b}"] = step
            else:
                steps[name] = st
        streams = {name: sim.stream(f"service:{name}") for name in steps}
        dists = {
            name: _dist(st.get("service", st.get("delay")), f"station {name!r}")
            for name, st in steps.items()
        }
        patience = {
            name: _dist(st["patience"], f"station {name!r} patience")
            for name, st in steps.items()
            if "patience" in st
        }
        for name in [*stations, *(s for s in steps if s not in stations)]:
            sim.metrics.counter(f"station.{name}.visits")
            if name in patience:
                sim.metrics.counter(f"station.{name}.abandoned")
        for s in sinks:
            sim.metrics.counter(f"sink.{s}")

        def do_step(entity: Any, key: str) -> Any:
            """Seize the resource (if any) and spend the service time; returns True if abandoned."""
            st = steps[key]
            duration = max(0.0, float(dists[key].sample(streams[key])))
            if "resource" not in st:
                yield duration
                return False
            res = resources[st["resource"]]
            reqs: list[Any] = []
            pat = float(patience[key].sample(streams[key])) if key in patience else None
            for _ in range(int(st.get("units", 1))):
                req = yield sim.request(
                    res, priority=float(st.get("priority", 0)), entity=entity, patience=pat
                )
                if not req.granted:
                    for r in reqs:
                        res.release(r)
                    sim.metrics.increment(f"station.{key}.abandoned")
                    return True
                reqs.append(req)
            yield duration
            for r in reqs:
                res.release(r)
            return False

        def parallel_step(entity: Any, where: str) -> Any:
            """Split into one branch per step, run them at once and join when all finish."""
            procs = []
            for b in stations[where]["parallel"]:
                sim.metrics.increment(f"station.{where}.{b}.visits")
                procs.append(sim.process(do_step(entity, f"{where}.{b}"), name=f"{where}.{b}"))
            yield sim.all_of(procs)
            return any(p.value for p in procs)

        gates: dict[str, dict[str, Any]] = {}

        def batch_step(entity: Any, where: str) -> Any:
            """Wait for a full batch (or the timeout), then do the step once for all of them."""
            b = stations[where]["batch"]
            size = int(b["size"] if isinstance(b, Mapping) else b)
            timeout = b.get("timeout") if isinstance(b, Mapping) else None
            gate = gates.get(where)
            leader = gate is None
            if gate is None:
                gate = gates[where] = {"n": 0, "full": sim.signal(), "done": sim.signal()}
            gate["n"] += 1
            if gate["n"] >= size:
                gates.pop(where, None)
                gate["full"].succeed()
            if not leader:
                return bool((yield gate["done"]))
            if not gate["full"].triggered:
                if timeout is None:
                    yield gate["full"]
                else:
                    yield sim.any_of(gate["full"], sim.timeout(float(timeout)))
            if gates.get(where) is gate:
                del gates[where]
            sim.metrics.observe(f"station.{where}.batch_size", gate["n"])
            abandoned = yield from do_step(entity, where)
            gate["done"].succeed(abandoned)
            return abandoned

        def journey(entity: Any, first: Any, origin: str) -> Any:
            where = route(origin, first)
            visits = 0
            while where not in sinks:
                visits += 1
                if visits > MAX_VISITS:
                    raise ConfigError(
                        f"an entity visited {MAX_VISITS} stations; check for a routing loop"
                    )
                st = stations[where]
                sim.metrics.increment(f"station.{where}.visits")
                if "parallel" in st:
                    abandoned = yield from parallel_step(entity, where)
                elif "batch" in st:
                    abandoned = yield from batch_step(entity, where)
                else:
                    abandoned = yield from do_step(entity, where)
                if abandoned:
                    where = "abandoned"
                    break
                where = route(where, st["next"])
            sim.metrics.increment(f"sink.{where}")
            sim.dispose(entity, where)

        def source(gap: Distribution[Any], stream: Any, spawn: Any, limit: Any) -> Any:
            k = 0
            while limit is None or k < int(limit):
                yield max(0.0, float(gap.sample(stream)))
                sim.process(spawn(sim, k))
                k += 1

        for i, src in enumerate(sources):
            sname = str(src.get("name", f"source{i}"))
            etype = str(src.get("entity", sname))
            nxt = src["next"]

            def spawn(
                sim: Simulation, k: int, etype: str = etype, nxt: Any = nxt, sname: str = sname
            ) -> Any:
                return journey(sim.entity(etype), nxt, f"source:{sname}")

            if "interarrival" in src:
                gap = _dist(src["interarrival"], f"source {sname!r}")
                stream = sim.stream(f"arrivals:{sname}")
                sim.process(source(gap, stream, spawn, src.get("limit")), name=f"source:{sname}")
            else:
                rate: Any = (
                    float(src["rate"])
                    if "rate" in src
                    else PiecewiseRate(
                        [tuple(x) for x in src["rate_table"]], period=src.get("period")
                    )
                )
                arrivals(
                    sim,
                    rate,
                    spawn,
                    stream=f"arrivals:{sname}",
                    limit=src.get("limit"),
                    name=f"source:{sname}",
                )

    if "abandoned" not in sinks:
        sinks.add("abandoned")
    resource_names = list((spec.get("resources") or {}).keys())
    etypes = {
        str(s.get("entity", s.get("name", f"source{i}"))) for i, s in enumerate(spec["sources"])
    }
    outputs = spec.get("outputs") or (
        [f"entity.{e}.time_in_system.mean" for e in sorted(etypes)]
        + [f"resource.{r}.{k}" for r in resource_names for k in ("utilization", "wait.mean")]
    )
    return Model(
        build,
        name=str(spec.get("name", "flowchart")),
        duration=float(spec["duration"]) if spec.get("duration") is not None else None,
        warmup=float(spec.get("warmup", 0.0)),
        parameters=params,
        version=str(spec.get("version", "1")),
        description=str(
            spec.get("description", "") or f"Flowchart model {spec.get('name', '')}".strip()
        ),
        outputs=outputs,
        presets=spec.get("presets") or {},
    )


def is_flowchart(path: str | Path) -> bool:
    """True if ``path`` is a YAML file with a top-level ``flow:`` section."""
    p = Path(path)
    if p.suffix.lower() not in (".yaml", ".yml") or not p.is_file():
        return False
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return False
    return isinstance(data, Mapping) and "flow" in data


def load_flowchart(source: str | Path | Mapping[str, Any]) -> Model:
    """Load a flowchart model from a YAML file, YAML text or an already-parsed mapping."""
    if isinstance(source, Mapping):
        data: Any = source
    else:
        text = (
            Path(source).read_text(encoding="utf-8")
            if Path(str(source)).suffix in (".yaml", ".yml")
            else str(source)
        )
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("a flowchart must be a mapping")
    if "flow" in data:
        data = data["flow"]
    if not isinstance(data, Mapping):
        raise ConfigError("'flow' must be a mapping")
    return flowchart_model(data)
