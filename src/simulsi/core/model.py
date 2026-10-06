"""Models: reusable, parameterised simulation definitions.

A :class:`Model` couples a *build function* ``build(sim, params)`` - which
creates resources and starts processes on a fresh :class:`Simulation` - with
a parameter schema, a run length and an optional warm-up. Models are what
experiments, Monte Carlo studies, sensitivity analysis, validation, the CLI
and optimisers all operate on.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

from simulsi.core.simulation import Simulation, SimulationResult
from simulsi.errors import ConfigError
from simulsi.randomness.distributions import Distribution, as_distribution, is_distribution_spec


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()

ParameterKind = Literal["any", "int", "float", "bool", "str", "probability", "distribution"]


@dataclass(frozen=True)
class Parameter:
    """Schema for one model parameter."""

    name: str
    default: Any = REQUIRED
    kind: ParameterKind = "any"
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""
    unit: str = ""

    @classmethod
    def infer(cls, name: str, default: Any) -> Parameter:
        if isinstance(default, Parameter):
            return default
        kind: ParameterKind = "any"
        if isinstance(default, bool):
            kind = "bool"
        elif isinstance(default, int):
            kind = "int"
        elif isinstance(default, float):
            kind = "float"
        elif isinstance(default, str):
            kind = "str"
        elif isinstance(default, Distribution) or is_distribution_spec(default):
            kind = "distribution"
        return cls(name, default, kind)

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def coerce(self, value: Any) -> Any:
        """Validate and normalise ``value``; raises :class:`ConfigError`."""
        k = self.kind
        try:
            if k in ("int",):
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise TypeError
                if isinstance(value, float) and not value.is_integer():
                    raise TypeError
                value = int(value)
            elif k in ("float", "probability"):
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise TypeError
                value = float(value)
                if math.isnan(value):
                    raise ValueError(f"parameter {self.name!r} must not be NaN")
            elif k == "bool":
                if not isinstance(value, bool):
                    raise TypeError
            elif k == "str":
                if not isinstance(value, str):
                    raise TypeError
            elif k == "distribution":
                value = as_distribution(value)
        except TypeError as exc:
            raise ConfigError(
                f"parameter {self.name!r} expects {k}, got {type(value).__name__} {value!r}"
            ) from exc
        except (ValueError, ConfigError) as exc:
            raise ConfigError(f"parameter {self.name!r}: {exc}") from exc
        if k == "probability" and not 0.0 <= value <= 1.0:
            raise ConfigError(f"parameter {self.name!r} is a probability; got {value}")
        if isinstance(value, int | float) and not isinstance(value, bool):
            if self.low is not None and value < self.low:
                raise ConfigError(f"parameter {self.name!r}={value} is below minimum {self.low}")
            if self.high is not None and value > self.high:
                raise ConfigError(f"parameter {self.name!r}={value} is above maximum {self.high}")
        if self.choices is not None and value not in self.choices:
            raise ConfigError(f"parameter {self.name!r}={value!r} not in {list(self.choices)}")
        return value

    def to_dict(self) -> dict[str, Any]:
        default = None if self.required else self.default
        if isinstance(default, Distribution):
            default = default.to_spec() if default.kind else repr(default)
        return {
            "name": self.name,
            "default": default,
            "required": self.required,
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "choices": list(self.choices) if self.choices is not None else None,
            "description": self.description,
            "unit": self.unit,
        }


class Params(Mapping[str, Any]):
    """Read-only resolved parameters; values are reachable as keys or attributes."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"no parameter named {key!r}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Params are read-only")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Params({self._data!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (Params, (self._data,))

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._data.items():
            if isinstance(v, Distribution):
                out[k] = v.to_spec() if v.kind else repr(v)
            else:
                out[k] = v
        return out


BuildFn = Callable[[Simulation, Params], Any]


def _import_object(module: str, qualname: str) -> Any:
    import importlib

    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _new_model() -> Model:
    return Model.__new__(Model)


class Model:
    """A parameterised simulation model.

    >>> def build(sim, p):
    ...     sim.metrics.set("answer", p.x * 2)
    >>> m = Model(build, name="demo", duration=1.0, parameters={"x": 21})
    >>> m.simulate(seed=1).metrics["answer"]
    42.0
    """

    def __init__(
        self,
        build: BuildFn,
        *,
        name: str | None = None,
        duration: float | None = None,
        warmup: float = 0.0,
        parameters: Mapping[str, Any] | Sequence[Parameter] | None = None,
        version: str = "0",
        description: str = "",
        outputs: Sequence[str] | None = None,
        sim_options: Mapping[str, Any] | None = None,
        strict: bool = True,
    ) -> None:
        if not callable(build):
            raise TypeError("build must be callable: build(sim, params)")
        if duration is not None and not (duration > 0 and math.isfinite(duration)):
            raise ConfigError(f"duration must be a positive finite number, got {duration}")
        if warmup < 0 or (duration is not None and warmup >= duration):
            raise ConfigError(f"warmup must be >= 0 and shorter than duration, got {warmup}")
        self.build = build
        self.name: str = name or str(getattr(build, "__name__", "model"))
        self.duration = duration
        self.warmup = warmup
        self.version = version
        self.description = description or (build.__doc__ or "").strip().split("\n")[0]
        self.outputs = list(outputs) if outputs else []
        self.sim_options = dict(sim_options or {})
        self.strict = strict
        if parameters is None:
            specs: list[Parameter] = []
        elif isinstance(parameters, Mapping):
            specs = [Parameter.infer(k, v) for k, v in parameters.items()]
        else:
            specs = list(parameters)
        self.parameters: dict[str, Parameter] = {}
        for spec in specs:
            if spec.name in self.parameters:
                raise ConfigError(f"duplicate parameter {spec.name!r}")
            if not spec.required:
                spec.coerce(spec.default)
            self.parameters[spec.name] = spec

    # -- parameters ------------------------------------------------------------

    @property
    def defaults(self) -> dict[str, Any]:
        return {k: p.default for k, p in self.parameters.items() if not p.required}

    def check_parameters(self, overrides: Mapping[str, Any] | None = None) -> list[str]:
        """All problems with ``overrides`` (empty list when valid)."""
        issues: list[str] = []
        given = dict(overrides or {})
        if self.strict:
            for key in given:
                if key not in self.parameters:
                    known = ", ".join(sorted(self.parameters)) or "none"
                    issues.append(f"unknown parameter {key!r} (known: {known})")
        for name, spec in self.parameters.items():
            if name in given:
                try:
                    spec.coerce(given[name])
                except ConfigError as exc:
                    issues.append(str(exc))
            elif spec.required:
                issues.append(f"missing required parameter {name!r}")
        return issues

    def resolve(self, overrides: Mapping[str, Any] | None = None) -> Params:
        issues = self.check_parameters(overrides)
        if issues:
            raise ConfigError("; ".join(issues))
        given = dict(overrides or {})
        values: dict[str, Any] = {}
        for name, spec in self.parameters.items():
            values[name] = spec.coerce(given[name]) if name in given else spec.default
            if isinstance(values[name], Mapping) and spec.kind == "distribution":
                values[name] = as_distribution(values[name])
        for key, value in given.items():
            if key not in values:
                values[key] = value
        return Params(values)

    # -- running -----------------------------------------------------------------

    def create(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        trace: bool = False,
        warmup: float | None = None,
        **sim_options: Any,
    ) -> Simulation:
        """Build (but do not run) a simulation - handy for stepping and inspection."""
        p = params if isinstance(params, Params) else self.resolve(params)
        options = {**self.sim_options, **sim_options}
        sim = Simulation(seed=seed, name=self.name, trace=trace, **options)
        self.build(sim, p)
        origin = {"trace": trace, **options}
        if warmup is not None:
            if warmup < 0:
                raise ConfigError(f"warmup must be >= 0, got {warmup}")
            origin["warmup"] = warmup
        sim._origin = (self, p, origin)
        effective = self.warmup if warmup is None else warmup
        if effective > 0:
            sim.warmup(effective)
        return sim

    def simulate(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        duration: float | None = None,
        warmup: float | None = None,
        trace: bool = False,
        **sim_options: Any,
    ) -> SimulationResult:
        """Run one replication and return its result.

        If ``duration`` is shorter than the model's warm-up and no ``warmup`` is
        given, the warm-up is scaled to the same fraction of the shorter run.
        """
        p = params if isinstance(params, Params) else self.resolve(params)
        horizon = duration if duration is not None else self.duration
        note = None
        if (
            warmup is None
            and self.warmup > 0
            and horizon is not None
            and self.duration is not None
            and horizon <= self.warmup
        ):
            warmup = self.warmup * horizon / self.duration
            note = f"warm-up scaled from {self.warmup:g} to {warmup:g} for run length {horizon:g}"
        sim = self.create(p, seed=seed, trace=trace, warmup=warmup, **sim_options)
        effective = self.warmup if warmup is None else warmup
        if effective > 0 and horizon is not None and horizon <= effective:
            sim.warn(
                f"run length {horizon:g} does not exceed the warm-up {effective:g}; statistics were not reset"
            )
        if note:
            sim.warn(note)
        result = sim.run(until=horizon)
        result.details["parameters"] = p.to_jsonable()
        result.details["model"] = {"name": self.name, "version": self.version}
        return result

    def evaluate(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        metric: str,
        replications: int = 10,
        seed: int = 0,
    ) -> float:
        """Mean of ``metric`` over ``replications`` runs with common random numbers.

        This is the bridge to optimisers: the same ``seed`` gives the same
        random streams for every parameter vector, which removes much of the
        noise when comparing candidates.
        """
        from simulsi.randomness.stream import derive_seed

        if replications < 1:
            raise ValueError("replications must be >= 1")
        total = 0.0
        for r in range(replications):
            m = self.simulate(params, seed=derive_seed(seed, "replication", r)).metrics
            if metric not in m:
                raise KeyError(f"model {self.name!r} produced no metric {metric!r}")
            total += m[metric]
        return total / replications

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "duration": self.duration,
            "warmup": self.warmup,
            "parameters": [p.to_dict() for p in self.parameters.values()],
            "outputs": list(self.outputs),
        }

    def __call__(self, sim: Simulation, params: Params) -> Any:
        return self.build(sim, params)

    def __reduce__(self) -> tuple[Any, ...]:
        # `@model` replaces the module-level function with this Model, so pickle
        # by reference to that global when possible (needed for worker processes).
        module = getattr(self.build, "__module__", None)
        qualname = getattr(self.build, "__qualname__", "").removesuffix(".build")
        if module and qualname and "<" not in qualname:
            try:
                if _import_object(module, qualname) is self:
                    return (_import_object, (module, qualname))
            except (ImportError, AttributeError):
                pass
        return (_new_model, (), self.__dict__)

    def with_options(
        self,
        *,
        duration: float | None = None,
        warmup: float | None = None,
        name: str | None = None,
        sim_options: Mapping[str, Any] | None = None,
    ) -> Model:
        """A copy with a different run length, warm-up, name or simulation options."""
        return Model(
            self.build,
            name=name or self.name,
            duration=duration if duration is not None else self.duration,
            warmup=warmup if warmup is not None else self.warmup,
            parameters=list(self.parameters.values()),
            version=self.version,
            description=self.description,
            outputs=self.outputs,
            sim_options={**self.sim_options, **(sim_options or {})},
            strict=self.strict,
        )

    def __repr__(self) -> str:
        return f"Model({self.name!r}, duration={self.duration}, parameters={list(self.parameters)})"


@overload
def model(build: BuildFn, /) -> Model: ...
@overload
def model(
    build: None = None,
    /,
    *,
    name: str | None = ...,
    duration: float | None = ...,
    warmup: float = ...,
    parameters: Mapping[str, Any] | Sequence[Parameter] | None = ...,
    version: str = ...,
    description: str = ...,
    outputs: Sequence[str] | None = ...,
    sim_options: Mapping[str, Any] | None = ...,
    strict: bool = ...,
) -> Callable[[BuildFn], Model]: ...
def model(build: BuildFn | None = None, /, **kwargs: Any) -> Model | Callable[[BuildFn], Model]:
    """Decorator turning a build function into a :class:`Model`.

    >>> @model(duration=100, parameters={"servers": 2})
    ... def bank(sim, p):
    ...     sim.resource("teller", p.servers)
    >>> bank.simulate(seed=1).metrics["resource.teller.utilization"]
    0.0
    """
    if build is not None:
        return _decorate(build, kwargs)

    def wrap(fn: BuildFn) -> Model:
        return _decorate(fn, kwargs)

    return wrap


def _decorate(fn: BuildFn, kwargs: Mapping[str, Any]) -> Model:
    m = Model(fn, **kwargs)
    # The decorated name now refers to the Model; point the function's qualified
    # name at `<name>.build` so pickle can still find the function (and copies
    # made with `with_options` remain usable in worker processes).
    qn = getattr(fn, "__qualname__", None)
    if isinstance(qn, str) and "<" not in qn:
        with contextlib.suppress(AttributeError, TypeError):
            fn.__qualname__ = f"{qn}.build"
    return m
