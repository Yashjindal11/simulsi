"""Scenarios: named parameter sets describing a "what if"."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from simulsi.errors import ConfigError


@dataclass(frozen=True)
class Scenario:
    """A named set of parameter values.

    Scenarios only hold *overrides*; the model supplies defaults for anything
    not mentioned. Derive variants instead of copying dictionaries:

    >>> base = Scenario("baseline", {"arrival_rate": 1.0, "capacity": 3})
    >>> base.derive("high_demand", arrival_rate=1.5).parameters["arrival_rate"]
    1.5
    >>> base.scale("double_capacity", capacity=2).parameters["capacity"]
    6
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ConfigError("scenario name must be a non-empty string")
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "tags", tuple(self.tags))

    def derive(self, name: str, description: str = "", **overrides: Any) -> Scenario:
        """A new scenario with some parameters replaced."""
        return Scenario(name, {**self.parameters, **overrides}, description, self.tags)

    def scale(self, name: str, description: str = "", **factors: float) -> Scenario:
        """A new scenario with numeric parameters multiplied by ``factors``.

        Integer parameters stay integers (rounded), so capacities remain valid.
        """
        params = dict(self.parameters)
        for key, factor in factors.items():
            if key not in params:
                raise ConfigError(f"cannot scale {key!r}: not set in scenario {self.name!r}")
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"cannot scale non-numeric parameter {key!r}={value!r}")
            scaled = value * factor
            params[key] = round(scaled) if isinstance(value, int) else scaled
        return Scenario(name, params, description, self.tags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "description": self.description,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scenario:
        try:
            return cls(
                name=str(data["name"]),
                parameters=dict(data.get("parameters", {})),
                description=str(data.get("description", "")),
                tags=tuple(data.get("tags", ())),
            )
        except KeyError as exc:
            raise ConfigError(f"scenario is missing {exc.args[0]!r}") from exc


BASELINE = "baseline"


def baseline(parameters: Mapping[str, Any] | None = None, description: str = "") -> Scenario:
    return Scenario(BASELINE, dict(parameters or {}), description or "model defaults")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def grid(
    axes: Mapping[str, Sequence[Any]],
    base: Scenario | Mapping[str, Any] | None = None,
    *,
    prefix: str = "",
) -> list[Scenario]:
    """Full factorial design: one scenario per combination of axis values.

    >>> " | ".join(s.name for s in grid({"capacity": [1, 2], "rate": [0.5]}))
    'capacity=1,rate=0.5 | capacity=2,rate=0.5'
    """
    if not axes:
        raise ConfigError("grid needs at least one axis")
    for key, values in axes.items():
        if isinstance(values, str | bytes) or not isinstance(values, Sequence) or not values:
            raise ConfigError(f"grid axis {key!r} must be a non-empty list of values")
    base_s = base if isinstance(base, Scenario) else Scenario(BASELINE, dict(base or {}))
    keys = list(axes)
    out: list[Scenario] = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        name = prefix + ",".join(f"{k}={_fmt(v)}" for k, v in overrides.items())
        out.append(base_s.derive(name, **overrides))
    return out


def one_at_a_time(base: Scenario, changes: Mapping[str, Iterable[Any]]) -> list[Scenario]:
    """Vary one parameter at a time around ``base`` (the base itself is not included)."""
    out = []
    for key, values in changes.items():
        for v in values:
            out.append(base.derive(f"{key}={_fmt(v)}", **{key: v}))
    return out


def check_unique_names(scenarios: Iterable[Scenario]) -> None:
    seen: set[str] = set()
    for s in scenarios:
        if s.name in seen:
            raise ConfigError(f"duplicate scenario name {s.name!r}")
        seen.add(s.name)
