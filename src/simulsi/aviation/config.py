"""Operating assumptions for the airline network simulation.

:class:`OpsConfig` holds the rules of the day (turn times, runway rates,
gates, curfews, crew duty limits, spares, standby crews, recovery
thresholds, costs). :class:`DelayModel` says how much *primary* delay
(technical, ATC, late passengers - anything not caused by an earlier
flight) each departure gets and how variable block times are; it can be
fitted to history with :func:`simulsi.aviation.fit_delay_model`.
:class:`WeatherEvent` cuts an airport's departure rate for a time window,
with a probability, so forecasts can carry weather scenarios.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from simulsi.aviation.schedule import Flight, parse_time
from simulsi.errors import ConfigError
from simulsi.randomness.stream import RandomStream

DelayPredictor = Callable[[Flight], float]


@dataclass
class DelayModel:
    """Primary departure delays and block-time variability.

    Each departure gets a primary delay with probability ``prob``; its size
    is exponential with mean ``mean`` minutes. ``table`` overrides both per
    origin airport (``"LHR"``) or per origin and hour (``"LHR@07"``), and
    ``block_bias`` scales the scheduled block time per route
    (``"LHR-CDG"``). ``predictor`` - for example a machine-learning model -
    returns the expected primary delay of a flight; the simulation then
    samples around that expectation and propagates it through the network.
    """

    prob: float = 0.2
    mean: float = 25.0
    block_cv: float = 0.06
    table: dict[str, tuple[float, float]] = field(default_factory=dict)
    block_bias: dict[str, float] = field(default_factory=dict)
    predictor: DelayPredictor | None = None

    def primary(self, flight: Flight, rs: RandomStream) -> float:
        if self.predictor is not None:
            expected = max(0.0, float(self.predictor(flight)))
            return rs.exponential(expected) if expected > 0 else 0.0
        hour = int(flight.std // 60) % 24
        prob, mean = self.table.get(
            f"{flight.origin}@{hour:02d}", self.table.get(flight.origin, (self.prob, self.mean))
        )
        if prob <= 0 or rs.random() >= prob:
            return 0.0
        return rs.exponential(mean)

    def block(self, flight: Flight, rs: RandomStream) -> float:
        planned = flight.block * self.block_bias.get(f"{flight.origin}-{flight.dest}", 1.0)
        if self.block_cv <= 0:
            return planned
        sigma = math.sqrt(math.log1p(self.block_cv**2))
        return planned * math.exp(rs.normal(-0.5 * sigma * sigma, sigma))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prob": self.prob,
            "mean": self.mean,
            "block_cv": self.block_cv,
            "table": {k: list(v) for k, v in self.table.items()},
            "block_bias": dict(self.block_bias),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DelayModel:
        known = {"prob", "mean", "block_cv", "table", "block_bias"}
        extra = set(data) - known
        if extra:
            raise ConfigError(f"unknown delay settings: {', '.join(sorted(extra))}")
        table = {str(k): (float(v[0]), float(v[1])) for k, v in (data.get("table") or {}).items()}
        return cls(
            prob=float(data.get("prob", 0.2)),
            mean=float(data.get("mean", 25.0)),
            block_cv=float(data.get("block_cv", 0.06)),
            table=table,
            block_bias={str(k): float(v) for k, v in (data.get("block_bias") or {}).items()},
        )


@dataclass(frozen=True)
class WeatherEvent:
    """Departure capacity at ``airport`` falls to ``capacity`` x normal from ``start`` to ``end``.

    ``probability`` is the chance the event happens on the day (each
    replication draws it); ``capacity=0`` closes the airport to departures.
    """

    airport: str
    start: float
    end: float
    capacity: float = 0.4
    probability: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ConfigError(f"weather at {self.airport}: end must be after start")
        if not 0 <= self.capacity <= 1:
            raise ConfigError("weather capacity must be in [0, 1]")
        if not 0 <= self.probability <= 1:
            raise ConfigError("weather probability must be in [0, 1]")

    @classmethod
    def parse(cls, text: str) -> WeatherEvent:
        """``"HUB 15:00-18:00 0.4 p=0.6"`` (capacity and probability optional)."""
        parts = text.split()
        if len(parts) < 2 or "-" not in parts[1]:
            raise ConfigError(
                f"weather must look like 'HUB 15:00-18:00 [capacity] [p=prob]': {text!r}"
            )
        start, _, end = parts[1].partition("-")
        cap, prob = 0.4, 1.0
        for extra in parts[2:]:
            if extra.startswith("p="):
                prob = float(extra[2:])
            else:
                cap = float(extra)
        return cls(
            parts[0].upper(),
            parse_time(start),
            parse_time(end),
            cap,
            prob,
            f"weather {parts[0].upper()}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def thunderstorm(
    airport: str, start: str | float, hours: float = 3.0, probability: float = 1.0
) -> WeatherEvent:
    """Convective weather: departure rate cut to 40 %."""
    s = parse_time(start)
    return WeatherEvent(airport, s, s + 60 * hours, 0.4, probability, f"thunderstorm {airport}")


def morning_fog(
    airport: str, until: str | float = "10:00", probability: float = 1.0
) -> WeatherEvent:
    """Low visibility procedures from 05:00: departure rate cut to 60 %."""
    return WeatherEvent(airport, 300.0, parse_time(until), 0.6, probability, f"fog {airport}")


def snow(
    airport: str, start: str | float = "06:00", hours: float = 6.0, probability: float = 1.0
) -> WeatherEvent:
    """Snow and de-icing: departure rate cut to 50 % for a long window."""
    s = parse_time(start)
    return WeatherEvent(airport, s, s + 60 * hours, 0.5, probability, f"snow {airport}")


def closure(
    airport: str, start: str | float, minutes: float, probability: float = 1.0
) -> WeatherEvent:
    """Full closure to departures (storm, runway incident, ATC strike)."""
    s = parse_time(start)
    return WeatherEvent(airport, s, s + minutes, 0.0, probability, f"closure {airport}")


@dataclass
class OpsConfig:
    """Rules and resources for one day of operations.

    Per-airport settings are dictionaries keyed by airport code; airports
    not listed are unconstrained (no runway queue, unlimited gates, no
    curfew, no spares or standby crews).
    """

    on_time: float = 15.0
    min_turn: float = 35.0
    min_turn_by_airport: dict[str, float] = field(default_factory=dict)
    runway_rate: dict[str, float] = field(default_factory=dict)
    weather_base_rate: float = 30.0
    gates: dict[str, int] = field(default_factory=dict)
    curfew: dict[str, float] = field(default_factory=dict)
    min_crew_connect: float = 30.0
    duty_limit: float = 13 * 60.0
    report_time: float = 60.0
    release_time: float = 30.0
    standby_callout: float = 60.0
    spares: dict[str, int] = field(default_factory=dict)
    standby_crews: dict[str, int] = field(default_factory=dict)
    swap_threshold: float = 90.0
    cancel_threshold: float = 240.0
    mct: float = 35.0
    delay_cost_per_minute: float = 100.0
    cancel_cost: float = 20000.0
    misconnect_cost_per_pax: float = 250.0
    eu261: bool = False
    delays: DelayModel = field(default_factory=DelayModel)

    def turn(self, airport: str) -> float:
        return self.min_turn_by_airport.get(airport, self.min_turn)

    def replace(self, **changes: Any) -> OpsConfig:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in changes.items():
            if key not in data:
                raise ConfigError(f"unknown ops setting {key!r}")
            data[key] = value
        return OpsConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "delays"}
        out["delays"] = self.delays.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OpsConfig:
        names = {f.name for f in fields(cls)}
        extra = set(data) - names - {"weather"}
        if extra:
            raise ConfigError(f"unknown ops settings: {', '.join(sorted(extra))}")
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key == "weather":
                continue
            if key == "delays":
                kwargs[key] = DelayModel.from_dict(value or {})
            elif key == "curfew":
                kwargs[key] = {str(a).upper(): parse_time(t) for a, t in (value or {}).items()}
            elif isinstance(value, Mapping):
                kwargs[key] = {str(a).upper(): v for a, v in value.items()}
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> OpsConfig:
        return cls.from_dict(_load_yaml(path))

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text


def weather_from_config(data: Mapping[str, Any]) -> list[WeatherEvent]:
    """The ``weather:`` list of an ops YAML file (strings or mappings)."""
    out = []
    for item in data.get("weather") or []:
        if isinstance(item, str):
            out.append(WeatherEvent.parse(item))
        else:
            d = dict(item)
            d["airport"] = str(d["airport"]).upper()
            d["start"], d["end"] = parse_time(d["start"]), parse_time(d["end"])
            out.append(WeatherEvent(**d))
    return out


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of settings")
    return data
