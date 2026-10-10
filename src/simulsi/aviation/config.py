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
from collections.abc import Callable, Mapping, Sequence
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

    ``shape`` replaces the exponential with an empirical distribution:
    equally spaced quantiles of primary delay divided by their mean (so
    ``mean`` still sets the size). Real delays have heavier tails than an
    exponential; :func:`~simulsi.aviation.fit_delay_model` fills it in.
    ``day_sigma`` makes whole days better or worse together (weather, ATC
    programmes): each simulated day draws a lognormal factor on every
    delay chance. ``airport_days`` does the same per airport from the
    empirical spread of good and bad days there (quantiles with mean 1;
    key ``"*"`` for any airport). ``cancel_rate`` is the chance a departure
    is cancelled for reasons outside the model (weather, crew, mechanical),
    scaled up on bad days; ``cancel_days`` replaces it with the empirical
    spread of each airport's daily cancelled share, drawn together with
    the delay factor.
    """

    prob: float = 0.2
    mean: float = 25.0
    block_cv: float = 0.06
    table: dict[str, tuple[float, float]] = field(default_factory=dict)
    block_bias: dict[str, float] = field(default_factory=dict)
    block_scale: float = 1.0
    shape: list[float] = field(default_factory=list)
    day_sigma: float = 0.0
    airport_days: dict[str, list[float]] = field(default_factory=dict)
    cancel_rate: float = 0.0
    cancel_days: dict[str, list[float]] = field(default_factory=dict)
    predictor: DelayPredictor | None = None

    def airport_factors(
        self, airports: Sequence[str], rs: RandomStream
    ) -> tuple[dict[str, float], dict[str, float]]:
        """How bad the day is at each airport, drawn from history: (delay factor, cancelled share).

        One draw per airport sets both, so a bad delay day is also a bad
        cancellation day.
        """
        factors: dict[str, float] = {}
        cancels: dict[str, float] = {}
        for a in sorted(airports):
            u = rs.random()
            q = self.airport_days.get(a) or self.airport_days.get("*")
            if q:
                factors[a] = _interp(q, u)
            c = self.cancel_days.get(a) or self.cancel_days.get("*")
            if c:
                cancels[a] = _interp(c, u)
        return factors, cancels

    def day_factor(self, rs: RandomStream) -> float:
        """How bad the whole day is: multiplies every flight's delay chance (mean 1)."""
        if self.day_sigma <= 0:
            return 1.0
        return math.exp(rs.normal(-0.5 * self.day_sigma**2, self.day_sigma))

    def _size(self, mean: float, rs: RandomStream) -> float:
        if not self.shape:
            return rs.exponential(mean)
        u = rs.random() * (len(self.shape) - 1)
        k = int(u)
        lo = self.shape[k]
        hi = self.shape[min(k + 1, len(self.shape) - 1)]
        return mean * (lo + (hi - lo) * (u - k))

    def primary(self, flight: Flight, rs: RandomStream, day: float = 1.0) -> float:
        if self.predictor is not None:
            expected = max(0.0, float(self.predictor(flight))) * day
            return self._size(expected, rs) if expected > 0 else 0.0
        hour = int(flight.std // 60) % 24
        prob, mean = self.table.get(
            f"{flight.origin}@{hour:02d}", self.table.get(flight.origin, (self.prob, self.mean))
        )
        if prob <= 0 or rs.random() >= min(1.0, prob * day):
            return 0.0
        return self._size(mean, rs)

    def block(self, flight: Flight, rs: RandomStream) -> float:
        planned = flight.block * self.block_bias.get(
            f"{flight.origin}-{flight.dest}", self.block_scale
        )
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
            "block_scale": self.block_scale,
            "shape": [round(x, 4) for x in self.shape],
            "day_sigma": self.day_sigma,
            "airport_days": {k: [round(x, 4) for x in v] for k, v in self.airport_days.items()},
            "cancel_rate": self.cancel_rate,
            "cancel_days": {k: [round(x, 5) for x in v] for k, v in self.cancel_days.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DelayModel:
        known = {
            "prob",
            "mean",
            "block_cv",
            "table",
            "block_bias",
            "block_scale",
            "shape",
            "day_sigma",
            "airport_days",
            "cancel_rate",
            "cancel_days",
        }
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
            block_scale=float(data.get("block_scale", 1.0)),
            shape=[float(x) for x in data.get("shape") or []],
            day_sigma=float(data.get("day_sigma", 0.0)),
            airport_days={
                str(k): [float(x) for x in v] for k, v in (data.get("airport_days") or {}).items()
            },
            cancel_rate=float(data.get("cancel_rate", 0.0)),
            cancel_days={
                str(k): [float(x) for x in v] for k, v in (data.get("cancel_days") or {}).items()
            },
        )


@dataclass(frozen=True)
class WeatherEvent:
    """Departure capacity at ``airport`` falls to ``capacity`` x normal from ``start`` to ``end``.

    ``probability`` is the chance the event happens on the day (each
    replication draws it); ``capacity=0`` closes the airport to departures.
    ``cancel`` is the share of departures in the window the airline cancels
    in advance (as for a forecast winter storm).
    """

    airport: str
    start: float
    end: float
    capacity: float = 0.4
    probability: float = 1.0
    name: str = ""
    cancel: float = 0.0

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ConfigError(f"weather at {self.airport}: end must be after start")
        if not 0 <= self.capacity <= 1:
            raise ConfigError("weather capacity must be in [0, 1]")
        if not 0 <= self.probability <= 1:
            raise ConfigError("weather probability must be in [0, 1]")
        if not 0 <= self.cancel <= 1:
            raise ConfigError("weather cancel share must be in [0, 1]")

    @classmethod
    def parse(cls, text: str) -> WeatherEvent:
        """``"HUB 15:00-18:00 0.4 p=0.6"`` (capacity and probability optional)."""
        parts = text.split()
        if len(parts) < 2 or "-" not in parts[1]:
            raise ConfigError(
                f"weather must look like 'HUB 15:00-18:00 [capacity] [p=prob]': {text!r}"
            )
        start, _, end = parts[1].partition("-")
        cap, prob, cancel = 0.4, 1.0, 0.0
        for extra in parts[2:]:
            if extra.startswith("p="):
                prob = float(extra[2:])
            elif extra.startswith("c="):
                cancel = float(extra[2:])
            else:
                cap = float(extra)
        return cls(
            parts[0].upper(),
            parse_time(start),
            parse_time(end),
            cap,
            prob,
            f"weather {parts[0].upper()}",
            cancel,
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
    late_turn_compression: float = 1.0
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
    load_factor: float = 0.85
    overnight_delay: float = 18 * 60.0
    spare_ferry_minutes: float | None = None
    delay_cost_per_minute: float = 100.0
    cancel_cost: float = 20000.0
    misconnect_cost_per_pax: float = 250.0
    eu261: bool = False
    delays: DelayModel = field(default_factory=DelayModel)
    otp_recalibration: tuple[float, float] = (0.0, 1.0)

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
        out["otp_recalibration"] = list(self.otp_recalibration)
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
            elif key == "otp_recalibration":
                a, b = value
                kwargs[key] = (float(a), float(b))
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


def _interp(q: Sequence[float], u: float) -> float:
    """The value at quantile ``u`` of equally spaced quantiles ``q``."""
    x = u * (len(q) - 1)
    k = int(x)
    return q[k] + (q[min(k + 1, len(q) - 1)] - q[k]) * (x - k)
