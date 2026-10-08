"""The live operation: what has already happened, and the actions controllers can take.

:class:`OpsState` captures the day so far - actual departure and arrival
times, known delays (estimated departure times), cancellations and
aircraft out of service - so a forecast can start from *now* instead of
from the plan. Actions (:class:`Cancel`, :class:`Retime`, :class:`Swap`)
change the plan; :func:`apply_actions` returns the changed schedule so
alternatives can be simulated side by side.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from simulsi.aviation.schedule import Schedule, format_time, parse_time
from simulsi.errors import ConfigError


@dataclass
class FlightStatus:
    """What is known about one flight: actual or estimated times, or a cancellation."""

    atd: float | None = None
    ata: float | None = None
    etd: float | None = None
    cancelled: bool = False


@dataclass
class OpsState:
    """The operation at time ``now`` (minutes from midnight)."""

    now: float
    flights: dict[str, FlightStatus] = field(default_factory=dict)
    aircraft_unavailable: dict[str, float] = field(default_factory=dict)

    def check(self, schedule: Schedule) -> OpsState:
        unknown = set(self.flights) - set(schedule.by_id)
        if unknown:
            raise ConfigError(f"status for unknown flights: {', '.join(sorted(unknown)[:5])}")
        tails = set(self.aircraft_unavailable) - set(schedule.rotations)
        if tails:
            raise ConfigError(f"unknown tails: {', '.join(sorted(tails))}")
        for fid, s in self.flights.items():
            if s.ata is not None and s.atd is None:
                raise ConfigError(f"{fid}: arrival time given without departure time")
            for name in ("atd", "ata"):
                t = getattr(s, name)
                if t is not None and t > self.now + 1e-9:
                    raise ConfigError(
                        f"{fid}: {name} {format_time(t)} is after now {format_time(self.now)}"
                    )
        return self

    @classmethod
    def from_records(
        cls,
        now: float | str,
        rows: Iterable[Mapping[str, Any]] = (),
        aircraft_unavailable: Mapping[str, float | str] | None = None,
    ) -> OpsState:
        """Rows with ``flight`` and any of ``atd``, ``ata``, ``etd``, ``status`` (``cancelled``)."""
        flights: dict[str, FlightStatus] = {}
        for r in rows:
            low = {str(k).strip().lower(): v for k, v in r.items()}
            fid = str(low.get("flight") or low.get("id") or "").strip()
            if not fid:
                raise ConfigError(f"status row without a flight: {r}")

            def t(key: str, row: dict[str, Any] = low) -> float | None:
                v = row.get(key)
                return None if v is None or str(v).strip() == "" else parse_time(v)

            status = str(low.get("status") or "").strip().lower()
            flights[fid] = FlightStatus(
                atd=t("atd"),
                ata=t("ata"),
                etd=t("etd"),
                cancelled=status in {"cancelled", "canceled", "cnl"},
            )
        unavailable = {k: parse_time(v) for k, v in (aircraft_unavailable or {}).items()}
        return cls(parse_time(now), flights, unavailable)

    @classmethod
    def from_csv(cls, path: str | Path, now: float | str) -> OpsState:
        try:
            with Path(path).open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        except OSError as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc
        aog = {}
        kept = []
        for r in rows:
            low = {str(k).strip().lower(): v for k, v in r.items()}
            if str(low.get("status", "")).strip().lower() == "aog":
                aog[str(low.get("tail") or "").strip()] = low.get("etd") or "48:00"
            else:
                kept.append(r)
        return cls.from_records(now, kept, aog)

    def to_csv(self, path: str | Path | None = None) -> str:
        """The flight statuses as CSV (flight, atd, ata, etd, status), readable by :meth:`from_csv`."""

        def t(v: float | None) -> str:
            return "" if v is None else format_time(v)

        lines = ["flight,atd,ata,etd,status,tail"]
        for fid, s in self.flights.items():
            lines.append(
                f"{fid},{t(s.atd)},{t(s.ata)},{t(s.etd)},{'cancelled' if s.cancelled else ''},"
            )
        for tail, until in self.aircraft_unavailable.items():
            lines.append(f",,,{t(until)},aog,{tail}")
        text = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(text)
        return text


@dataclass(frozen=True)
class Cancel:
    """Cancel flights (cancel a round trip to keep the aircraft where it is)."""

    flights: tuple[str, ...]

    def __init__(self, *flights: str) -> None:
        object.__setattr__(self, "flights", tuple(flights))

    def describe(self) -> str:
        return "cancel " + ", ".join(self.flights)


@dataclass(frozen=True)
class Retime:
    """Delay a flight by ``minutes`` on purpose (later flights of the tail move only if they must)."""

    flight: str
    minutes: float

    def describe(self) -> str:
        return f"retime {self.flight} +{self.minutes:g} min"


@dataclass(frozen=True)
class Swap:
    """Swap the remaining rotations of two aircraft from ``after`` (minutes) onwards.

    Both aircraft must be at the same airport at the swap point.
    """

    tail_a: str
    tail_b: str
    after: float

    def describe(self) -> str:
        return f"swap {self.tail_a}<->{self.tail_b} after {format_time(self.after)}"


Action = Cancel | Retime | Swap


def apply_actions(schedule: Schedule, actions: Iterable[Action]) -> tuple[Schedule, set[str]]:
    """The schedule after ``actions``, and the set of cancelled flight ids."""
    flights = {f.id: f for f in schedule.flights}
    cancelled: set[str] = set()
    for action in actions:
        if isinstance(action, Cancel):
            for fid in action.flights:
                if fid not in flights:
                    raise ConfigError(f"cancel: unknown flight {fid!r}")
                cancelled.add(fid)
        elif isinstance(action, Retime):
            if action.flight not in flights:
                raise ConfigError(f"retime: unknown flight {action.flight!r}")
            f = flights[action.flight]
            flights[f.id] = replace(f, std=f.std + action.minutes, sta=f.sta + action.minutes)
        elif isinstance(action, Swap):
            for tail in (action.tail_a, action.tail_b):
                if tail not in schedule.rotations:
                    raise ConfigError(f"swap: unknown tail {tail!r}")
            a_next = _next_leg(flights, action.tail_a, action.after)
            b_next = _next_leg(flights, action.tail_b, action.after)
            if a_next and b_next and a_next.origin != b_next.origin:
                raise ConfigError(
                    f"swap: {action.tail_a} is at {a_next.origin} but {action.tail_b} is at {b_next.origin}"
                )
            for fid, f in list(flights.items()):
                if f.std >= action.after:
                    if f.tail == action.tail_a:
                        flights[fid] = replace(f, tail=action.tail_b)
                    elif f.tail == action.tail_b:
                        flights[fid] = replace(f, tail=action.tail_a)
        else:
            raise ConfigError(f"unknown action {action!r}")
    return schedule.replace_flights(flights.values()), cancelled


def _next_leg(flights: Mapping[str, Any], tail: str, after: float) -> Any:
    legs = sorted(
        (f for f in flights.values() if f.tail == tail and f.std >= after), key=lambda f: f.std
    )
    return legs[0] if legs else None


def parse_action(text: str) -> Action:
    """``"cancel F101 F102"``, ``"retime F101 30"``, ``"swap T01 T02 12:00"``."""
    parts = text.replace(",", " ").split()
    if not parts:
        raise ConfigError("empty action")
    verb, args = parts[0].lower(), parts[1:]
    if verb == "cancel" and args:
        return Cancel(*args)
    if verb in {"retime", "delay"} and len(args) == 2:
        return Retime(args[0], float(args[1]))
    if verb == "swap" and len(args) == 3:
        return Swap(args[0], args[1], parse_time(args[2]))
    raise ConfigError(
        f"cannot parse action {text!r}; use 'cancel F1 F2', 'retime F1 30' or 'swap T1 T2 12:00'"
    )
