"""Flight schedules: flights, aircraft rotations, crew pairings and passenger connections.

Times are minutes from midnight of the operating day. CSV files may give
times as ``HH:MM`` (hours may exceed 23 for after-midnight arrivals), as
minutes, or as ISO date-times (``2026-10-08T06:30``), which are converted
relative to midnight of the earliest date in the file.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from simulsi.errors import ConfigError
from simulsi.randomness.stream import RandomStream

_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("flight", "flight_id", "id", "flight_no", "flight_number"),
    "tail": ("tail", "tail_number", "registration", "aircraft", "reg"),
    "origin": ("origin", "from", "dep", "departure_airport"),
    "dest": ("dest", "destination", "to", "arr", "arrival_airport"),
    "std": ("std", "sched_dep", "scheduled_departure", "departure", "dep_time"),
    "sta": ("sta", "sched_arr", "scheduled_arrival", "arrival", "arr_time"),
    "pax": ("pax", "passengers", "booked"),
    "seats": ("seats", "capacity"),
    "crew": ("crew", "crew_id", "pairing"),
    "aircraft_type": ("type", "aircraft_type", "equipment"),
}


def parse_time(value: Any, base: datetime | None = None) -> float:
    """Minutes from midnight: ``"06:30"`` -> 390, ``"25:10"`` -> 1510, ``95`` -> 95."""
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ConfigError("empty time")
    if "T" in text or (" " in text and "-" in text):
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ConfigError(f"cannot parse time {text!r}") from exc
        ref = base or moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return (moment - ref.replace(tzinfo=moment.tzinfo)).total_seconds() / 60
    if ":" in text:
        h, _, m = text.partition(":")
        try:
            return int(h) * 60 + float(m)
        except ValueError as exc:
            raise ConfigError(f"cannot parse time {text!r}") from exc
    try:
        return float(text)
    except ValueError as exc:
        raise ConfigError(f"cannot parse time {text!r}") from exc


def format_time(minutes: float) -> str:
    """``390`` -> ``"06:30"``; times past midnight keep counting (``"25:10"``)."""
    if not math.isfinite(minutes):
        return "-"
    total = round(minutes)
    sign = "-" if total < 0 else ""
    h, m = divmod(abs(total), 60)
    return f"{sign}{h:02d}:{m:02d}"


def _base_date(values: Iterable[Any]) -> datetime | None:
    dates = []
    for v in values:
        text = str(v).strip()
        if "T" in text or (" " in text and "-" in text):
            try:
                dates.append(datetime.fromisoformat(text))
            except ValueError:
                continue
    if not dates:
        return None
    first = min(d.replace(tzinfo=None) for d in dates)
    return first.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class Flight:
    """One scheduled flight leg."""

    id: str
    tail: str
    origin: str
    dest: str
    std: float
    sta: float
    pax: int = 150
    crew: str = ""
    aircraft_type: str = ""
    seats: int = 0

    @property
    def block(self) -> float:
        return self.sta - self.std

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["std"], row["sta"] = format_time(self.std), format_time(self.sta)
        return row


@dataclass(frozen=True)
class Connection:
    """``pax`` passengers connecting from flight ``inbound`` to flight ``outbound``."""

    inbound: str
    outbound: str
    pax: int


class Schedule:
    """A day of flights, with rotations (by tail), crew pairings and connections.

    >>> s = Schedule([
    ...     Flight("A1", "T1", "HUB", "AAA", 360, 450),
    ...     Flight("A2", "T1", "AAA", "HUB", 495, 585),
    ... ])
    >>> [f.id for f in s.rotations["T1"]]
    ['A1', 'A2']
    """

    def __init__(self, flights: Iterable[Flight], connections: Iterable[Connection] = ()) -> None:
        self.flights: tuple[Flight, ...] = tuple(sorted(flights, key=lambda f: (f.std, f.id)))
        self.connections: tuple[Connection, ...] = tuple(connections)
        self.by_id: dict[str, Flight] = {}
        for f in self.flights:
            if f.id in self.by_id:
                raise ConfigError(f"duplicate flight id {f.id!r}")
            self.by_id[f.id] = f
        self.rotations: dict[str, list[Flight]] = {}
        self.pairings: dict[str, list[Flight]] = {}
        for f in self.flights:
            self.rotations.setdefault(f.tail, []).append(f)
            if f.crew:
                self.pairings.setdefault(f.crew, []).append(f)
        self.inbound_connections: dict[str, list[Connection]] = {}
        self.outbound_connections: dict[str, list[Connection]] = {}
        for c in self.connections:
            self.inbound_connections.setdefault(c.inbound, []).append(c)
            self.outbound_connections.setdefault(c.outbound, []).append(c)

    def __len__(self) -> int:
        return len(self.flights)

    def __iter__(self) -> Any:
        return iter(self.flights)

    def __repr__(self) -> str:
        return (
            f"Schedule({len(self.flights)} flights, {len(self.rotations)} tails, "
            f"{len(self.airports)} airports, {len(self.connections)} connections)"
        )

    @property
    def airports(self) -> list[str]:
        return sorted({f.origin for f in self.flights} | {f.dest for f in self.flights})

    @property
    def tails(self) -> list[str]:
        return list(self.rotations)

    def validate(self) -> list[str]:
        """Problems that make the schedule unflyable (empty list when consistent)."""
        issues: list[str] = []
        for f in self.flights:
            if f.sta <= f.std:
                issues.append(f"{f.id}: arrival {format_time(f.sta)} not after departure")
            if f.origin == f.dest:
                issues.append(f"{f.id}: origin equals destination ({f.origin})")
        for tail, legs in self.rotations.items():
            for a, b in pairwise(legs):
                if a.dest != b.origin:
                    issues.append(
                        f"tail {tail}: {a.id} arrives at {a.dest} but {b.id} departs {b.origin}"
                    )
                if b.std < a.sta:
                    issues.append(f"tail {tail}: {b.id} departs before {a.id} arrives")
        for crew, legs in self.pairings.items():
            for a, b in pairwise(legs):
                if a.dest != b.origin:
                    issues.append(
                        f"crew {crew}: {a.id} ends at {a.dest} but {b.id} starts {b.origin}"
                    )
        for c in self.connections:
            if c.inbound not in self.by_id or c.outbound not in self.by_id:
                issues.append(f"connection {c.inbound}->{c.outbound}: unknown flight")
                continue
            i, o = self.by_id[c.inbound], self.by_id[c.outbound]
            if i.dest != o.origin:
                issues.append(f"connection {c.inbound}->{c.outbound}: {i.dest} != {o.origin}")
            elif o.std <= i.sta:
                issues.append(f"connection {c.inbound}->{c.outbound}: outbound leaves first")
        return issues

    def check(self) -> Schedule:
        issues = self.validate()
        if issues:
            more = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
            raise ConfigError("invalid schedule: " + "; ".join(issues[:5]) + more)
        return self

    def replace_flights(self, flights: Iterable[Flight]) -> Schedule:
        """A schedule with ``flights`` and the connections that still refer to them."""
        flights = list(flights)
        ids = {f.id for f in flights}
        kept = [c for c in self.connections if c.inbound in ids and c.outbound in ids]
        return Schedule(flights, kept)

    def retime(self, shifts: Mapping[str, float]) -> Schedule:
        """Move flights by ``{flight_id: minutes}`` (positive = later)."""
        unknown = set(shifts) - set(self.by_id)
        if unknown:
            raise ConfigError(f"unknown flights: {', '.join(sorted(unknown))}")
        return self.replace_flights(
            replace(f, std=f.std + shifts.get(f.id, 0.0), sta=f.sta + shifts.get(f.id, 0.0))
            for f in self.flights
        )

    # -- input / output ------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        rows: Iterable[Mapping[str, Any]],
        connections: Iterable[Mapping[str, Any]] | None = None,
        *,
        default_pax: int = 150,
    ) -> Schedule:
        rows = [{str(k).strip().lower(): v for k, v in r.items()} for r in rows]
        if not rows:
            raise ConfigError("schedule has no flights")
        columns = set().union(*(r.keys() for r in rows))
        col: dict[str, str] = {}
        for key, names in _ALIASES.items():
            found = next((n for n in names if n in columns), None)
            if found:
                col[key] = found
        missing = [k for k in ("id", "tail", "origin", "dest", "std", "sta") if k not in col]
        if missing:
            raise ConfigError(
                f"schedule is missing columns {missing}; expected e.g. "
                "flight, tail, origin, dest, std, sta (optional: pax, crew, type, seats)"
            )
        base = _base_date(r[col["std"]] for r in rows)
        flights = []
        for n, r in enumerate(rows, 1):
            try:
                flights.append(
                    Flight(
                        id=str(r[col["id"]]).strip(),
                        tail=str(r[col["tail"]]).strip(),
                        origin=str(r[col["origin"]]).strip().upper(),
                        dest=str(r[col["dest"]]).strip().upper(),
                        std=parse_time(r[col["std"]], base),
                        sta=parse_time(r[col["sta"]], base),
                        pax=_int(r.get(col.get("pax", "")), default_pax),
                        crew=str(r.get(col.get("crew", ""), "") or "").strip(),
                        aircraft_type=str(r.get(col.get("aircraft_type", ""), "") or "").strip(),
                        seats=_int(r.get(col.get("seats", "")), 0),
                    )
                )
            except ConfigError as exc:
                raise ConfigError(f"schedule row {n}: {exc}") from exc
        flights = [
            replace(f, sta=f.sta + 1440) if f.sta < f.std else f for f in flights
        ]  # overnight arrivals given as clock times
        conns = []
        for r in connections or []:
            low = {str(k).strip().lower(): v for k, v in r.items()}
            try:
                conns.append(
                    Connection(
                        str(low["inbound"]).strip(), str(low["outbound"]).strip(), int(low["pax"])
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ConfigError(f"connection rows need inbound, outbound, pax: {r}") from exc
        return cls(flights, conns)

    @classmethod
    def from_csv(
        cls, path: str | Path, connections: str | Path | None = None, *, default_pax: int = 150
    ) -> Schedule:
        """Load a schedule CSV (and optionally a connections CSV: inbound, outbound, pax)."""
        rows = _read_csv(path)
        conns = _read_csv(connections) if connections else None
        return cls.from_records(rows, conns, default_pax=default_pax)

    def to_csv(self, path: str | Path | None = None) -> str:
        buf = io.StringIO()
        fields = [
            "id",
            "tail",
            "origin",
            "dest",
            "std",
            "sta",
            "pax",
            "crew",
            "aircraft_type",
            "seats",
        ]
        w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for f in self.flights:
            w.writerow(f.to_row())
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text)
        return text

    def connections_csv(self, path: str | Path | None = None) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["inbound", "outbound", "pax"])
        for c in self.connections:
            w.writerow([c.inbound, c.outbound, c.pax])
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text)
        return text

    # -- synthetic -----------------------------------------------------------

    @classmethod
    def synthetic(
        cls,
        *,
        hub: str = "HUB",
        spokes: Sequence[str] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"),
        tails: int = 12,
        first_departure: float = 360.0,
        last_departure: float = 1260.0,
        turn: float = 45.0,
        banks: bool = True,
        bank_spacing: float = 120.0,
        crews_per_day: int = 2,
        connect_share: float = 0.3,
        seed: int = 0,
    ) -> Schedule:
        """A hub-and-spoke day: each tail flies hub-spoke round trips.

        With ``banks`` the hub departures cluster in waves ``bank_spacing``
        apart (good connections, peaky runway load); without, they are spread
        evenly. Crews fly half-day pairings; a share of passengers on each
        inbound connects to the next hub departures.
        """
        rs = RandomStream(seed)
        blocks = {s: 50.0 + 10.0 * ((k * 7) % 10) for k, s in enumerate(spokes)}
        flights: list[Flight] = []
        spread = bank_spacing if banks else (last_departure - first_departure) / 2
        for t in range(tails):
            tail = f"T{t + 1:02d}"
            offset = (t % 4) * 5.0 if banks else t * spread / tails
            now = first_departure + offset
            k = 0
            trips: list[tuple[float, str, float]] = []
            while True:
                spoke = spokes[(t + k) % len(spokes)]
                block = blocks[spoke]
                if banks:
                    wave = math.ceil((now - first_departure - offset - 1e-9) / bank_spacing)
                    now = max(now, first_departure + offset + max(0, wave) * bank_spacing)
                if now > last_departure:
                    break
                trips.append((now, spoke, block))
                now += 2 * (block + turn)
                k += 1
            per_crew = max(1, math.ceil(len(trips) / max(1, crews_per_day)))
            for j, (dep, spoke, block) in enumerate(trips):
                crew = f"C{t + 1:02d}{'abcdefgh'[min(7, j // per_crew)]}"
                out_id, back_id = f"F{len(flights) + 100}", f"F{len(flights) + 101}"
                flights.append(Flight(out_id, tail, hub, spoke, dep, dep + block, crew=crew))
                back = dep + block + turn
                flights.append(Flight(back_id, tail, spoke, hub, back, back + block, crew=crew))
        flights = [replace(f, pax=int(rs.integers(110, 181)), seats=180) for f in flights]
        flights = _swap_crews(flights, hub)
        conns: list[Connection] = []
        hub_out = sorted((f for f in flights if f.origin == hub), key=lambda f: f.std)
        for f in flights:
            if f.dest != hub:
                continue
            options = [o for o in hub_out if 45 <= o.std - f.sta <= 180 and o.tail != f.tail]
            if not options:
                continue
            total = int(f.pax * connect_share)
            picks = options[:3]
            for o in picks:
                n = total // len(picks)
                if n > 0:
                    conns.append(Connection(f.id, o.id, n))
        return cls(flights, conns)


def _swap_crews(flights: list[Flight], hub: str, min_connect: float = 30.0) -> list[Flight]:
    """Swap the last round trip of neighbouring aircraft's first crews where timing allows.

    The crews then change aircraft at the hub, so a late aircraft can delay
    another aircraft's departure through its crew.
    """
    by_crew: dict[str, list[Flight]] = {}
    for f in flights:
        by_crew.setdefault(f.crew, []).append(f)
    firsts = sorted(c for c in by_crew if c.endswith("a") and len(by_crew[c]) >= 4)
    relabel: dict[str, str] = {}
    for x, y in zip(firsts[::2], firsts[1::2], strict=False):
        lx, ly = by_crew[x], by_crew[y]
        if lx[-2].origin != hub or ly[-2].origin != hub:
            continue
        if ly[-2].std - lx[-3].sta >= min_connect and lx[-2].std - ly[-3].sta >= min_connect:
            relabel[ly[-2].id] = relabel[ly[-1].id] = x
            relabel[lx[-2].id] = relabel[lx[-1].id] = y
    return [replace(f, crew=relabel[f.id]) if f.id in relabel else f for f in flights]


def _int(value: Any, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(value))
    except ValueError as exc:
        raise ConfigError(f"not a number: {value!r}") from exc


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    try:
        with Path(path).open(newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
