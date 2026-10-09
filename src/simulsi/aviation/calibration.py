"""Calibrate on history and check forecasts against what really happened.

:class:`History` holds actual flights (scheduled and actual times, tails,
cancellations) for many days - from a generic CSV or the US BTS on-time
dataset. :func:`fit_delay_model` separates *primary* delay (what the
departure added on its own) from knock-on delay of a late inbound
aircraft, and estimates how often and how much primary delay happens per
airport and hour. :func:`backtest` forecasts each historical day from its
schedule and scores the forecasts: whether flights given a 70 % on-time
chance really are on time about 70 % of the time (reliability), the Brier
score, delay-quantile coverage and the error of the predicted OTP.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from simulsi.aviation.config import DelayModel, OpsConfig
from simulsi.aviation.engine import simulate_day
from simulsi.aviation.forecast import forecast
from simulsi.aviation.schedule import Flight, Schedule, format_time, parse_time
from simulsi.aviation.state import FlightStatus, OpsState
from simulsi.errors import ConfigError
from simulsi.randomness.stream import derive_seed

_HIST_ALIASES = {
    "date": ("date", "fl_date", "flightdate", "day"),
    "id": ("flight", "flight_id", "id"),
    "tail": ("tail", "tail_num", "tail_number", "registration"),
    "origin": ("origin",),
    "dest": ("dest", "destination"),
    "std": ("std", "crs_dep_time", "crsdeptime", "sched_dep"),
    "sta": ("sta", "crs_arr_time", "crsarrtime", "sched_arr"),
    "atd": ("atd", "dep_time", "deptime", "actual_dep"),
    "ata": ("ata", "arr_time", "arrtime", "actual_arr"),
    "dep_delay": ("dep_delay", "depdelay"),
    "arr_delay": ("arr_delay", "arrdelay"),
    "cancelled": ("cancelled", "canceled"),
    "pax": ("pax", "passengers"),
    "elapsed": ("crs_elapsed_time", "crselapsedtime", "sched_block"),
    "carrier": ("op_unique_carrier", "op_carrier", "reporting_airline", "carrier", "airline"),
    "number": ("op_carrier_fl_num", "flight_number_reporting_airline", "flight_number"),
}
_TRUE = {"1", "1.0", "1.00", "true", "yes", "y"}


@dataclass
class ActualFlight:
    date: str
    flight: Flight
    atd: float
    ata: float
    cancelled: bool

    @property
    def dep_delay(self) -> float:
        return self.atd - self.flight.std

    @property
    def arr_delay(self) -> float:
        return self.ata - self.flight.sta


@dataclass
class History:
    """Actual flights over several days.

    When the file has scheduled block times (BTS ``CRSElapsedTime``), local
    clock times are converted to one clock - the easternmost airport's,
    or ``clock`` - using UTC offsets inferred from the data itself
    (``tz_offsets``, minutes relative to that clock), so rotations through
    several time zones line up.
    """

    flights: list[ActualFlight] = field(default_factory=list)
    tz_offsets: dict[str, float] = field(default_factory=dict)
    clock: str = ""

    @property
    def dates(self) -> list[str]:
        return sorted({a.date for a in self.flights})

    def day(self, date: str) -> list[ActualFlight]:
        return [a for a in self.flights if a.date == date]

    def schedule(self, date: str, *, repair: bool = True) -> Schedule:
        """The day's schedule. With ``repair``, a tail whose recorded legs do not chain (a
        diverted or missing flight, overlapping times) is split into separate rotations
        (``N123#2``) at each break, so the schedule always validates."""
        flights = [a.flight for a in self.day(date)]
        if not repair:
            return Schedule(flights)
        by_tail: dict[str, list[Flight]] = {}
        for f in sorted(flights, key=lambda f: (f.std, f.id)):
            by_tail.setdefault(f.tail, []).append(f)
        out: list[Flight] = []
        for tail, legs in by_tail.items():
            part, prev = 1, None
            for f in legs:
                if prev is not None and (prev.dest != f.origin or f.std < prev.sta):
                    part += 1
                out.append(f if part == 1 else replace(f, tail=f"{tail}#{part}"))
                prev = f
        return Schedule(out)

    def state(self, date: str, now: float) -> OpsState:
        """What was known at ``now`` on ``date``: departures, arrivals and cancellations so far."""
        flights: dict[str, FlightStatus] = {}
        for a in self.day(date):
            f = a.flight
            if a.cancelled:
                if f.std - 60 <= now:
                    flights[f.id] = FlightStatus(cancelled=True)
            elif a.atd <= now:
                flights[f.id] = FlightStatus(atd=a.atd, ata=a.ata if a.ata <= now else None)
        return OpsState(now, flights)

    def summary(self) -> dict[str, float]:
        flown = [a for a in self.flights if not a.cancelled]
        return {
            "days": float(len(self.dates)),
            "flights": float(len(self.flights)),
            "cancelled": float(len(self.flights) - len(flown)),
            "otp": float(np.mean([a.arr_delay <= 15 for a in flown])) if flown else math.nan,
            "dep_delay.mean": float(np.mean([max(0.0, a.dep_delay) for a in flown]))
            if flown
            else math.nan,
        }

    @classmethod
    def from_records(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        carrier: str | None = None,
        clock: str | None = None,
    ) -> History:
        """Rows with date, flight, tail, origin, dest, std, sta, atd/ata (or dep_delay/arr_delay), cancelled.

        Times may be ``HH:MM``, minutes or ``HHMM`` (BTS style, e.g. ``0730``).
        ``carrier`` keeps one airline's flights (BTS files hold all of them).
        """
        rows = [{str(k).strip().lower(): v for k, v in r.items()} for r in rows]
        if not rows:
            raise ConfigError("history is empty")
        cols = set().union(*(r.keys() for r in rows))
        col = {k: next((n for n in names if n in cols), "") for k, names in _HIST_ALIASES.items()}
        need = ["date", "origin", "dest", "std", "sta"]
        missing = [k for k in need if not col[k]] + ([] if col["id"] or col["number"] else ["id"])
        if missing:
            raise ConfigError(f"history is missing columns {missing}")
        if carrier and col["carrier"]:
            rows = [r for r in rows if str(r[col["carrier"]]).strip().upper() == carrier.upper()]
            if not rows:
                raise ConfigError(f"no flights for carrier {carrier!r}")
        parsed = []
        for n, r in enumerate(rows, 1):
            try:
                parsed.append(_parse_row(r, col))
            except (KeyError, ValueError, ConfigError) as exc:
                raise ConfigError(f"history row {n}: {exc}") from exc
        offsets: dict[str, float] = {}
        if col["elapsed"]:
            offsets = _tz_offsets(parsed, clock)
        out = []
        counter: dict[tuple[str, str], int] = {}
        for p in parsed:
            if p is None:
                continue
            date, fid, tail, o, d, std, sta, elapsed, dep_delay, arr_delay, cancelled, pax = p
            counter[(date, fid)] = counter.get((date, fid), 0) + 1
            if counter[(date, fid)] > 1:
                fid = f"{fid}.{counter[(date, fid)]}"
            if offsets and elapsed is not None and o in offsets:
                std = std - offsets[o]
                sta = std + elapsed
            elif sta < std:
                sta += 1440
            if not cancelled and (dep_delay is None or arr_delay is None):
                continue  # diverted or incomplete
            f = Flight(fid, tail or f"?{fid}", o, d, std, sta, pax)
            atd = math.nan if cancelled else std + float(dep_delay or 0.0)
            ata = math.nan if cancelled else sta + float(arr_delay or 0.0)
            out.append(ActualFlight(date, f, atd, ata, cancelled))
        ref = ""
        if offsets:
            ref = clock.upper() if clock else max(offsets, key=lambda a: (offsets[a] == 0, a))
        return cls(out, offsets, ref)

    @classmethod
    def from_csv(
        cls, path: str | Path, *, carrier: str | None = None, clock: str | None = None
    ) -> History:
        """A history CSV, including US BTS on-time files (both the download-form names such as
        ``FL_DATE, TAIL_NUM, CRS_DEP_TIME`` and the monthly PREZIP names such as
        ``FlightDate, Tail_Number, CRSDepTime``). ``carrier`` filters while reading."""
        try:
            with Path(path).open(newline="") as fh:
                reader = csv.DictReader(fh)
                if carrier:
                    names = {str(n).strip().lower(): n for n in reader.fieldnames or []}
                    key = next((names[n] for n in _HIST_ALIASES["carrier"] if n in names), None)
                    rows: Iterable[Mapping[str, Any]] = (
                        r
                        for r in reader
                        if key is None or str(r[key]).strip().upper() == carrier.upper()
                    )
                else:
                    rows = reader
                return cls.from_records(list(rows), carrier=carrier, clock=clock)
        except OSError as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc

    def to_csv(self, path: str | Path) -> None:
        from simulsi.aviation.schedule import format_time

        with Path(path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "date",
                    "flight",
                    "tail",
                    "origin",
                    "dest",
                    "std",
                    "sta",
                    "atd",
                    "ata",
                    "cancelled",
                    "pax",
                ]
            )
            for a in self.flights:
                f = a.flight
                w.writerow(
                    [
                        a.date,
                        f.id,
                        f.tail,
                        f.origin,
                        f.dest,
                        format_time(f.std),
                        format_time(f.sta),
                        "" if a.cancelled else format_time(a.atd),
                        "" if a.cancelled else format_time(a.ata),
                        int(a.cancelled),
                        f.pax,
                    ]
                )

    @classmethod
    def simulated(
        cls,
        schedule: Schedule,
        config: OpsConfig | None = None,
        *,
        days: int = 30,
        seed: int = 0,
        start: str = "2026-01-01",
    ) -> History:
        """A synthetic history: ``days`` simulated days of ``schedule`` (for testing calibration)."""
        from datetime import date, timedelta

        cfg = config or OpsConfig()
        d0 = date.fromisoformat(start)
        out = []
        for k in range(days):
            day = simulate_day(schedule, cfg, seed=derive_seed(seed, "history", k))
            label = (d0 + timedelta(days=k)).isoformat()
            for f in schedule.flights:
                o = day.flights[f.id]
                out.append(ActualFlight(label, f, o.dep, o.arr, o.cancelled))
        return cls(out)


def _hhmm(value: Any) -> float:
    text = str(value).strip()
    if text.replace(".", "", 1).isdigit() and ":" not in text:
        v = float(text)
        if len(text.split(".")[0]) in (3, 4) and v < 2400 and v % 100 < 60:
            return (v // 100) * 60 + v % 100
        return v
    return parse_time(text)


def _parse_row(r: Mapping[str, Any], col: Mapping[str, str]) -> tuple[Any, ...] | None:
    def text(key: str) -> str:
        return str(r.get(col[key], "") or "").strip() if col[key] else ""

    date = text("date")[:10]
    fid = text("id") or f"{text('carrier')}{text('number')}"
    std, sta = _hhmm(r[col["std"]]), _hhmm(r[col["sta"]])
    cancelled = text("cancelled").lower() in _TRUE
    elapsed = float(text("elapsed")) if text("elapsed") else None
    dep_delay = arr_delay = None
    if not cancelled:
        if text("dep_delay"):
            dep_delay = float(text("dep_delay"))
        elif text("atd"):
            dep_delay = _actual(_hhmm(text("atd")), std) - std
        if text("arr_delay"):
            arr_delay = float(text("arr_delay"))
        elif text("ata"):
            arr_delay = _actual(_hhmm(text("ata")), sta if sta >= std else sta + 1440) - (
                sta if sta >= std else sta + 1440
            )
    pax = int(float(text("pax"))) if text("pax") else 150
    o, d = text("origin").upper(), text("dest").upper()
    return (date, fid, text("tail"), o, d, std, sta, elapsed, dep_delay, arr_delay, cancelled, pax)


def _tz_offsets(parsed: Sequence[tuple[Any, ...] | None], clock: str | None) -> dict[str, float]:
    """UTC offsets (minutes, relative to the reference airport) implied by local times and block times."""
    votes: dict[tuple[str, str], Counter[float]] = {}
    for p in parsed:
        if p is None or p[7] is None:
            continue
        _, _, _, o, d, std, sta, elapsed = p[:8]
        diff = ((sta - std - elapsed) + 720) % 1440 - 720
        votes.setdefault((o, d), Counter())[round(diff / 30) * 30.0] += 1
    graph: dict[str, list[tuple[str, float]]] = {}
    for (o, d), c in votes.items():
        diff = c.most_common(1)[0][0]
        graph.setdefault(o, []).append((d, diff))
        graph.setdefault(d, []).append((o, -diff))
    if not graph:
        return {}
    offsets: dict[str, float] = {}
    for start in sorted(graph, key=lambda a: -len(graph[a])):
        if start in offsets:
            continue
        offsets[start] = 0.0
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b, diff in graph[a]:
                if b not in offsets:
                    offsets[b] = offsets[a] + diff
                    queue.append(b)
    ref = clock.upper() if clock else max(offsets, key=lambda a: (offsets[a], a))
    if ref not in offsets:
        raise ConfigError(f"clock airport {ref!r} is not in the history")
    base = offsets[ref]
    return {a: v - base for a, v in offsets.items()}


def _actual(t: float, sched: float) -> float:
    """Actual clock time nearest to the scheduled time (handles crossing midnight)."""
    return min((t + k * 1440 for k in (-1, 0, 1)), key=lambda x: abs(x - sched))


def fit_delay_model(
    history: History,
    *,
    min_turn: float = 35.0,
    by: str = "origin",
    min_samples: int = 30,
    threshold: float = 1.0,
    empirical: bool = True,
) -> DelayModel:
    """Estimate primary-delay rates and block-time variability from history.

    A departure's knock-on delay is how late its aircraft was ready (the
    actual arrival of the tail's previous flight that day plus
    ``min_turn``); the rest of its departure delay is primary. Primary
    delays above ``threshold`` minutes give the probability, and their
    mean the exponential mean, per origin (``by="origin"``) or per origin
    and hour (``by="origin_hour"``) when there are at least ``min_samples``
    departures, else network-wide. Block-time variability and per-route
    bias come from actual against scheduled block times.
    """
    if by not in {"origin", "origin_hour"}:
        raise ConfigError("by must be 'origin' or 'origin_hour'")
    prim: list[tuple[Flight, float]] = []
    ratios: dict[str, list[float]] = {}
    by_day_tail: dict[tuple[str, str], list[ActualFlight]] = {}
    for a in history.flights:
        by_day_tail.setdefault((a.date, a.flight.tail), []).append(a)
    for legs in by_day_tail.values():
        legs.sort(key=lambda a: a.flight.std)
        ready = -math.inf
        for a in legs:
            if a.cancelled:
                continue
            f = a.flight
            knock_on = max(0.0, ready - f.std) if not f.tail.startswith("?") else 0.0
            prim.append((f, max(0.0, a.dep_delay - knock_on)))
            if f.block > 0 and a.ata > a.atd:
                ratios.setdefault(f"{f.origin}-{f.dest}", []).append((a.ata - a.atd) / f.block)
            ready = a.ata + min_turn
    if not prim:
        raise ConfigError("no flown flights in history")

    def rate(values: Sequence[float]) -> tuple[float, float]:
        hits = [v for v in values if v > threshold]
        p = len(hits) / len(values)
        return p, float(np.mean(hits)) if hits else 1.0

    prob, mean = rate([v for _, v in prim])
    table: dict[str, tuple[float, float]] = {}
    groups: dict[str, list[float]] = {}
    for f, v in prim:
        key = f.origin if by == "origin" else f"{f.origin}@{int(f.std // 60) % 24:02d}"
        groups.setdefault(key, []).append(v)
    for key, values in groups.items():
        if len(values) >= min_samples:
            p, m = rate(values)
            table[key] = (round(p, 4), round(m, 2))
    all_ratios = [r for rs in ratios.values() for r in rs]
    scale = float(np.mean(all_ratios)) if all_ratios else 1.0
    within = [r / float(np.mean(rs)) for rs in ratios.values() if len(rs) >= 3 for r in rs]
    cv = float(np.std(within)) if len(within) > 1 else 0.06
    bias = {k: round(float(np.mean(v)), 4) for k, v in ratios.items() if len(v) >= min_samples}
    hits = np.array([v for _, v in prim if v > threshold])
    shape: list[float] = []
    if empirical and len(hits) >= 200:
        q = np.quantile(hits, np.linspace(0, 1, 41))
        sampled_mean = (q.sum() - (q[0] + q[-1]) / 2) / (
            len(q) - 1
        )  # mean of the interpolated draw
        shape = [float(x) for x in q / sampled_mean]
    return DelayModel(
        prob=round(prob, 4),
        mean=round(mean, 2),
        block_cv=round(cv, 4),
        table=table,
        block_bias=bias,
        block_scale=round(scale, 4),
        shape=shape,
    )


def fit_turn_times(
    history: History, *, quantile: float = 0.1, min_samples: int = 20, floor: float = 15.0
) -> dict[str, float]:
    """Minimum turn time per airport: a low quantile of the actual ground times between
    consecutive legs of the same aircraft (the fastest turns the operation achieves)."""
    ground: dict[str, list[float]] = {}
    by_day_tail: dict[tuple[str, str], list[ActualFlight]] = {}
    for a in history.flights:
        if not a.cancelled and not a.flight.tail.startswith("?"):
            by_day_tail.setdefault((a.date, a.flight.tail), []).append(a)
    for legs in by_day_tail.values():
        legs.sort(key=lambda a: a.flight.std)
        for prev, nxt in pairwise(legs):
            if prev.flight.dest == nxt.flight.origin and nxt.atd > prev.ata:
                ground.setdefault(nxt.flight.origin, []).append(nxt.atd - prev.ata)
    return {
        a: round(max(floor, float(np.quantile(v, quantile))), 1)
        for a, v in sorted(ground.items())
        if len(v) >= min_samples
    }


def calibrate(
    history: History,
    config: OpsConfig | None = None,
    *,
    iterations: int = 6,
    replications: int = 10,
    max_days: int = 7,
    seed: int = 0,
    late: float = 15.0,
    day_effect: bool = True,
) -> DelayModel:
    """Fit primary delays so that *simulated* departures look like history.

    :func:`fit_delay_model` gives a first guess, but it counts everything
    that is not knock-on delay as primary - including runway queues and
    crew waits that the simulation adds by itself. This function
    re-simulates up to ``max_days`` historical schedules with ``config``
    and nudges each airport's primary-delay probability and mean until
    the simulated share of departures more than ``late`` minutes late and
    the mean departure delay match the history (a simple
    method-of-simulated-moments loop). With ``day_effect`` it then sets
    ``day_sigma`` so simulated days vary as much as real ones do.
    """
    cfg = config or OpsConfig()
    model = fit_delay_model(history, min_turn=cfg.min_turn, threshold=5.0)
    obs: dict[str, list[float]] = {}
    for a in history.flights:
        if not a.cancelled:
            obs.setdefault(a.flight.origin, []).append(a.dep_delay)
    target = {k: _departure_stats(v, late) for k, v in obs.items()}
    everything = [d for v in obs.values() for d in v]
    target_all = _departure_stats(everything, late)
    days = [d for d in history.dates if not history.schedule(d).validate()][:max_days]
    if not days:
        raise ConfigError("no valid historical schedules to calibrate on")
    schedules = [history.schedule(d) for d in days]
    for _ in range(iterations):
        trial = cfg.replace(delays=model)
        sim: dict[str, list[float]] = {}
        for k, sched in enumerate(schedules):
            for r in range(replications):
                day = simulate_day(sched, trial, seed=derive_seed(seed, f"calibrate{k}", r))
                for f in sched.flights:
                    o = day.flights[f.id]
                    if not o.cancelled:
                        sim.setdefault(f.origin, []).append(o.dep_delay)
        sim_all = _departure_stats([d for v in sim.values() for d in v], late)
        model.prob, model.mean = _nudge((model.prob, model.mean), target_all, sim_all)
        for key in list(model.table):
            origin = key.split("@")[0]
            if origin in sim and origin in target:
                model.table[key] = _nudge(
                    model.table[key], target[origin], _departure_stats(sim[origin], late)
                )
    model.prob, model.mean = round(model.prob, 4), round(model.mean, 2)
    model.table = {k: (round(p, 4), round(m, 2)) for k, (p, m) in model.table.items()}
    if day_effect:
        model.day_sigma = _fit_day_sigma(history, cfg, model, schedules, replications, seed, late)
    return model


def _late_share_spread(
    cfg: OpsConfig,
    model: DelayModel,
    schedules: Sequence[Schedule],
    reps: int,
    seed: int,
    late: float,
) -> float:
    """Standard deviation of log(daily share of late departures) across simulated days."""
    trial = cfg.replace(delays=model)
    logs = []
    for k, sched in enumerate(schedules):
        for r in range(reps):
            day = simulate_day(sched, trial, seed=derive_seed(seed, f"spread{k}", r))
            d = [o.dep_delay for o in day.flights.values() if not o.cancelled]
            logs.append(math.log(max(0.01, float(np.mean([x > late for x in d])))))
    return float(np.std(logs))


def _fit_day_sigma(
    history: History,
    cfg: OpsConfig,
    model: DelayModel,
    schedules: Sequence[Schedule],
    reps: int,
    seed: int,
    late: float,
) -> float:
    """Day-to-day variation the flight-level randomness does not explain, as a lognormal sigma."""
    shares = []
    for date in history.dates:
        d = [a.dep_delay for a in history.day(date) if not a.cancelled]
        if d:
            shares.append(math.log(max(0.01, float(np.mean([x > late for x in d])))))
    if len(shares) < 5:
        return 0.0
    target = float(np.std(shares))
    base = _late_share_spread(cfg, replace(model, day_sigma=0.0), schedules, reps, seed, late)
    if target <= base:
        return 0.0
    sigma = math.sqrt(target**2 - base**2)
    for _ in range(2):  # the share of late flights reacts less than one-for-one to the factor
        got = _late_share_spread(cfg, replace(model, day_sigma=sigma), schedules, reps, seed, late)
        extra = math.sqrt(max(1e-9, got**2 - base**2))
        sigma = min(1.5, sigma * math.sqrt(target**2 - base**2) / extra)
    return round(sigma, 3)


def _departure_stats(delays: Sequence[float], late: float) -> tuple[float, float]:
    return float(np.mean([d > late for d in delays])), float(np.mean([max(0.0, d) for d in delays]))


def _nudge(
    current: tuple[float, float], target: tuple[float, float], simulated: tuple[float, float]
) -> tuple[float, float]:
    prob, mean = current
    prob = max(prob, 0.005)
    (t_late, t_mean), (s_late, s_mean) = target, simulated
    new_prob = min(1.0, max(0.005, prob * t_late / s_late)) if s_late > 0 else prob
    # mean delay scales with prob x mean: what the probability change does not explain goes to the mean
    if s_mean > 0:
        mean = min(600.0, max(1.0, mean * (t_mean / s_mean) / (new_prob / prob)))
    return new_prob, mean


def reliability_table(
    probabilities: Sequence[float], outcomes: Sequence[bool], bins: int = 10
) -> list[dict[str, Any]]:
    """Calibration of probability forecasts: predicted against observed frequency per bin."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (p >= lo) & ((p < hi) if k < bins - 1 else (p <= hi))
        if mask.any():
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "predicted": round(float(p[mask].mean()), 3),
                    "observed": round(float(y[mask].mean()), 3),
                    "flights": int(mask.sum()),
                }
            )
    return rows


@dataclass
class BacktestResult:
    days: list[dict[str, Any]]
    reliability: list[dict[str, Any]]
    brier: float
    brier_climatology: float
    coverage: dict[str, float]
    otp_mae: float
    live: list[dict[str, Any]] = field(default_factory=list)

    @property
    def skill(self) -> float:
        """Brier skill score against always predicting the overall on-time rate (> 0 is better)."""
        return 1 - self.brier / self.brier_climatology if self.brier_climatology > 0 else math.nan

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        cov = ", ".join(f"{k}: {v:.0%}" for k, v in self.coverage.items())
        return "\n".join(
            [
                format_table(self.days),
                "",
                f"OTP error (mean absolute): {self.otp_mae:.1%}",
                f"Brier score {self.brier:.4f} (climatology {self.brier_climatology:.4f}, skill {self.skill:+.2f})",
                f"departure delays at or below the predicted quantile ({cov})",
                "",
                "reliability of on-time probabilities:",
                format_table(self.reliability),
            ]
            + (
                [
                    "",
                    "live re-forecasts (flights not yet departed at that time):",
                    format_table(self.live),
                ]
                if self.live
                else []
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "reliability": self.reliability,
            "brier": self.brier,
            "brier_climatology": self.brier_climatology,
            "skill": self.skill,
            "coverage": self.coverage,
            "otp_mae": self.otp_mae,
            "live": self.live,
        }


def backtest(
    history: History,
    config: OpsConfig | None = None,
    *,
    replications: int = 100,
    seed: int = 0,
    dates: Sequence[str] | None = None,
    live_at: Sequence[float] = (),
) -> BacktestResult:
    """Forecast each historical day and score the forecasts against what happened.

    Day-ahead forecasts use the schedule alone. For each time in
    ``live_at`` (minutes on the history's clock) the day is also
    re-forecast from what was known at that time, and the flights still to
    depart are scored - showing how much the live state adds.
    """
    cfg = config or OpsConfig()
    probs: list[float] = []
    outcomes: list[bool] = []
    hits = {"p50": 0, "p80": 0, "p95": 0}
    n_dep = 0
    rows = []
    errors = []
    live: dict[float, dict[str, list[float]]] = {
        t: {"p": [], "y": [], "err": [], "day_p": [], "day_err": []} for t in live_at
    }
    for date in dates or history.dates:
        schedule = history.schedule(date)
        if not schedule.flights:
            continue
        actual = {a.flight.id: a for a in history.day(date)}
        fc = forecast(schedule, cfg, replications=replications, seed=seed)
        flown = [a for a in actual.values() if not a.cancelled]
        act_otp = float(np.mean([a.arr_delay <= cfg.on_time for a in flown])) if flown else math.nan
        pred_otp = fc.summary()["otp"]["mean"]
        errors.append(abs(pred_otp - act_otp))
        for a in flown:
            p = fc.flights[a.flight.id]
            probs.append(p.p_on_time)
            outcomes.append(a.arr_delay <= cfg.on_time)
            if math.isfinite(p.dep_delay_p95):
                n_dep += 1
                hits["p50"] += a.dep_delay <= p.dep_delay_p50
                hits["p80"] += a.dep_delay <= p.dep_delay_p80
                hits["p95"] += a.dep_delay <= p.dep_delay_p95
        for t, acc in live.items():
            lfc = forecast(
                schedule, cfg, replications=replications, seed=seed, state=history.state(date, t)
            )
            for a in flown:
                if a.flight.std - 60 <= t:
                    continue  # already decided or departed
                lp, dp = lfc.flights[a.flight.id], fc.flights[a.flight.id]
                on_time = float(a.arr_delay <= cfg.on_time)
                acc["p"].append(lp.p_on_time)
                acc["y"].append(on_time)
                acc["day_p"].append(dp.p_on_time)
                if math.isfinite(lp.arr_delay_mean) and math.isfinite(dp.arr_delay_mean):
                    acc["err"].append(abs(max(0.0, a.arr_delay) - lp.arr_delay_mean))
                    acc["day_err"].append(abs(max(0.0, a.arr_delay) - dp.arr_delay_mean))
        rows.append(
            {
                "date": date,
                "flights": len(actual),
                "predicted_otp": round(pred_otp, 3),
                "actual_otp": round(act_otp, 3),
                "predicted_cancelled": round(fc.summary()["cancelled"]["mean"], 2),
                "actual_cancelled": sum(a.cancelled for a in actual.values()),
            }
        )
    if not probs:
        raise ConfigError("no usable days in history")
    pr = np.array(probs)
    y = np.array(outcomes, dtype=float)
    brier = float(np.mean((pr - y) ** 2))
    base = float(np.mean(y))
    live_rows = []
    for t, acc in live.items():
        if not acc["y"]:
            continue
        y_live = np.array(acc["y"])
        live_rows.append(
            {
                "at": format_time(t),
                "flights_scored": len(y_live),
                "brier_live": round(float(np.mean((np.array(acc["p"]) - y_live) ** 2)), 4),
                "brier_day_ahead": round(float(np.mean((np.array(acc["day_p"]) - y_live) ** 2)), 4),
                "mae_live_min": round(float(np.mean(acc["err"])), 1) if acc["err"] else None,
                "mae_day_ahead_min": round(float(np.mean(acc["day_err"])), 1)
                if acc["day_err"]
                else None,
            }
        )
    return BacktestResult(
        rows,
        reliability_table(probs, outcomes),
        brier,
        float(np.mean((base - y) ** 2)),
        {k: v / n_dep for k, v in hits.items()} if n_dep else {},
        float(np.mean(errors)),
        live_rows,
    )
