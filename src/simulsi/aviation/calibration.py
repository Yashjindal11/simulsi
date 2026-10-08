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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from simulsi.aviation.config import DelayModel, OpsConfig
from simulsi.aviation.engine import simulate_day
from simulsi.aviation.forecast import forecast
from simulsi.aviation.schedule import Flight, Schedule, parse_time
from simulsi.errors import ConfigError
from simulsi.randomness.stream import derive_seed

_HIST_ALIASES = {
    "date": ("date", "fl_date", "day"),
    "id": ("flight", "flight_id", "id"),
    "tail": ("tail", "tail_num", "registration"),
    "origin": ("origin",),
    "dest": ("dest", "destination"),
    "std": ("std", "crs_dep_time", "sched_dep"),
    "sta": ("sta", "crs_arr_time", "sched_arr"),
    "atd": ("atd", "dep_time", "actual_dep"),
    "ata": ("ata", "arr_time", "actual_arr"),
    "dep_delay": ("dep_delay",),
    "arr_delay": ("arr_delay",),
    "cancelled": ("cancelled", "canceled"),
    "pax": ("pax", "passengers"),
}


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
    """Actual flights over several days."""

    flights: list[ActualFlight] = field(default_factory=list)

    @property
    def dates(self) -> list[str]:
        return sorted({a.date for a in self.flights})

    def day(self, date: str) -> list[ActualFlight]:
        return [a for a in self.flights if a.date == date]

    def schedule(self, date: str) -> Schedule:
        return Schedule([a.flight for a in self.day(date)])

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
    def from_records(cls, rows: Iterable[Mapping[str, Any]]) -> History:
        """Rows with date, flight, tail, origin, dest, std, sta, atd/ata (or dep_delay/arr_delay), cancelled.

        Times may be ``HH:MM``, minutes or ``HHMM`` (BTS style, e.g. ``0730``).
        """
        rows = [{str(k).strip().lower(): v for k, v in r.items()} for r in rows]
        if not rows:
            raise ConfigError("history is empty")
        cols = set().union(*(r.keys() for r in rows))
        col = {k: next((n for n in names if n in cols), "") for k, names in _HIST_ALIASES.items()}
        carrier = next((n for n in ("op_unique_carrier", "op_carrier", "carrier") if n in cols), "")
        flno = "op_carrier_fl_num" if "op_carrier_fl_num" in cols else ""
        need = ["date", "origin", "dest", "std", "sta"]
        missing = [k for k in need if not col[k]] + ([] if col["id"] or flno else ["id"])
        if missing:
            raise ConfigError(f"history is missing columns {missing}")
        out = []
        counter: dict[tuple[str, str], int] = {}
        for n, r in enumerate(rows, 1):
            try:
                date = str(r[col["date"]]).strip()[:10]
                fid = (
                    str(r[col["id"]]).strip()
                    if col["id"]
                    else f"{r.get(carrier, '')}{r[flno]}".strip()
                )
                key = (date, fid)
                counter[key] = counter.get(key, 0) + 1
                if counter[key] > 1:
                    fid = f"{fid}.{counter[key]}"
                std, sta = _hhmm(r[col["std"]]), _hhmm(r[col["sta"]])
                if sta < std:
                    sta += 1440
                cancelled = str(r.get(col["cancelled"], "") or "0").strip().lower() in {
                    "1",
                    "1.0",
                    "true",
                    "yes",
                    "y",
                }
                tail = str(r.get(col["tail"], "") or "").strip() or f"?{fid}"
                pax = (
                    int(float(r[col["pax"]])) if col["pax"] and str(r[col["pax"]]).strip() else 150
                )
                f = Flight(
                    fid,
                    tail,
                    str(r[col["origin"]]).strip().upper(),
                    str(r[col["dest"]]).strip().upper(),
                    std,
                    sta,
                    pax,
                )
                atd, ata = math.nan, math.nan
                if not cancelled:
                    if col["dep_delay"] and str(r[col["dep_delay"]]).strip():
                        atd = std + float(r[col["dep_delay"]])
                    elif col["atd"] and str(r[col["atd"]]).strip():
                        atd = _actual(_hhmm(r[col["atd"]]), std)
                    if col["arr_delay"] and str(r[col["arr_delay"]]).strip():
                        ata = sta + float(r[col["arr_delay"]])
                    elif col["ata"] and str(r[col["ata"]]).strip():
                        ata = _actual(_hhmm(r[col["ata"]]), sta)
                    if math.isnan(atd) or math.isnan(ata):
                        continue  # diverted or incomplete
                out.append(ActualFlight(date, f, atd, ata, cancelled))
            except (KeyError, ValueError, ConfigError) as exc:
                raise ConfigError(f"history row {n}: {exc}") from exc
        return cls(out)

    @classmethod
    def from_csv(cls, path: str | Path) -> History:
        """A history CSV - including US BTS on-time files (FL_DATE, TAIL_NUM, CRS_DEP_TIME, ...)."""
        try:
            with Path(path).open(newline="") as fh:
                return cls.from_records(csv.DictReader(fh))
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
    cv = float(np.std(all_ratios) / np.mean(all_ratios)) if len(all_ratios) > 1 else 0.06
    bias = {k: round(float(np.mean(v)), 4) for k, v in ratios.items() if len(v) >= min_samples}
    return DelayModel(
        prob=round(prob, 4),
        mean=round(mean, 2),
        block_cv=round(cv, 4),
        table=table,
        block_bias=bias,
    )


def calibrate(
    history: History,
    config: OpsConfig | None = None,
    *,
    iterations: int = 6,
    replications: int = 10,
    max_days: int = 7,
    seed: int = 0,
    late: float = 15.0,
) -> DelayModel:
    """Fit primary delays so that *simulated* departures look like history.

    :func:`fit_delay_model` gives a first guess, but it counts everything
    that is not knock-on delay as primary - including runway queues and
    crew waits that the simulation adds by itself. This function
    re-simulates up to ``max_days`` historical schedules with ``config``
    and nudges each airport's primary-delay probability and mean until
    the simulated share of departures more than ``late`` minutes late and
    the mean departure delay match the history (a simple
    method-of-simulated-moments loop).
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
    return model


def _departure_stats(delays: Sequence[float], late: float) -> tuple[float, float]:
    return float(np.mean([d > late for d in delays])), float(np.mean([max(0.0, d) for d in delays]))


def _nudge(
    current: tuple[float, float], target: tuple[float, float], simulated: tuple[float, float]
) -> tuple[float, float]:
    prob, mean = current
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
        }


def backtest(
    history: History,
    config: OpsConfig | None = None,
    *,
    replications: int = 100,
    seed: int = 0,
    dates: Sequence[str] | None = None,
) -> BacktestResult:
    """Forecast each historical day from its schedule alone and score the forecasts."""
    cfg = config or OpsConfig()
    probs: list[float] = []
    outcomes: list[bool] = []
    hits = {"p50": 0, "p80": 0, "p95": 0}
    n_dep = 0
    rows = []
    errors = []
    for date in dates or history.dates:
        actual = history.day(date)
        schedule = Schedule([a.flight for a in actual])
        if schedule.validate():
            schedule = Schedule([a.flight for a in actual if not a.flight.tail.startswith("?")])
            if schedule.validate() or not schedule.flights:
                continue
            actual = [a for a in actual if a.flight.id in schedule.by_id]
        fc = forecast(schedule, cfg, replications=replications, seed=seed)
        flown = [a for a in actual if not a.cancelled]
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
        rows.append(
            {
                "date": date,
                "flights": len(actual),
                "predicted_otp": round(pred_otp, 3),
                "actual_otp": round(act_otp, 3),
                "predicted_cancelled": round(fc.summary()["cancelled"]["mean"], 2),
                "actual_cancelled": sum(a.cancelled for a in actual),
            }
        )
    if not probs:
        raise ConfigError("no usable days in history")
    pr = np.array(probs)
    y = np.array(outcomes, dtype=float)
    brier = float(np.mean((pr - y) ** 2))
    base = float(np.mean(y))
    return BacktestResult(
        rows,
        reliability_table(probs, outcomes),
        brier,
        float(np.mean((base - y) ** 2)),
        {k: v / n_dep for k, v in hits.items()} if n_dep else {},
        float(np.mean(errors)),
    )
