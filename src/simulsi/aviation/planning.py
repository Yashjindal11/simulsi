"""Planning before the day (and months ahead): reserves, schedule buffers, schedule changes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from simulsi.aviation.config import OpsConfig, WeatherEvent
from simulsi.aviation.engine import _hub, simulate_day
from simulsi.aviation.schedule import Schedule, format_time
from simulsi.optimization.pareto import pareto_front
from simulsi.randomness.stream import derive_seed


def _metric_samples(
    schedule: Schedule,
    cfg: OpsConfig,
    *,
    replications: int,
    seed: int,
    weather: Sequence[WeatherEvent],
) -> dict[str, np.ndarray]:
    days = [
        simulate_day(schedule, cfg, seed=derive_seed(seed, "replication", r), weather=weather)
        for r in range(replications)
    ]
    keys = sorted({k for d in days for k in d.metrics})
    return {k: np.array([d.metrics.get(k, math.nan) for d in days], dtype=float) for k in keys}


@dataclass
class ReservePlan:
    rows: list[dict[str, Any]]
    front: list[int]
    recommended: dict[str, Any]
    airport: str

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        marked = [{**r, "pareto": "*" if i in self.front else ""} for i, r in enumerate(self.rows)]
        rec = self.recommended
        return (
            format_table(marked)
            + "\n* = Pareto front: no other mix has both lower reserve cost and lower disruption cost"
            + f"\nlowest total cost at {self.airport}: {rec['spares']} spare aircraft, "
            + f"{rec['standby_crews']} standby crews"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "airport": self.airport,
            "rows": self.rows,
            "front": self.front,
            "recommended": self.recommended,
        }


def plan_reserves(
    schedule: Schedule,
    config: OpsConfig | None = None,
    *,
    airport: str | None = None,
    spares: Sequence[int] = (0, 1, 2, 3),
    standby_crews: Sequence[int] = (0, 1, 2, 3),
    spare_cost: float = 15000.0,
    standby_cost: float = 2500.0,
    weather: Sequence[WeatherEvent] = (),
    replications: int = 60,
    seed: int = 0,
) -> ReservePlan:
    """How many spare aircraft and standby crews to hold at ``airport`` (default: the hub).

    Every combination is simulated on the same disturbances; the total cost
    adds the daily cost of the reserves to the expected disruption cost.
    The Pareto front marks combinations where no other one has both a lower
    reserve cost and a lower expected disruption cost.
    """
    cfg0 = config or OpsConfig()
    schedule.check()
    where = airport or _hub(schedule)
    rows = []
    for s in spares:
        for c in standby_crews:
            cfg = cfg0.replace(
                spares={**cfg0.spares, where: s}, standby_crews={**cfg0.standby_crews, where: c}
            )
            m = _metric_samples(
                schedule, cfg, replications=replications, seed=seed, weather=weather
            )
            reserve = s * spare_cost + c * standby_cost
            disruption = float(np.mean(m["cost"]))
            rows.append(
                {
                    "spares": s,
                    "standby_crews": c,
                    "otp": round(float(np.nanmean(m["otp"])), 3),
                    "cancelled": round(float(np.mean(m["cancelled"])), 2),
                    "misconnected_pax": round(float(np.mean(m["misconnected_pax"])), 1),
                    "reserve_cost": reserve,
                    "disruption_cost": round(disruption, 0),
                    "total_cost": round(reserve + disruption, 0),
                }
            )
    front = pareto_front([(r["reserve_cost"], r["disruption_cost"]) for r in rows], ["min", "min"])
    best = min(rows, key=lambda r: r["total_cost"])
    return ReservePlan(rows, front, best, where)


@dataclass
class BufferPlan:
    shifts: dict[str, float]
    added: dict[str, float]
    before: dict[str, float]
    after: dict[str, float]
    schedule: Schedule

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        rows = [
            {"turn before": fid, "extra minutes": m}
            for fid, m in sorted(self.added.items(), key=lambda kv: -kv[1])
        ]
        lines = [format_table(rows) if rows else "no buffer added"]
        for k, label in (
            ("otp", "OTP"),
            ("delay_minutes", "delay minutes"),
            ("cost", "cost"),
            ("last_arrival", "last arrival"),
        ):
            b, a = self.before.get(k, math.nan), self.after.get(k, math.nan)
            if k == "last_arrival":
                lines.append(f"{label:>14}: {format_time(b)} -> {format_time(a)}")
            elif k == "otp":
                lines.append(f"{label:>14}: {b:.1%} -> {a:.1%}")
            else:
                lines.append(f"{label:>14}: {b:,.0f} -> {a:,.0f}")
        return "\n".join(lines)


def optimize_buffers(
    schedule: Schedule,
    config: OpsConfig | None = None,
    *,
    budget: float = 60.0,
    step: float = 5.0,
    per_tail_limit: float | None = None,
    replications: int = 100,
    seed: int = 0,
    weather: Sequence[WeatherEvent] = (),
) -> BufferPlan:
    """Spend ``budget`` minutes of extra ground time where it stops the most delay.

    Simulates the plan, records how late each aircraft was for each turn
    (knock-on delay), and adds buffer in ``step``-minute pieces to the turn
    with the largest expected benefit: the knock-on delay a piece absorbs
    times the number of legs it protects (that leg and the rest of the
    rotation). Later legs of the tail move later by the added buffer. The
    new schedule is then simulated on the same disturbances.
    """
    cfg = config or OpsConfig()
    schedule.check()
    days = [
        simulate_day(schedule, cfg, seed=derive_seed(seed, "replication", r), weather=weather)
        for r in range(replications)
    ]
    react: dict[str, np.ndarray] = {}
    weight: dict[str, int] = {}
    tail_of: dict[str, str] = {}
    for tail, legs in schedule.rotations.items():
        for k, f in enumerate(legs[1:], 1):
            react[f.id] = np.array(
                [
                    0.0 if d.flights[f.id].cancelled else d.flights[f.id].reactionary_aircraft
                    for d in days
                ]
            )
            weight[f.id] = len(legs) - k
            tail_of[f.id] = tail
    added: dict[str, float] = dict.fromkeys(react, 0.0)
    per_tail: dict[str, float] = {}
    spent = 0.0
    while spent + step <= budget + 1e-9 and react:

        def gain(fid: str) -> float:
            x = added[fid]
            if (
                per_tail_limit is not None
                and per_tail.get(tail_of[fid], 0.0) + step > per_tail_limit
            ):
                return -1.0
            r = react[fid]
            return float(np.mean(np.minimum(r, x + step) - np.minimum(r, x))) * weight[fid]

        best = max(react, key=gain)
        if gain(best) <= 0:
            break
        added[best] += step
        per_tail[tail_of[best]] = per_tail.get(tail_of[best], 0.0) + step
        spent += step
    shifts: dict[str, float] = {}
    for _tail, legs in schedule.rotations.items():
        total = 0.0
        for f in legs:
            total += added.get(f.id, 0.0)
            if total:
                shifts[f.id] = total
    new = schedule.retime(shifts)
    kw: dict[str, Any] = {"replications": replications, "seed": seed, "weather": weather}
    before = {k: float(np.nanmean(v)) for k, v in _metric_samples(schedule, cfg, **kw).items()}
    after = {k: float(np.nanmean(v)) for k, v in _metric_samples(new, cfg, **kw).items()}
    return BufferPlan(shifts, {k: v for k, v in added.items() if v}, before, after, new)


@dataclass
class ScheduleImpact:
    rows: list[dict[str, Any]]
    airports: list[dict[str, Any]]

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        return format_table(self.rows) + "\n\nby departure airport:\n" + format_table(self.airports)


def schedule_impact(
    base: Schedule,
    changed: Schedule,
    config: OpsConfig | None = None,
    *,
    names: tuple[str, str] = ("base", "changed"),
    replications: int = 100,
    seed: int = 0,
    weather: Sequence[WeatherEvent] = (),
) -> ScheduleImpact:
    """Compare two schedules (an added rotation, a new bank structure, retimed flights).

    Flights with the same id see the same random disturbances in both, so
    the difference shows the effect of the change itself.
    """
    cfg = config or OpsConfig()
    base.check()
    changed.check()
    kw: dict[str, Any] = {"replications": replications, "seed": seed, "weather": weather}
    a = _metric_samples(base, cfg, **kw)
    b = _metric_samples(changed, cfg, **kw)
    rows = []
    for key in (
        "flights",
        "otp",
        "arrival_delay.mean",
        "cancelled",
        "misconnected_pax",
        "misconnect_rate",
        "gate_wait.mean",
        "runway_share",
        "cost",
    ):
        if key in a and key in b:
            va, vb = float(np.nanmean(a[key])), float(np.nanmean(b[key]))
            rows.append(
                {
                    "metric": key,
                    names[0]: round(va, 3),
                    names[1]: round(vb, 3),
                    "change": round(vb - va, 3),
                }
            )
    airports = []
    for ap in sorted(set(base.airports) | set(changed.airports)):
        na = sum(1 for f in base.flights if f.origin == ap)
        nb = sum(1 for f in changed.flights if f.origin == ap)
        if na or nb:
            peak_a, peak_b = _peak_hour(base, ap), _peak_hour(changed, ap)
            airports.append(
                {
                    "airport": ap,
                    f"departures {names[0]}": na,
                    f"departures {names[1]}": nb,
                    f"peak/h {names[0]}": peak_a,
                    f"peak/h {names[1]}": peak_b,
                }
            )
    return ScheduleImpact(rows, airports)


def _peak_hour(schedule: Schedule, airport: str) -> int:
    times = sorted(f.std for f in schedule.flights if f.origin == airport)
    best, j = 0, 0
    for i, t in enumerate(times):
        while times[j] < t - 60 + 1e-9:
            j += 1
        best = max(best, i - j + 1)
    return best
