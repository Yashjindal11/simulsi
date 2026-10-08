"""Monte Carlo forecasts of a day of operations: per flight, per rotation, per connection.

:func:`forecast` simulates the day many times (from the plan, or from the
live state at ``now``) and turns the outcomes into probabilities and delay
ranges: the chance each flight is on time or cancelled, its departure delay
quantiles, the expected passengers who misconnect, which aircraft
rotations are fragile, and alerts when risks cross thresholds.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from simulsi.aviation.config import OpsConfig, WeatherEvent
from simulsi.aviation.engine import DayOutcome, simulate_day
from simulsi.aviation.schedule import Schedule, format_time
from simulsi.aviation.state import Action, FlightStatus, OpsState, apply_actions
from simulsi.randomness.stream import derive_seed

CAUSES = ("primary", "aircraft", "crew", "runway")


@dataclass
class FlightForecast:
    id: str
    tail: str
    origin: str
    dest: str
    std: float
    sta: float
    pax: int
    p_on_time: float
    p_departure_on_time: float
    p_cancel: float
    dep_delay_p50: float
    dep_delay_p80: float
    dep_delay_p95: float
    arr_delay_mean: float
    p_crew_risk: float
    p_spare: float
    p_standby: float
    misconnect_pax: float
    main_cause: str
    status: str = "planned"

    def row(self) -> dict[str, Any]:
        d = asdict(self)
        d["std"], d["sta"] = format_time(self.std), format_time(self.sta)
        d["etd_p50"] = (
            format_time(self.std + self.dep_delay_p50) if math.isfinite(self.dep_delay_p50) else "-"
        )
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3) if math.isfinite(v) else None
        return d


@dataclass
class Alert:
    level: str
    kind: str
    message: str
    flight: str = ""

    def __str__(self) -> str:
        return f"[{self.level}] {self.kind}: {self.message}"


class Forecast:
    """Per-flight and network forecasts from ``replications`` simulated days."""

    def __init__(
        self,
        schedule: Schedule,
        config: OpsConfig,
        days: Sequence[DayOutcome],
        *,
        now: float | None = None,
        state: OpsState | None = None,
        weather: Sequence[WeatherEvent] = (),
        cancelled: Sequence[str] = (),
    ) -> None:
        self.schedule = schedule
        self.config = config
        self.replications = len(days)
        self.now = now
        self.weather = list(weather)
        self.days = list(days)
        self.flights: dict[str, FlightForecast] = {}
        statuses = state.flights if state else {}
        cancelled_set = set(cancelled)
        names = sorted({k for d in days for k in d.metrics})
        self.metric_samples: dict[str, np.ndarray] = {
            k: np.array([d.metrics.get(k, math.nan) for d in days], dtype=float) for k in names
        }
        miss_by_flight: dict[str, float] = {}
        miss_conn: dict[tuple[str, str], int] = {}
        for d in days:
            for (i, o), n in d.misconnected.items():
                miss_conn[(i, o)] = miss_conn.get((i, o), 0) + 1
                miss_by_flight[i] = miss_by_flight.get(i, 0.0) + n / len(days)
        self.connection_risk: list[dict[str, Any]] = [
            {
                "inbound": c.inbound,
                "outbound": c.outbound,
                "pax": c.pax,
                "p_miss": miss_conn.get((c.inbound, c.outbound), 0) / len(days),
                "scheduled_connection": schedule.by_id[c.outbound].std
                - schedule.by_id[c.inbound].sta,
            }
            for c in schedule.connections
        ]
        for f in schedule.flights:
            outs = [d.flights[f.id] for d in days]
            flown = [o for o in outs if not o.cancelled]
            dep = np.array([o.dep_delay for o in flown], dtype=float)
            arr = np.array([o.arr_delay for o in flown], dtype=float)
            n = len(outs)
            causes = {
                "primary": sum(o.primary for o in flown),
                "aircraft": sum(o.reactionary_aircraft for o in flown),
                "crew": sum(o.reactionary_crew for o in flown),
                "runway": sum(o.runway for o in flown),
            }
            main = max(causes, key=causes.__getitem__) if max(causes.values()) > 1e-9 else "-"
            st = statuses.get(f.id, FlightStatus())
            status = (
                "cancelled"
                if st.cancelled or f.id in cancelled_set
                else "landed"
                if st.ata is not None
                else "airborne"
                if st.atd is not None
                else "planned"
            )
            self.flights[f.id] = FlightForecast(
                id=f.id,
                tail=f.tail,
                origin=f.origin,
                dest=f.dest,
                std=f.std,
                sta=f.sta,
                pax=f.pax,
                p_on_time=float(np.sum(arr <= config.on_time)) / n,
                p_departure_on_time=float(np.sum(dep <= config.on_time)) / n,
                p_cancel=1 - len(flown) / n,
                dep_delay_p50=_quantile(dep, 0.5),
                dep_delay_p80=_quantile(dep, 0.8),
                dep_delay_p95=_quantile(dep, 0.95),
                arr_delay_mean=float(np.mean(np.maximum(arr, 0))) if len(arr) else math.nan,
                p_crew_risk=sum(o.crew_risk for o in outs) / n,
                p_spare=sum(o.spare for o in outs) / n,
                p_standby=sum(o.standby for o in outs) / n,
                misconnect_pax=miss_by_flight.get(f.id, 0.0),
                main_cause=main,
                status=status,
            )

    # -- views -----------------------------------------------------------------

    def summary(self) -> dict[str, dict[str, float]]:
        """Network metrics: mean and 10th/90th percentiles over the simulated days."""
        out = {}
        for k, v in self.metric_samples.items():
            v = v[np.isfinite(v)]
            if len(v):
                out[k] = {
                    "mean": float(np.mean(v)),
                    "p10": float(np.quantile(v, 0.1)),
                    "p90": float(np.quantile(v, 0.9)),
                }
        return out

    def table(self, *, sort: str = "std") -> list[dict[str, Any]]:
        rows = [f.row() for f in self.flights.values()]
        if sort != "std":
            key = sort.lstrip("-")
            rows.sort(
                key=lambda r: (r.get(key) is None, r.get(key) or 0), reverse=sort.startswith("-")
            )
        return rows

    def rotations(self) -> list[dict[str, Any]]:
        """Aircraft rotations ranked by risk (expected delay minutes plus cancellations)."""
        out: list[dict[str, Any]] = []
        for tail, legs in self.schedule.rotations.items():
            fc = [self.flights[f.id] for f in legs]
            exp_delay = sum(
                (1 - f.p_cancel) * (f.arr_delay_mean if math.isfinite(f.arr_delay_mean) else 0)
                for f in fc
            )
            p_cancel = 1 - float(np.prod([1 - f.p_cancel for f in fc]))
            worst = max(fc, key=lambda f: f.dep_delay_p80 if math.isfinite(f.dep_delay_p80) else -1)
            first_late = next((f for f in fc if f.p_on_time < 0.8), None)
            out.append(
                {
                    "tail": tail,
                    "legs": len(legs),
                    "expected_delay_minutes": round(exp_delay, 1),
                    "p_any_cancel": round(p_cancel, 3),
                    "worst_leg": worst.id,
                    "worst_p80_delay": round(worst.dep_delay_p80, 1)
                    if math.isfinite(worst.dep_delay_p80)
                    else None,
                    "first_at_risk": first_late.id if first_late else "",
                    "risk_score": round(
                        exp_delay
                        + self.config.cancel_cost
                        / max(self.config.delay_cost_per_minute, 1e-9)
                        * p_cancel,
                        1,
                    ),
                }
            )
        out.sort(key=lambda r: -float(r["risk_score"]))
        return out

    def airports(self) -> list[dict[str, Any]]:
        out = []
        for a in self.schedule.airports:
            deps = [f for f in self.flights.values() if f.origin == a]
            if not deps:
                continue
            out.append(
                {
                    "airport": a,
                    "departures": len(deps),
                    "p_on_time": round(sum(f.p_departure_on_time for f in deps) / len(deps), 3),
                    "mean_p50_delay": round(
                        float(np.nanmean([f.dep_delay_p50 for f in deps]))
                        if any(math.isfinite(f.dep_delay_p50) for f in deps)
                        else math.nan,
                        1,
                    ),
                    "expected_cancellations": round(sum(f.p_cancel for f in deps), 2),
                }
            )
        return out

    def alerts(
        self,
        *,
        cancel: float = 0.3,
        delay_p80: float = 60.0,
        crew: float = 0.3,
        misconnect: float = 0.5,
        min_pax: int = 10,
    ) -> list[Alert]:
        """Warnings for flights and connections whose risks cross the thresholds."""
        out: list[Alert] = []
        for f in self.flights.values():
            if f.status != "planned":
                continue
            if f.p_cancel >= cancel:
                out.append(
                    Alert(
                        "high",
                        "cancellation",
                        f"{f.id} {f.origin}-{f.dest} {format_time(f.std)}: {f.p_cancel:.0%} chance of cancellation",
                        f.id,
                    )
                )
            elif math.isfinite(f.dep_delay_p80) and f.dep_delay_p80 >= delay_p80:
                out.append(
                    Alert(
                        "medium",
                        "delay",
                        f"{f.id} {f.origin}-{f.dest} {format_time(f.std)}: 1 in 5 chance of {f.dep_delay_p80:.0f}+ min delay (mostly {f.main_cause})",
                        f.id,
                    )
                )
            if f.p_crew_risk >= crew:
                out.append(
                    Alert(
                        "high",
                        "crew duty",
                        f"{f.id}: crew {self.schedule.by_id[f.id].crew or '?'} may exceed the duty limit ({f.p_crew_risk:.0%})",
                        f.id,
                    )
                )
        for c in self.connection_risk:
            if c["p_miss"] >= misconnect and c["pax"] >= min_pax:
                out.append(
                    Alert(
                        "medium",
                        "connection",
                        f"{c['pax']} pax {c['inbound']}->{c['outbound']}: {c['p_miss']:.0%} chance to misconnect",
                        c["inbound"],
                    )
                )
        for w in self.weather:
            out.append(
                Alert(
                    "info",
                    "weather",
                    f"{w.name or w.airport} {format_time(w.start)}-{format_time(w.end)} capacity {w.capacity:.0%}, p={w.probability:.0%}",
                )
            )
        order = {"high": 0, "medium": 1, "info": 2}
        out.sort(key=lambda a: order.get(a.level, 3))
        return out

    def gantt(self) -> list[dict[str, Any]]:
        """Bars for a tail-by-time chart: planned times plus predicted median delay."""
        rows = []
        for tail, legs in self.schedule.rotations.items():
            for f in legs:
                fc = self.flights[f.id]
                d = fc.dep_delay_p50 if math.isfinite(fc.dep_delay_p50) else 0.0
                rows.append(
                    {
                        "tail": tail,
                        "flight": f.id,
                        "origin": f.origin,
                        "dest": f.dest,
                        "std": f.std,
                        "sta": f.sta,
                        "etd": f.std + max(0.0, d),
                        "eta": f.sta + max(0.0, d),
                        "p_on_time": fc.p_on_time,
                        "p_cancel": fc.p_cancel,
                        "p80": fc.dep_delay_p80 if math.isfinite(fc.dep_delay_p80) else None,
                        "cause": fc.main_cause,
                        "status": fc.status,
                    }
                )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "replications": self.replications,
            "now": self.now,
            "summary": self.summary(),
            "flights": self.table(),
            "rotations": self.rotations(),
            "airports": self.airports(),
            "connections": self.connection_risk,
            "alerts": [asdict(a) for a in self.alerts()],
            "gantt": self.gantt(),
            "weather": [w.to_dict() for w in self.weather],
        }

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(jsonable(self.to_dict()), indent=2, allow_nan=False, default=_nan_none)
        if path is not None:
            Path(path).write_text(text)
        return text

    def to_csv(self, path: str | Path | None = None) -> str:
        rows = self.table()
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text)
        return text

    def format(self, top: int = 15) -> str:
        from simulsi.analysis.report import format_table

        s = self.summary()
        lines = [
            f"Forecast from {self.replications} simulated days"
            + (f", starting at {format_time(self.now)}" if self.now is not None else ""),
        ]
        for key, label, pct in (
            ("otp", "on-time arrivals (A15)", True),
            ("cancelled", "cancellations", False),
            ("misconnected_pax", "misconnected passengers", False),
            ("arrival_delay.mean", "mean arrival delay (min)", False),
            ("cost", "disruption cost", False),
        ):
            if key in s:
                v = s[key]
                spec = ".1%" if pct else ",.1f"
                lines.append(
                    f"  {label:<28} {v['mean']:{spec}}  (80% range {v['p10']:{spec}} - {v['p90']:{spec}})"
                )
        rows = sorted(
            (f for f in self.flights.values() if f.status == "planned"),
            key=lambda f: (f.p_on_time, -f.p_cancel),
        )[:top]
        lines.append("")
        lines.append(f"Flights most at risk (top {len(rows)} not yet departed):")
        lines.append(
            format_table(
                [
                    {
                        "flight": f.id,
                        "route": f"{f.origin}-{f.dest}",
                        "std": format_time(f.std),
                        "p_on_time": round(f.p_on_time, 2),
                        "p_cancel": round(f.p_cancel, 2),
                        "delay p50/p80/p95": "/".join(
                            "-" if not math.isfinite(x) else f"{x:.0f}"
                            for x in (f.dep_delay_p50, f.dep_delay_p80, f.dep_delay_p95)
                        ),
                        "cause": f.main_cause,
                    }
                    for f in rows
                ]
            )
        )
        rot = self.rotations()[:5]
        lines.append("")
        lines.append("Most fragile rotations:")
        lines.append(format_table(rot))
        alerts = self.alerts()
        if alerts:
            lines.append("")
            lines.append(f"Alerts ({len(alerts)}):")
            lines.extend(f"  {a}" for a in alerts[:20])
            if len(alerts) > 20:
                lines.append(f"  ... {len(alerts) - 20} more")
        return "\n".join(lines)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else math.nan


def _nan_none(x: Any) -> Any:
    if isinstance(x, np.floating | np.integer):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    return obj


def forecast(
    schedule: Schedule,
    config: OpsConfig | None = None,
    *,
    replications: int = 200,
    seed: int = 0,
    weather: Sequence[WeatherEvent] = (),
    state: OpsState | None = None,
    actions: Sequence[Action] = (),
) -> Forecast:
    """Simulate the day ``replications`` times and summarise per flight and per network.

    Pass ``state`` to forecast the rest of the day from the live situation,
    and ``actions`` to forecast a changed plan. The same ``seed`` gives the
    same disturbances, so two forecasts with different actions compare
    like with like.

    >>> from simulsi.aviation import Schedule
    >>> fc = forecast(Schedule.synthetic(tails=4), replications=20)
    >>> 0 <= fc.summary()["otp"]["mean"] <= 1
    True
    """
    if replications < 1:
        raise ValueError("replications must be >= 1")
    cfg = config or OpsConfig()
    schedule.check()
    days = [
        simulate_day(
            schedule,
            cfg,
            seed=derive_seed(seed, "replication", r),
            weather=weather,
            state=state,
            actions=actions,
        )
        for r in range(replications)
    ]
    plan, cancelled = apply_actions(schedule, actions) if actions else (schedule, set())
    return Forecast(
        plan,
        cfg,
        days,
        now=state.now if state else None,
        state=state,
        weather=weather,
        cancelled=sorted(cancelled),
    )


def state_at(schedule: Schedule, day: DayOutcome, now: float) -> OpsState:
    """The live state at ``now`` of a simulated (or real) day: what had departed, landed or been cancelled.

    Useful to test rolling forecasts against a known "truth" day.
    """
    flights: dict[str, FlightStatus] = {}
    for f in schedule.flights:
        o = day.flights[f.id]
        if o.cancelled and f.std - 60 <= now:
            flights[f.id] = FlightStatus(cancelled=True)
        elif not o.cancelled and o.dep <= now:
            flights[f.id] = FlightStatus(atd=o.dep, ata=o.arr if o.arr <= now else None)
    return OpsState(now, flights)


def rolling_forecast(
    schedule: Schedule,
    config: OpsConfig | None = None,
    *,
    truth: DayOutcome,
    times: Sequence[float],
    replications: int = 100,
    seed: int = 0,
    weather: Sequence[WeatherEvent] = (),
) -> list[dict[str, Any]]:
    """Re-forecast through a day as it unfolds, and score each forecast against what happened.

    ``truth`` is the day that "really" happened (a simulated day or real
    actuals). For each time in ``times`` the forecast starts from the
    state at that time; the rows report the predicted OTP against the
    actual one and the mean absolute error of predicted arrival delays.
    """
    cfg = config or OpsConfig()
    actual_otp = truth.metrics["otp"]
    rows = []
    for t in times:
        fc = forecast(
            schedule,
            cfg,
            replications=replications,
            seed=seed,
            weather=weather,
            state=state_at(schedule, truth, t),
        )
        errors = []
        for f in schedule.flights:
            o = truth.flights[f.id]
            p = fc.flights[f.id]
            if not o.cancelled and math.isfinite(p.arr_delay_mean) and f.std > t:
                errors.append(abs(max(0.0, o.arr_delay) - p.arr_delay_mean))
        s = fc.summary()["otp"]
        rows.append(
            {
                "time": format_time(t),
                "predicted_otp": round(s["mean"], 3),
                "range_p10": round(s["p10"], 3),
                "range_p90": round(s["p90"], 3),
                "actual_otp": round(actual_otp, 3),
                "remaining_flights": len(errors),
                "mae_arrival_delay": round(float(np.mean(errors)), 1) if errors else None,
            }
        )
    return rows
