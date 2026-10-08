"""Operations control: compare recovery options and search for a good set of actions.

Every option is simulated with the *same* random disturbances (common
random numbers), so differences between options come from the actions,
not from luck. Costs combine delay minutes, cancellations, misconnected
passengers and (optionally) EU261 compensation - see :class:`OpsConfig`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulsi.aviation.config import OpsConfig, WeatherEvent
from simulsi.aviation.engine import simulate_day
from simulsi.aviation.forecast import Forecast, forecast
from simulsi.aviation.schedule import Schedule
from simulsi.aviation.state import Action, Cancel, OpsState, Retime, Swap, apply_actions
from simulsi.errors import ConfigError
from simulsi.randomness.stream import derive_seed

SHOWN = ("cost", "otp", "cancelled", "misconnected_pax", "delay_minutes", "eu261_compensation")


@dataclass
class PlanComparison:
    """Mean metrics per option and the paired difference in cost against the first option."""

    names: list[str]
    samples: dict[str, dict[str, np.ndarray]]
    actions: dict[str, list[Action]]

    def mean(self, option: str, metric: str) -> float:
        return float(np.nanmean(self.samples[option][metric]))

    def table(self, metrics: Sequence[str] = SHOWN) -> list[dict[str, Any]]:
        base = self.names[0]
        rows = []
        for name in self.names:
            row: dict[str, Any] = {"option": name}
            for m in metrics:
                if m in self.samples[name]:
                    row[m] = round(self.mean(name, m), 3)
            d = self.samples[name]["cost"] - self.samples[base]["cost"]
            half = 1.96 * float(np.std(d, ddof=1)) / math.sqrt(len(d)) if len(d) > 1 else math.nan
            row["cost_vs_" + base] = round(float(np.mean(d)), 1)
            row["95% CI"] = f"±{half:,.0f}" if math.isfinite(half) else "-"
            rows.append(row)
        return rows

    @property
    def best(self) -> str:
        return min(self.names, key=lambda n: self.mean(n, "cost"))

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        return format_table(self.table()) + f"\nlowest expected cost: {self.best}"


def _samples(
    schedule: Schedule,
    cfg: OpsConfig,
    actions: Sequence[Action],
    *,
    replications: int,
    seed: int,
    weather: Sequence[WeatherEvent],
    state: OpsState | None,
) -> dict[str, np.ndarray]:
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
    keys = sorted({k for d in days for k in d.metrics})
    return {k: np.array([d.metrics.get(k, math.nan) for d in days], dtype=float) for k in keys}


def compare_plans(
    schedule: Schedule,
    options: Mapping[str, Sequence[Action]],
    config: OpsConfig | None = None,
    *,
    replications: int = 100,
    seed: int = 0,
    weather: Sequence[WeatherEvent] = (),
    state: OpsState | None = None,
    include_baseline: bool = True,
) -> PlanComparison:
    """Simulate each option (a list of actions) on the same disturbances and compare.

    >>> from simulsi.aviation import Schedule, Cancel
    >>> s = Schedule.synthetic(tails=4)
    >>> cmp = compare_plans(s, {"cancel": [Cancel("F100", "F101")]}, replications=10)
    >>> cmp.names
    ['as planned', 'cancel']
    """
    cfg = config or OpsConfig()
    schedule.check()
    plans: dict[str, list[Action]] = {}
    if include_baseline:
        plans["as planned"] = []
    for name, acts in options.items():
        plans[name] = list(acts)
    if not plans:
        raise ConfigError("no options to compare")
    samples = {
        name: _samples(
            schedule, cfg, acts, replications=replications, seed=seed, weather=weather, state=state
        )
        for name, acts in plans.items()
    }
    return PlanComparison(list(plans), samples, plans)


@dataclass
class RecoveryResult:
    actions: list[Action]
    steps: list[dict[str, Any]]
    before: dict[str, float]
    after: dict[str, float]
    candidates_tried: int
    forecast_before: Forecast | None = field(default=None, repr=False)

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        lines = []
        if not self.actions:
            lines.append(
                f"No action beats the current plan ({self.candidates_tried} candidates tried)."
            )
        else:
            lines.append(f"Recommended actions ({self.candidates_tried} candidates tried):")
            lines.extend(f"  {i}. {a.describe()}" for i, a in enumerate(self.actions, 1))
        lines.append("")
        lines.append(format_table(self.steps))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.describe() for a in self.actions],
            "steps": self.steps,
            "before": self.before,
            "after": self.after,
            "candidates_tried": self.candidates_tried,
        }


def candidate_actions(
    schedule: Schedule,
    fc: Forecast,
    *,
    now: float | None = None,
    max_flights: int = 8,
    hold_minutes: Sequence[float] = (15.0,),
) -> list[Action]:
    """Plausible controller actions for the riskiest flights in a forecast.

    * cancel the round trip of a badly delayed flight,
    * swap the rest of a late aircraft's rotation with an aircraft that is
      on the ground at the same airport and expected to be on time,
    * hold a departure for a large group of connecting passengers.
    """
    t0 = -math.inf if now is None else now
    planned = [f for f in fc.flights.values() if f.status == "planned" and f.std > t0]

    def badness(f: Any) -> float:
        d = f.arr_delay_mean if math.isfinite(f.arr_delay_mean) else 0.0
        return float(d + 300 * f.p_cancel)

    risky = sorted(planned, key=badness, reverse=True)[:max_flights]
    out: list[Action] = []
    seen: set[str] = set()
    for f in risky:
        if badness(f) < 20:
            continue
        legs = schedule.rotations[f.tail]
        k = next(i for i, g in enumerate(legs) if g.id == f.id)
        if k + 1 < len(legs) and legs[k + 1].origin == f.dest and legs[k + 1].dest == f.origin:
            act: Action = Cancel(f.id, legs[k + 1].id)
            if act.describe() not in seen:
                out.append(act)
                seen.add(act.describe())
        # aircraft at the same airport whose next departure is later and expected on time
        for tail, other in schedule.rotations.items():
            if tail == f.tail:
                continue
            nxt = next((g for g in other if g.std >= f.std), None)
            prev = [g for g in other if g.std < f.std]
            if nxt is None or nxt.origin != f.origin or not prev or prev[-1].dest != f.origin:
                continue
            if not 0 < nxt.std - f.std <= 180:
                continue
            p_prev = fc.flights[prev[-1].id]
            p_late = p_prev.arr_delay_mean if math.isfinite(p_prev.arr_delay_mean) else 0.0
            if prev[-1].sta + p_late + fc.config.turn(f.origin) > f.std:
                continue
            act = Swap(f.tail, tail, f.std - 0.5)
            if act.describe() not in seen:
                out.append(act)
                seen.add(act.describe())
            break
    for c in sorted(fc.connection_risk, key=lambda c: -c["pax"] * c["p_miss"])[:max_flights]:
        if c["p_miss"] < 0.3 or c["pax"] < 15:
            continue
        if fc.flights[c["outbound"]].status != "planned" or schedule.by_id[c["outbound"]].std <= t0:
            continue
        for m in hold_minutes:
            act = Retime(c["outbound"], m)
            if act.describe() not in seen:
                out.append(act)
                seen.add(act.describe())
    return out


def recover(
    schedule: Schedule,
    config: OpsConfig | None = None,
    *,
    state: OpsState | None = None,
    weather: Sequence[WeatherEvent] = (),
    candidates: Sequence[Action] | None = None,
    max_actions: int = 3,
    replications: int = 60,
    seed: int = 0,
    min_saving: float = 0.0,
) -> RecoveryResult:
    """Greedy search for a few actions that lower the expected disruption cost.

    Each round tries every remaining candidate on top of the actions chosen
    so far (same disturbances for all), keeps the one that saves most, and
    stops when the best saving is not clearly above noise (two standard
    errors of the paired difference) or below ``min_saving``, or after
    ``max_actions`` actions. Candidates default to :func:`candidate_actions`.
    """
    cfg = config or OpsConfig()
    schedule.check()
    fc = forecast(schedule, cfg, replications=replications, seed=seed, weather=weather, state=state)
    pool = (
        list(candidates)
        if candidates is not None
        else candidate_actions(schedule, fc, now=state.now if state else None)
    )
    kw: dict[str, Any] = {
        "replications": replications,
        "seed": seed,
        "weather": weather,
        "state": state,
    }
    base = _samples(schedule, cfg, [], **kw)
    chosen: list[Action] = []
    current = base
    steps = [_step("as planned", current)]
    tried = 0
    while pool and len(chosen) < max_actions:
        best: tuple[float, float, Action, dict[str, np.ndarray]] | None = None
        for act in pool:
            try:
                apply_actions(schedule, [*chosen, act])
            except ConfigError:
                continue
            tried += 1
            s = _samples(schedule, cfg, [*chosen, act], **kw)
            diff = current["cost"] - s["cost"]
            saving = float(np.mean(diff))
            se = float(np.std(diff, ddof=1)) / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
            if best is None or saving > best[0]:
                best = (saving, se, act, s)
        if best is None or best[0] <= max(min_saving, 2 * best[1]):
            break
        saving, _se, act, s = best
        chosen.append(act)
        pool.remove(act)
        current = s
        steps.append(_step("+ " + act.describe(), current, saving))
    return RecoveryResult(
        chosen,
        steps,
        {k: float(np.nanmean(v)) for k, v in base.items()},
        {k: float(np.nanmean(v)) for k, v in current.items()},
        tried,
        fc,
    )


def _step(label: str, s: Mapping[str, np.ndarray], saving: float | None = None) -> dict[str, Any]:
    return {
        "plan": label,
        "cost": round(float(np.mean(s["cost"])), 0),
        "saving": None if saving is None else round(saving, 0),
        "otp": round(float(np.nanmean(s["otp"])), 3),
        "cancelled": round(float(np.mean(s["cancelled"])), 2),
        "misconnected_pax": round(float(np.mean(s["misconnected_pax"])), 1),
    }
