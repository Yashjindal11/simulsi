"""Scenario comparison against a baseline."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from simulsi.analysis.report import format_table
from simulsi.statistics.core import (
    AdjustMethod,
    Difference,
    adjust_p_values,
    paired_difference,
    welch_difference,
)

if TYPE_CHECKING:
    from simulsi.experiments.experiment import ExperimentResult


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    baseline: str
    scenario: str
    baseline_mean: float
    scenario_mean: float
    absolute_difference: float
    percentage_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    n: int
    significant: bool
    p_adjusted: float = math.nan

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Comparison:
    """Differences ``scenario - baseline`` for each metric, with confidence intervals.

    The intervals quantify *simulation sampling error* only. A significant
    difference means the model, as specified, responds to the change; it says
    nothing about whether the real system would (that depends on model
    validity). With ``adjust="none"`` no multiple-comparison correction is
    applied; see :func:`compare` for the alternatives.
    """

    def __init__(
        self, rows: Sequence[ComparisonRow], confidence: float, adjust: AdjustMethod = "none"
    ) -> None:
        self.rows = list(rows)
        self.confidence = confidence
        self.adjust = adjust

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rows]

    def get(self, metric: str, scenario: str) -> ComparisonRow:
        for r in self.rows:
            if r.metric == metric and r.scenario == scenario:
                return r
        raise KeyError((metric, scenario))

    def format(self) -> str:
        rows = []
        for r in self.rows:
            row: dict[str, Any] = {
                "metric": r.metric,
                "scenario": r.scenario,
                "baseline": r.baseline_mean,
                "value": r.scenario_mean,
                "diff": r.absolute_difference,
                "diff%": r.percentage_difference,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
            }
            if self.adjust != "none":
                row["p_adj"] = r.p_adjusted
            row["sig"] = "*" if r.significant else ""
            rows.append(row)
        return format_table(rows)

    def __repr__(self) -> str:
        return f"Comparison({len(self.rows)} rows, confidence={self.confidence})"


def _row(metric: str, base: str, sc: str, d: Difference) -> ComparisonRow:
    pct = d.relative_difference * 100
    return ComparisonRow(
        metric,
        base,
        sc,
        d.mean_a,
        d.mean_b,
        d.difference,
        pct,
        d.ci_low,
        d.ci_high,
        d.p_value,
        d.method,
        min(d.n_a, d.n_b),
        d.significant,
        d.p_value,
    )


def _finalize(
    specs: list[tuple[str, str, str, Any, Any, bool]], confidence: float, adjust: AdjustMethod
) -> Comparison:
    """Compute differences for ``(metric, baseline, scenario, a, b, paired)`` and apply ``adjust``."""
    m = len(specs)
    # Bonferroni has matching simultaneous intervals: each at level 1 - alpha / m.
    level = 1 - (1 - confidence) / m if adjust == "bonferroni" and m > 1 else confidence
    rows = []
    for metric, base, sc, a, b, paired in specs:
        d = paired_difference(a, b, level) if paired else welch_difference(a, b, level)
        rows.append(_row(metric, base, sc, d))
    if adjust == "none":
        return Comparison(rows, confidence, adjust)
    p_adj = adjust_p_values([r.p_value for r in rows], adjust)
    alpha = 1 - confidence
    rows = [
        replace(r, p_adjusted=p, significant=bool(not math.isnan(p) and p < alpha))
        for r, p in zip(rows, p_adj, strict=True)
    ]
    return Comparison(rows, confidence, adjust)


def compare(
    result: ExperimentResult,
    baseline: str = "baseline",
    scenarios: Iterable[str] | None = None,
    metrics: Iterable[str] | None = None,
    *,
    confidence: float = 0.95,
    adjust: AdjustMethod = "none",
) -> Comparison:
    """Compare scenarios of one experiment with its ``baseline`` scenario.

    When the experiment used common random numbers and both scenarios have the
    same replication seeds, a paired-t interval is used (replication *i* of
    each scenario saw the same random streams); otherwise Welch's t.

    ``adjust`` corrects for testing many (metric, scenario) pairs at once:
    ``"bonferroni"`` widens every interval to level ``1 - alpha/m`` (simultaneous
    coverage) and adjusts p-values; ``"holm"`` (family-wise error) and ``"bh"``
    (false discovery rate) adjust p-values only, and ``significant`` then
    means ``p_adjusted < 1 - confidence``.
    """
    if baseline not in result.scenarios:
        raise KeyError(f"baseline {baseline!r} not in experiment scenarios {result.scenarios}")
    others = [
        s for s in (scenarios if scenarios is not None else result.scenarios) if s != baseline
    ]
    names = list(metrics) if metrics is not None else result.metric_names
    specs = []
    for sc in others:
        paired = result.metadata.common_random_numbers and result.seeds(sc) == result.seeds(
            baseline
        )
        for m in names:
            a, b = result.values(m, baseline), result.values(m, sc)
            specs.append((m, baseline, sc, a, b, bool(paired and len(a) == len(b))))
    return _finalize(specs, confidence, adjust)


def compare_samples(
    baseline: Mapping[str, Sequence[float]],
    scenario: Mapping[str, Sequence[float]],
    *,
    paired: bool = False,
    confidence: float = 0.95,
    adjust: AdjustMethod = "none",
    baseline_name: str = "baseline",
    scenario_name: str = "scenario",
) -> Comparison:
    """Compare two ``{metric: values}`` samples (e.g. collected outside an Experiment)."""
    specs = []
    for m in baseline:
        if m not in scenario:
            continue
        a, b = np.asarray(baseline[m], dtype=float), np.asarray(scenario[m], dtype=float)
        specs.append((m, baseline_name, scenario_name, a, b, paired))
    return _finalize(specs, confidence, adjust)
