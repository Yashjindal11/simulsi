"""Scenario comparison against a baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from simulsi.analysis.report import format_table
from simulsi.statistics.core import Difference, paired_difference, welch_difference

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Comparison:
    """Differences ``scenario - baseline`` for each metric, with confidence intervals.

    The intervals quantify *simulation sampling error* only. A significant
    difference means the model, as specified, responds to the change; it says
    nothing about whether the real system would (that depends on model
    validity). No multiple-comparison correction is applied.
    """

    def __init__(self, rows: Sequence[ComparisonRow], confidence: float) -> None:
        self.rows = list(rows)
        self.confidence = confidence

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rows]

    def get(self, metric: str, scenario: str) -> ComparisonRow:
        for r in self.rows:
            if r.metric == metric and r.scenario == scenario:
                return r
        raise KeyError((metric, scenario))

    def format(self) -> str:
        return format_table(
            [
                {
                    "metric": r.metric,
                    "scenario": r.scenario,
                    "baseline": r.baseline_mean,
                    "value": r.scenario_mean,
                    "diff": r.absolute_difference,
                    "diff%": r.percentage_difference,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "sig": "*" if r.significant else "",
                }
                for r in self.rows
            ]
        )

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
    )


def compare(
    result: ExperimentResult,
    baseline: str = "baseline",
    scenarios: Iterable[str] | None = None,
    metrics: Iterable[str] | None = None,
    *,
    confidence: float = 0.95,
) -> Comparison:
    """Compare scenarios of one experiment with its ``baseline`` scenario.

    When the experiment used common random numbers and both scenarios have the
    same replication seeds, a paired-t interval is used (replication *i* of
    each scenario saw the same random streams); otherwise Welch's t.
    """
    if baseline not in result.scenarios:
        raise KeyError(f"baseline {baseline!r} not in experiment scenarios {result.scenarios}")
    others = [
        s for s in (scenarios if scenarios is not None else result.scenarios) if s != baseline
    ]
    names = list(metrics) if metrics is not None else result.metric_names
    rows = []
    for sc in others:
        paired = result.metadata.common_random_numbers and result.seeds(sc) == result.seeds(
            baseline
        )
        for m in names:
            a, b = result.values(m, baseline), result.values(m, sc)
            d = (
                paired_difference(a, b, confidence)
                if paired and len(a) == len(b)
                else welch_difference(a, b, confidence)
            )
            rows.append(_row(m, baseline, sc, d))
    return Comparison(rows, confidence)


def compare_samples(
    baseline: Mapping[str, Sequence[float]],
    scenario: Mapping[str, Sequence[float]],
    *,
    paired: bool = False,
    confidence: float = 0.95,
    baseline_name: str = "baseline",
    scenario_name: str = "scenario",
) -> Comparison:
    """Compare two ``{metric: values}`` samples (e.g. collected outside an Experiment)."""
    rows = []
    for m in baseline:
        if m not in scenario:
            continue
        a, b = np.asarray(baseline[m], dtype=float), np.asarray(scenario[m], dtype=float)
        d = paired_difference(a, b, confidence) if paired else welch_difference(a, b, confidence)
        rows.append(_row(m, baseline_name, scenario_name, d))
    return Comparison(rows, confidence)
