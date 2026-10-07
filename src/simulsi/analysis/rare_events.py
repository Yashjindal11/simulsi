"""Rare events: probabilities of extreme outcomes, with honest intervals.

"How likely is it that the hospital overflows?" is a question about a small
probability. :func:`rare_event_probability` runs replications, counts how
often ``metric`` exceeds ``threshold`` and reports an exact (Clopper-Pearson)
interval plus the number of replications needed for a target relative
error. When the event is too rare to observe often, it also fits a
generalised Pareto distribution to the upper tail (peaks over threshold,
extreme value theory) and extrapolates the probability, with a bootstrap
interval. Extrapolation assumes the tail keeps its shape beyond the data,
so treat it as an informed estimate rather than a measurement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from simulsi.analysis.report import format_table, html_table
from simulsi.core.model import Model
from simulsi.randomness.stream import derive_seed


@dataclass
class TailEstimate:
    probability: float
    ci_low: float
    ci_high: float
    shape: float
    scale: float
    tail_threshold: float
    exceedances: int


def gpd_tail_probability(
    values: Sequence[float] | np.ndarray[Any, Any],
    threshold: float,
    *,
    tail_quantile: float = 0.9,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> TailEstimate:
    """P(X > threshold) from a generalised Pareto fit to the values above a high quantile."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        raise ValueError("need at least 30 values for a tail fit")
    u = float(np.quantile(x, tail_quantile))
    if threshold <= u:
        p = float(np.mean(x > threshold))
        return TailEstimate(p, p, p, math.nan, math.nan, u, int(np.sum(x > u)))

    def estimate(sample: np.ndarray[Any, Any]) -> tuple[float, float, float]:
        excess = sample[sample > u] - u
        if len(excess) < 5:
            return math.nan, math.nan, math.nan
        shape, _, scale = stats.genpareto.fit(excess, floc=0)
        p_u = len(excess) / len(sample)
        return (
            p_u * float(stats.genpareto.sf(threshold - u, shape, loc=0, scale=scale)),
            shape,
            scale,
        )

    p, shape, scale = estimate(x)
    rng = np.random.default_rng(seed)
    boots = [estimate(rng.choice(x, size=len(x), replace=True))[0] for _ in range(n_bootstrap)]
    boots = [b for b in boots if math.isfinite(b)]
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha]) if boots else (math.nan, math.nan)
    return TailEstimate(p, float(lo), float(hi), float(shape), float(scale), u, int(np.sum(x > u)))


@dataclass
class RareEventResult:
    metric: str
    threshold: float
    n: int
    hits: int
    probability: float
    ci_low: float
    ci_high: float
    relative_error: float
    replications_for_10pct: float
    tail: TailEstimate | None

    def _rows(self) -> list[dict[str, Any]]:
        rows = [
            {
                "method": "empirical (Clopper-Pearson)",
                "probability": self.probability,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "detail": f"{self.hits} of {self.n} runs",
            }
        ]
        if self.tail is not None:
            rows.append(
                {
                    "method": "tail extrapolation (GPD)",
                    "probability": self.tail.probability,
                    "ci_low": self.tail.ci_low,
                    "ci_high": self.tail.ci_high,
                    "detail": f"shape {self.tail.shape:.3g}, {self.tail.exceedances} tail points",
                }
            )
        return rows

    def format(self) -> str:
        head = f"P({self.metric} > {self.threshold:g})"
        lines = [head, format_table(self._rows())]
        if math.isfinite(self.replications_for_10pct):
            lines.append(
                f"about {self.replications_for_10pct:,.0f} replications give a 10% relative error"
            )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return html_table(self._rows(), caption=f"P({self.metric} > {self.threshold:g})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "n": self.n,
            "hits": self.hits,
            "probability": self.probability,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "relative_error": self.relative_error,
            "replications_for_10pct": self.replications_for_10pct,
            "tail": None if self.tail is None else self.tail.__dict__,
        }


def clopper_pearson(hits: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval."""
    alpha = 1 - confidence
    lo = 0.0 if hits == 0 else float(stats.beta.ppf(alpha / 2, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(stats.beta.ppf(1 - alpha / 2, hits + 1, n - hits))
    return lo, hi


def rare_event_probability(
    model: Model,
    metric: str,
    threshold: float,
    *,
    replications: int = 200,
    seed: int = 0,
    params: Mapping[str, Any] | None = None,
    tail: bool = True,
    confidence: float = 0.95,
) -> RareEventResult:
    """Estimate P(metric > threshold) from ``replications`` independent runs."""
    if replications < 2:
        raise ValueError("replications must be >= 2")
    values = []
    for r in range(replications):
        m = model.simulate(params, seed=derive_seed(seed, "replication", r)).metrics
        if metric not in m:
            raise KeyError(f"model produced no metric {metric!r}")
        values.append(m[metric])
    x = np.asarray(values, dtype=float)
    hits = int(np.sum(x > threshold))
    p = hits / len(x)
    lo, hi = clopper_pearson(hits, len(x), confidence)
    rel = math.sqrt((1 - p) / (p * len(x))) if p > 0 else math.inf
    need = (1 - p) / (p * 0.1**2) if p > 0 else math.inf
    tail_est = None
    if tail and len(x) >= 30:
        tail_est = gpd_tail_probability(x, threshold, confidence=confidence, seed=seed)
    return RareEventResult(metric, threshold, len(x), hits, p, lo, hi, rel, need, tail_est)
