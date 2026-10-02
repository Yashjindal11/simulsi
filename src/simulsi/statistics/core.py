"""Statistical analysis of simulation output.

Everything here works on plain sequences of numbers (one value per
independent replication, or per Monte Carlo iteration). The confidence
intervals assume independent, identically distributed observations; for
autocorrelated output from a *single* long run use :func:`batch_means`
first. Nothing here can guarantee the model itself is right - only that the
sampling error of the reported estimates is quantified.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as _st

FloatArray = npt.NDArray[np.float64]


def _arr(values: Sequence[float] | FloatArray) -> FloatArray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError("expected a one-dimensional sequence of values")
    return a[~np.isnan(a)]


def _check_conf(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    variance: float
    min: float
    max: float
    median: float
    ci_low: float
    ci_high: float
    confidence: float
    quantiles: dict[str, float] = field(default_factory=dict)

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2

    @property
    def relative_half_width(self) -> float:
        return abs(self.half_width / self.mean) if self.mean else math.inf

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["half_width"] = self.half_width
        return d


def t_half_width(values: Sequence[float] | FloatArray, confidence: float = 0.95) -> float:
    a = _arr(values)
    _check_conf(confidence)
    if len(a) < 2:
        return math.nan
    t = float(_st.t.ppf((1 + confidence) / 2, len(a) - 1))
    return t * float(a.std(ddof=1)) / math.sqrt(len(a))


def mean_ci(values: Sequence[float] | FloatArray, confidence: float = 0.95) -> tuple[float, float]:
    """Student-t confidence interval for the mean (NaNs dropped)."""
    a = _arr(values)
    if len(a) < 2:
        return (math.nan, math.nan)
    m, h = float(a.mean()), t_half_width(a, confidence)
    return (m - h, m + h)


def summarize(
    values: Sequence[float] | FloatArray,
    confidence: float = 0.95,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> Summary:
    a = _arr(values)
    _check_conf(confidence)
    if len(a) == 0:
        nan = math.nan
        return Summary(0, nan, nan, nan, nan, nan, nan, nan, nan, confidence, {})
    lo, hi = mean_ci(a, confidence)
    qs: list[float] = np.quantile(a, list(quantiles)).tolist() if len(quantiles) else []
    var = float(a.var(ddof=1)) if len(a) > 1 else math.nan
    return Summary(
        n=len(a),
        mean=float(a.mean()),
        std=math.sqrt(var) if not math.isnan(var) else math.nan,
        variance=var,
        min=float(np.min(a)),
        max=float(np.max(a)),
        median=float(np.median(a)),
        ci_low=lo,
        ci_high=hi,
        confidence=confidence,
        quantiles={f"p{q * 100:g}": float(v) for q, v in zip(quantiles, qs, strict=True)},
    )


def bootstrap_ci(
    values: Sequence[float] | FloatArray,
    statistic: Callable[..., Any] = np.mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for any statistic (e.g. median, p95).

    ``statistic`` must accept a 2-D array and an ``axis`` keyword (NumPy
    reductions do), which lets resampling run vectorised.
    """
    a = _arr(values)
    _check_conf(confidence)
    if len(a) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    boot = np.asarray(statistic(a[idx], axis=1), dtype=np.float64)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha]).tolist()
    return (float(lo), float(hi))


def proportion_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a probability estimated from ``successes / n``."""
    _check_conf(confidence)
    if n <= 0:
        return (math.nan, math.nan)
    z = float(_st.norm.ppf((1 + confidence) / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Clamp so floating-point error never excludes the point estimate (e.g. at k = n).
    return (max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half)))


@dataclass(frozen=True)
class ReplicationAdvice:
    """Answer to "have I run enough replications?" for one metric.

    Uses the sequential approximation from Law (2015), *Simulation Modeling
    and Analysis*, sec. 9.4.1: the smallest ``n`` such that the t-based
    relative half-width, using the current variance estimate, is within
    ``target``. It is an *estimate*: the variance itself is uncertain, so
    re-check after running the suggested number.
    """

    n: int
    mean: float
    half_width: float
    relative_half_width: float
    target_relative_half_width: float
    required_n: int | None
    sufficient: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def required_replications(
    values: Sequence[float] | FloatArray,
    relative_precision: float = 0.05,
    confidence: float = 0.95,
    max_n: int = 1_000_000,
) -> ReplicationAdvice:
    a = _arr(values)
    _check_conf(confidence)
    if relative_precision <= 0:
        raise ValueError("relative_precision must be > 0")
    n = len(a)
    if n < 3:
        return ReplicationAdvice(
            n,
            float(a.mean()) if n else math.nan,
            math.nan,
            math.nan,
            relative_precision,
            None,
            False,
            "need at least 3 replications to estimate variance",
        )
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    h = t_half_width(a, confidence)
    if mean == 0:
        return ReplicationAdvice(
            n,
            mean,
            h,
            math.inf,
            relative_precision,
            None,
            False,
            "mean is zero; relative precision is undefined - use an absolute half-width",
        )
    rel = abs(h / mean)
    # Law's adjusted target accounts for using the estimated mean in the denominator.
    gamma = relative_precision / (1 + relative_precision)
    if sd == 0:
        return ReplicationAdvice(n, mean, 0.0, 0.0, relative_precision, n, True, "zero variance")
    required: int | None = None
    i = max(n, 2)
    while i <= max_n:
        t = float(_st.t.ppf((1 + confidence) / 2, i - 1))
        if t * sd / math.sqrt(i) / abs(mean) <= gamma:
            required = i
            break
        i = i + 1 if i < 1000 else int(i * 1.05) + 1
    note = "" if required is not None else f"more than {max_n} replications needed"
    return ReplicationAdvice(n, mean, h, rel, relative_precision, required, rel <= gamma, note)


@dataclass(frozen=True)
class Convergence:
    n: FloatArray
    running_mean: FloatArray
    running_half_width: FloatArray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "n": self.n.tolist(),
            "running_mean": self.running_mean.tolist(),
            "running_half_width": self.running_half_width.tolist(),
        }


def convergence(values: Sequence[float] | FloatArray, confidence: float = 0.95) -> Convergence:
    """Running mean and t half-width after each replication (for convergence plots)."""
    a = _arr(values)
    _check_conf(confidence)
    n = np.arange(1, len(a) + 1, dtype=float)
    csum = np.cumsum(a)
    csum2 = np.cumsum(a * a)
    mean = csum / n
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (csum2 - n * mean * mean) / (n - 1)
        var = np.where(var < 0, 0.0, var)
        t = np.asarray(_st.t.ppf((1 + confidence) / 2, np.maximum(n - 1, 1)), dtype=float)
        hw = np.where(n > 1, t * np.sqrt(var / n), np.nan)
    return Convergence(n, mean, hw)


def lag1_autocorrelation(values: Sequence[float] | FloatArray) -> float:
    a = _arr(values)
    if len(a) < 3:
        return math.nan
    d = a - a.mean()
    denom = float((d * d).sum())
    return float((d[:-1] * d[1:]).sum() / denom) if denom > 0 else math.nan


@dataclass(frozen=True)
class BatchMeans:
    batch_means: FloatArray
    summary: Summary
    lag1_autocorrelation: float

    @property
    def batches_look_independent(self) -> bool:
        """Heuristic: |lag-1 autocorrelation| within the ~95% band ``2/sqrt(k)``
        expected for ``k`` independent batch means. Passing does not prove independence."""
        r = self.lag1_autocorrelation
        k = len(self.batch_means)
        return not math.isnan(r) and abs(r) < 2 / math.sqrt(k)


def batch_means(
    observations: Sequence[float] | FloatArray, n_batches: int = 20, confidence: float = 0.95
) -> BatchMeans:
    """Non-overlapping batch means for a CI from one long (steady-state) run.

    Observations in the same run are usually autocorrelated (one long wait
    follows another), so the naive t-interval is too narrow. Averaging over
    large batches reduces the correlation; check ``lag1_autocorrelation`` and
    use fewer, larger batches if it is high.
    """
    a = _arr(observations)
    if n_batches < 2:
        raise ValueError("need at least 2 batches")
    size = len(a) // n_batches
    if size < 1:
        raise ValueError(f"{len(a)} observations are too few for {n_batches} batches")
    means = a[: size * n_batches].reshape(n_batches, size).mean(axis=1)
    return BatchMeans(means, summarize(means, confidence), lag1_autocorrelation(means))


@dataclass(frozen=True)
class Difference:
    """Estimated difference ``b - a`` with a confidence interval."""

    mean_a: float
    mean_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    n_a: int
    n_b: int
    confidence: float

    @property
    def relative_difference(self) -> float:
        return self.difference / abs(self.mean_a) if self.mean_a else math.nan

    @property
    def significant(self) -> bool:
        """True when the CI excludes zero (no correction for multiple comparisons)."""
        return not math.isnan(self.ci_low) and (self.ci_low > 0 or self.ci_high < 0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["relative_difference"] = self.relative_difference
        d["significant"] = self.significant
        return d


def paired_difference(
    a: Sequence[float] | FloatArray, b: Sequence[float] | FloatArray, confidence: float = 0.95
) -> Difference:
    """Paired-t CI for ``b - a`` - appropriate with common random numbers."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired comparison needs equally many observations")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    d = y - x
    n = len(d)
    if n < 2:
        nan = math.nan
        return Difference(
            float(np.mean(x)) if n else nan,
            float(np.mean(y)) if n else nan,
            float(np.mean(d)) if n else nan,
            nan,
            nan,
            nan,
            "paired-t",
            n,
            n,
            confidence,
        )
    lo, hi = mean_ci(d, confidence)
    sd = float(d.std(ddof=1))
    if sd > 0:
        p = float(_st.ttest_rel(y, x).pvalue)
    else:
        p = 1.0 if float(d.mean()) == 0 else 0.0
    return Difference(
        float(x.mean()), float(y.mean()), float(d.mean()), lo, hi, p, "paired-t", n, n, confidence
    )


def welch_difference(
    a: Sequence[float] | FloatArray, b: Sequence[float] | FloatArray, confidence: float = 0.95
) -> Difference:
    """Welch (unequal-variance) t CI for ``mean(b) - mean(a)`` from independent samples."""
    x, y = _arr(a), _arr(b)
    nx, ny = len(x), len(y)
    nan = math.nan
    if nx < 2 or ny < 2:
        return Difference(
            float(x.mean()) if nx else nan,
            float(y.mean()) if ny else nan,
            nan,
            nan,
            nan,
            nan,
            "welch-t",
            nx,
            ny,
            confidence,
        )
    vx, vy = float(x.var(ddof=1)) / nx, float(y.var(ddof=1)) / ny
    diff = float(y.mean() - x.mean())
    se = math.sqrt(vx + vy)
    if se == 0:
        return Difference(
            float(x.mean()),
            float(y.mean()),
            diff,
            diff,
            diff,
            1.0 if diff == 0 else 0.0,
            "welch-t",
            nx,
            ny,
            confidence,
        )
    df = (vx + vy) ** 2 / (vx**2 / (nx - 1) + vy**2 / (ny - 1))
    t = float(_st.t.ppf((1 + confidence) / 2, df))
    p = float(2 * _st.t.sf(abs(diff / se), df))
    return Difference(
        float(x.mean()),
        float(y.mean()),
        diff,
        diff - t * se,
        diff + t * se,
        p,
        "welch-t",
        nx,
        ny,
        confidence,
    )
