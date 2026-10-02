from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as st

from simulsi.statistics import (
    batch_means,
    bootstrap_ci,
    convergence,
    lag1_autocorrelation,
    mean_ci,
    paired_difference,
    proportion_ci,
    required_replications,
    summarize,
    welch_difference,
)


def test_mean_ci_matches_scipy() -> None:
    x = np.random.default_rng(1).normal(10, 3, 30)
    lo, hi = mean_ci(x, 0.95)
    ref = st.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=st.sem(x))
    assert lo == pytest.approx(ref[0]) and hi == pytest.approx(ref[1])


def test_summarize_basic_and_empty() -> None:
    s = summarize([1, 2, 3, 4, float("nan")])
    assert s.n == 4 and s.mean == 2.5 and s.median == 2.5
    assert s.quantiles["p50"] == 2.5
    assert s.ci_low < 2.5 < s.ci_high
    e = summarize([])
    assert e.n == 0 and math.isnan(e.mean)
    one = summarize([5.0])
    assert one.mean == 5 and math.isnan(one.ci_low)


def test_confidence_validation() -> None:
    with pytest.raises(ValueError):
        summarize([1, 2], confidence=1.5)


def test_t_interval_coverage_is_close_to_nominal() -> None:
    rng = np.random.default_rng(7)
    hits = 0
    trials = 2000
    for _ in range(trials):
        lo, hi = mean_ci(rng.exponential(2.0, 25), 0.9)
        hits += lo <= 2.0 <= hi
    # skewed data at n=25: coverage a little under 90% is expected
    assert 0.85 <= hits / trials <= 0.93


def test_bootstrap_ci_contains_median() -> None:
    x = np.random.default_rng(3).lognormal(0, 1, 400)
    lo, hi = bootstrap_ci(x, np.median, seed=1)
    assert lo < 1.0 < hi
    assert bootstrap_ci(x, np.median, seed=1) == (lo, hi)


def test_proportion_ci() -> None:
    lo, hi = proportion_ci(0, 100)
    assert lo == pytest.approx(0, abs=1e-12) and 0 < hi < 0.05
    lo, hi = proportion_ci(50, 100)
    assert lo < 0.5 < hi
    assert all(math.isnan(v) for v in proportion_ci(0, 0))


def test_required_replications() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(100, 20, 10)
    adv = required_replications(x, relative_precision=0.01)
    assert not adv.sufficient and adv.required_n is not None and adv.required_n > 10
    # with that many replications the precision target should be met (roughly)
    y = rng.normal(100, 20, adv.required_n)
    assert required_replications(y, 0.01).relative_half_width < 0.013
    assert required_replications([1, 1, 1], 0.05).sufficient
    assert required_replications([1, 2], 0.05).required_n is None
    assert required_replications([-1, 0, 1], 0.05).note.startswith("mean is zero")


def test_convergence_running_values() -> None:
    c = convergence([1, 2, 3, 4])
    assert c.running_mean.tolist() == [1, 1.5, 2, 2.5]
    assert math.isnan(c.running_half_width[0])
    assert c.running_half_width[-1] == pytest.approx(mean_ci([1, 2, 3, 4])[1] - 2.5)


def test_batch_means_on_ar1() -> None:
    rng = np.random.default_rng(5)
    n, phi = 50_000, 0.9
    x = np.empty(n)
    x[0] = 0
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    assert lag1_autocorrelation(x) > 0.85
    bm = batch_means(x, n_batches=20)
    assert bm.summary.ci_low < 0 < bm.summary.ci_high
    assert bm.batches_look_independent
    with pytest.raises(ValueError):
        batch_means([1, 2, 3], n_batches=10)


def test_paired_vs_welch() -> None:
    rng = np.random.default_rng(11)
    common = rng.normal(50, 10, 40)
    a = common + rng.normal(0, 0.5, 40)
    b = common + 1.0 + rng.normal(0, 0.5, 40)
    p = paired_difference(a, b)
    w = welch_difference(a, b)
    assert p.significant and p.ci_low < 1.0 < p.ci_high
    assert (p.ci_high - p.ci_low) < (w.ci_high - w.ci_low) / 5
    assert p.method == "paired-t" and w.method == "welch-t"
    with pytest.raises(ValueError):
        paired_difference([1, 2], [1, 2, 3])


def test_identical_samples_are_not_significant() -> None:
    d = paired_difference([1, 2, 3], [1, 2, 3])
    assert d.difference == 0 and not d.significant and d.p_value == 1.0
