"""How much of the uncertainty comes from the input data?

Simulation confidence intervals usually cover *simulation noise* only. But
input distributions are fitted to finite data, and a different sample of
data would give different parameters. :func:`input_uncertainty` estimates
that extra uncertainty by bootstrapping: resample each data set, refit its
distribution, run the model, repeat. The spread of the results across
bootstrap fits, after removing simulation noise, is the input uncertainty
(Barton and Schruben 2001; Barton 2012).

>>> # interarrival times observed at a desk, used to set the arrival rate:
>>> # input_uncertainty(mmc, {"arrival_rate": InputData(gaps, "exponential", lambda d: 1 / d.mean)},
>>> #                   "resource.server.wait.mean")
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from simulsi.analysis.report import format_table, html_table
from simulsi.core.model import Model
from simulsi.randomness.distributions import Distribution
from simulsi.randomness.fitting import fit_distribution
from simulsi.randomness.stream import derive_seed


@dataclass
class InputData:
    """Observed data for one model parameter.

    ``family`` is the distribution family to fit (``"exponential"``,
    ``"gamma"``, ...). ``transform`` turns the fitted distribution into the
    parameter value (for example ``lambda d: 1 / d.mean`` for a rate); by
    default the distribution itself is passed, for ``distribution`` parameters.
    """

    values: Sequence[float]
    family: str = "exponential"
    transform: Callable[[Distribution[Any]], Any] | None = None

    def fit(self, values: Sequence[float] | np.ndarray[Any, Any]) -> Any:
        report = fit_distribution(values, [self.family])
        dist = report.best.distribution
        return self.transform(dist) if self.transform is not None else dist


@dataclass
class InputUncertaintyResult:
    metric: str
    point_estimate: float
    simulation_half_width: float
    ci_low: float
    ci_high: float
    input_variance: float
    simulation_variance: float
    bootstrap_means: list[float]
    replications: int
    confidence: float

    @property
    def input_share(self) -> float:
        """Share of the estimator's variance caused by finite input data."""
        total = self.input_variance + self.simulation_variance
        return self.input_variance / total if total > 0 else math.nan

    def _row(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "estimate": self.point_estimate,
            "sim_only_ci": f"±{self.simulation_half_width:.3g}",
            "ci_with_input": f"[{self.ci_low:.4g}, {self.ci_high:.4g}]",
            "input_share": self.input_share,
        }

    def format(self) -> str:
        head = (
            f"{len(self.bootstrap_means)} bootstrap refits x {self.replications} replications; "
            f"{self.confidence:.0%} intervals"
        )
        note = ""
        if self.input_share > 0.5:
            note = (
                "\nnote: most of the uncertainty comes from the input data - "
                "collecting more data helps more than more replications"
            )
        return head + "\n" + format_table([self._row()]) + note

    def _repr_html_(self) -> str:
        return html_table([self._row()], caption="Input uncertainty")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._row(),
            "input_variance": self.input_variance,
            "simulation_variance": self.simulation_variance,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "bootstrap_means": list(self.bootstrap_means),
        }


def input_uncertainty(
    model: Model,
    inputs: Mapping[str, InputData],
    metric: str,
    *,
    n_bootstrap: int = 30,
    replications: int = 5,
    seed: int = 0,
    fixed: Mapping[str, Any] | None = None,
    confidence: float = 0.95,
) -> InputUncertaintyResult:
    """Bootstrap the input data to see how much it moves ``metric``.

    For each of ``n_bootstrap`` rounds every data set in ``inputs`` is
    resampled with replacement and refitted; the model then runs
    ``replications`` times with common seeds. The total variance of the
    round means is split into simulation noise (mean within-round variance
    / replications) and input variance (the rest). The returned interval is
    the bootstrap percentile interval, which includes both.
    """
    if n_bootstrap < 5:
        raise ValueError("n_bootstrap must be >= 5")
    if replications < 2:
        raise ValueError("replications must be >= 2 to estimate simulation noise")
    if not inputs:
        raise ValueError("give at least one input data set")
    rng = np.random.default_rng(derive_seed(seed, "input-bootstrap"))
    base = dict(fixed or {})
    data = {k: np.asarray(v.values, dtype=float) for k, v in inputs.items()}

    def run(params: Mapping[str, Any]) -> np.ndarray[Any, Any]:
        vals = []
        for r in range(replications):
            m = model.simulate({**base, **params}, seed=derive_seed(seed, "replication", r)).metrics
            if metric not in m:
                raise KeyError(f"model produced no metric {metric!r}")
            vals.append(m[metric])
        return np.asarray(vals, dtype=float)

    original = run({k: inputs[k].fit(data[k]) for k in inputs})
    means, variances = [], []
    for _ in range(n_bootstrap):
        params = {
            k: inputs[k].fit(rng.choice(data[k], size=len(data[k]), replace=True)) for k in inputs
        }
        y = run(params)
        means.append(float(np.mean(y)))
        variances.append(float(np.var(y, ddof=1)))
    total = float(np.var(means, ddof=1))
    noise = float(np.mean(variances)) / replications
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    from simulsi.statistics.core import t_half_width

    return InputUncertaintyResult(
        metric=metric,
        point_estimate=float(np.mean(original)),
        simulation_half_width=float(t_half_width(original, confidence)),
        ci_low=float(lo),
        ci_high=float(hi),
        input_variance=max(0.0, total - noise),
        simulation_variance=noise,
        bootstrap_means=means,
        replications=replications,
        confidence=confidence,
    )
