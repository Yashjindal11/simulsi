"""Fit input distributions to data.

:func:`fit_distribution` fits a set of candidate families by maximum
likelihood (closed form where one exists), ranks them by AIC, BIC or the
Kolmogorov-Smirnov statistic, and returns ready-to-use SimulSI
distributions:

>>> import numpy as np
>>> data = np.random.default_rng(1).gamma(2.0, 3.0, size=500)
>>> report = fit_distribution(data)
>>> report.best.name
'gamma'
>>> service = report.best.distribution     # a simulsi Gamma, use it in a model

Fitting is no substitute for looking at the data: check the histogram
(``report.plot()`` needs matplotlib), and consider
:class:`~simulsi.randomness.Empirical` when no family fits well.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from simulsi.randomness.distributions import (
    Distribution,
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    Poisson,
    Triangular,
    Uniform,
)

Criterion = Literal["aic", "bic", "ks"]

CANDIDATES = ("exponential", "gamma", "lognormal", "normal", "uniform", "triangular", "poisson")


@dataclass
class FitResult:
    """One fitted family. Lower ``aic``/``bic``/``ks_statistic`` is better."""

    name: str
    distribution: Distribution[Any]
    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "spec": self.distribution.to_spec(),
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
        }


@dataclass
class FitReport:
    """Ranked fits plus the families that were skipped and why."""

    n: int
    criterion: Criterion
    results: list[FitResult]
    skipped: dict[str, str] = field(default_factory=dict)
    data: npt.NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def best(self) -> FitResult:
        if not self.results:
            raise ValueError("no candidate distribution could be fitted")
        return self.results[0]

    def get(self, name: str) -> FitResult:
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(name)

    @property
    def notes(self) -> list[str]:
        out = []
        if self.n < 30:
            out.append(f"only {self.n} observations: rankings are unreliable")
        if self.results and self.results[0].ks_pvalue < 0.05:
            out.append(
                "the best fit is rejected by the KS test at 5%; consider an Empirical distribution"
            )
        out.append(
            "KS p-values use parameters estimated from the same data, so they are optimistic"
        )
        return out

    def format(self) -> str:
        head = f"{'distribution':<12} {'AIC':>12} {'BIC':>12} {'KS':>8} {'KS p':>8}  parameters"
        lines = [f"n = {self.n}, ranked by {self.criterion}", head, "-" * len(head)]
        for r in self.results:
            params = ", ".join(f"{k}={v:.4g}" for k, v in r.params.items())
            lines.append(
                f"{r.name:<12} {r.aic:>12.2f} {r.bic:>12.2f} {r.ks_statistic:>8.4f} "
                f"{r.ks_pvalue:>8.3f}  {params}"
            )
        for name, why in self.skipped.items():
            lines.append(f"{name:<12} skipped: {why}")
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "criterion": self.criterion,
            "best": self.best.to_dict() if self.results else None,
            "results": [r.to_dict() for r in self.results],
            "skipped": dict(self.skipped),
            "notes": self.notes,
        }

    def plot(self, top: int = 3, ax: Any = None) -> Any:
        """Histogram of the data with the ``top`` fitted densities (needs matplotlib)."""
        import matplotlib.pyplot as plt

        if self.data is None:
            raise ValueError("the report holds no data to plot")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        x = self.data
        ax.hist(x, bins="auto", density=True, alpha=0.4, color="grey", label="data")
        grid = np.linspace(np.min(x), np.max(x), 300)
        for r in self.results[:top]:
            frozen = _frozen(r.name, r.params)
            if r.name == "poisson":
                k = np.arange(math.floor(np.min(x)), math.ceil(np.max(x)) + 1)
                ax.plot(k, frozen.pmf(k), "o-", label=r.name)
            else:
                ax.plot(grid, frozen.pdf(grid), label=r.name)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend()
        return ax


def _frozen(name: str, p: dict[str, float]) -> Any:
    if name == "exponential":
        return stats.expon(scale=p["mean"])
    if name == "gamma":
        return stats.gamma(p["shape"], scale=p["scale"])
    if name == "lognormal":
        return stats.lognorm(p["sigma"], scale=math.exp(p["mu"]))
    if name == "normal":
        return stats.norm(p["mean"], p["std"])
    if name == "uniform":
        return stats.uniform(p["low"], p["high"] - p["low"])
    if name == "triangular":
        width = p["high"] - p["low"]
        return stats.triang((p["mode"] - p["low"]) / width, loc=p["low"], scale=width)
    if name == "poisson":
        return stats.poisson(p["lam"])
    raise KeyError(name)


def _fit_one(name: str, x: npt.NDArray[np.float64]) -> tuple[Distribution[Any], dict[str, float]]:
    n = len(x)
    lo, hi = float(np.min(x)), float(np.max(x))
    if name == "exponential":
        if lo < 0:
            raise ValueError("needs non-negative data")
        m = float(x.mean())
        return Exponential(mean=m), {"mean": m}
    if name == "gamma":
        if lo <= 0:
            raise ValueError("needs positive data")
        a, _, scale = stats.gamma.fit(x, floc=0)
        return Gamma(float(a), float(scale)), {"shape": float(a), "scale": float(scale)}
    if name == "lognormal":
        if lo <= 0:
            raise ValueError("needs positive data")
        logs = np.log(x)
        mu, sigma = float(logs.mean()), float(logs.std())
        return LogNormal(mu, sigma), {"mu": mu, "sigma": sigma}
    if name == "normal":
        mean, std = float(x.mean()), float(x.std())
        return Normal(mean, std), {"mean": mean, "std": std}
    if name == "uniform":
        # unbiased end points: the sample range underestimates the true one
        pad = (hi - lo) / (n - 1)
        return Uniform(lo - pad, hi + pad), {"low": lo - pad, "high": hi + pad}
    if name == "triangular":
        # widen slightly so the extreme observations have positive density
        pad = (hi - lo) / n
        low, high = lo - pad, hi + pad
        mode = min(max(3 * float(x.mean()) - low - high, low), high)
        return Triangular(low, mode, high), {"low": low, "mode": mode, "high": high}
    if name == "poisson":
        if lo < 0 or not np.all(x == np.round(x)):
            raise ValueError("needs non-negative integer data")
        lam = float(x.mean())
        return Poisson(lam), {"lam": lam}
    raise ValueError(f"unknown candidate {name!r}; choose from {CANDIDATES}")


_N_PARAMS = {
    "exponential": 1,
    "gamma": 2,
    "lognormal": 2,
    "normal": 2,
    "uniform": 2,
    "triangular": 3,
    "poisson": 1,
}

_ORDER: dict[str, Callable[[FitResult], float]] = {
    "aic": lambda r: r.aic,
    "bic": lambda r: r.bic,
    "ks": lambda r: r.ks_statistic,
}


def fit_distribution(
    data: Iterable[float],
    candidates: Sequence[str] = CANDIDATES,
    *,
    criterion: Criterion = "aic",
) -> FitReport:
    """Fit each candidate family to ``data`` and rank them by ``criterion``.

    Families whose support does not match the data (e.g. gamma with zeros,
    Poisson with fractional values) are listed in ``report.skipped``.
    """
    if criterion not in _ORDER:
        raise ValueError(f"criterion must be one of {tuple(_ORDER)}, got {criterion!r}")
    x = np.asarray(list(data), dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("need at least two observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("data contain NaN or infinite values")
    n = len(x)
    results: list[FitResult] = []
    skipped: dict[str, str] = {}
    for name in candidates:
        if name not in _N_PARAMS:
            raise ValueError(f"unknown candidate {name!r}; choose from {CANDIDATES}")
        if float(np.max(x)) == float(np.min(x)):
            skipped[name] = "all observations are equal (use Constant)"
            continue
        try:
            dist, params = _fit_one(name, x)
        except ValueError as e:
            skipped[name] = str(e)
            continue
        frozen = _frozen(name, params)
        discrete = name == "poisson"
        ll = float(np.sum(frozen.logpmf(x) if discrete else frozen.logpdf(x)))
        if not math.isfinite(ll):
            skipped[name] = "zero likelihood for some observations"
            continue
        k = _N_PARAMS[name]
        if discrete:
            # KS is not distribution-free for discrete data; report the
            # largest CDF gap and no p-value.
            values = np.sort(x)
            ecdf = np.arange(1, n + 1) / n
            ks_stat, ks_p = float(np.max(np.abs(ecdf - frozen.cdf(values)))), math.nan
        else:
            ks = stats.kstest(x, frozen.cdf)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        results.append(
            FitResult(
                name=name,
                distribution=dist,
                params=params,
                loglik=ll,
                aic=2 * k - 2 * ll,
                bic=k * math.log(n) - 2 * ll,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
            )
        )
    results.sort(key=_ORDER[criterion])
    return FitReport(n=n, criterion=criterion, results=results, skipped=skipped, data=x)


def load_values(path: str | Path, column: str | None = None) -> list[float]:
    """Read numbers from a CSV column (default: the first numeric one) or a plain text file.

    Plain text files may separate values by whitespace or commas. Blank cells
    are skipped.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        if not rows:
            raise ValueError(f"{p} has no data rows")
        if column is None:
            column = next((c for c in rows[0] if _is_number(rows[0][c])), None)
            if column is None:
                raise ValueError(f"{p}: no numeric column found; pass column=")
        if column not in rows[0]:
            raise ValueError(f"{p}: no column {column!r}; columns are {list(rows[0])}")
        cells = [r[column] for r in rows]
    else:
        if column is not None:
            raise ValueError("column= is only used for .csv files")
        cells = text.replace(",", " ").split()
    values = []
    for i, cell in enumerate(cells):
        if cell is None or not str(cell).strip():
            continue
        if not _is_number(cell):
            raise ValueError(f"{p}: value {i + 1} is not a number: {cell!r}")
        values.append(float(cell))
    return values


def _is_number(s: Any) -> bool:
    try:
        float(s)
    except (TypeError, ValueError):
        return False
    return True
