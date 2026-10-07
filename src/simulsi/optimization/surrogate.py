"""Gaussian-process surrogates (metamodels) of simulation outputs.

A surrogate is a cheap statistical approximation of an expensive model,
fitted to a few dozen simulation runs. Use it to explore a response surface,
to find promising regions before confirming them with full experiments, or
inside :func:`~simulsi.optimization.optimize` (Bayesian optimisation).

>>> import numpy as np
>>> X = np.linspace(0, 1, 12)[:, None]
>>> gp = GaussianProcess(seed=0).fit(X, np.sin(6 * X[:, 0]))
>>> round(float(gp.predict(np.array([[0.5]]))[0]), 2)
0.14

The GP uses a squared-exponential kernel with one length scale per input
(automatic relevance determination) and a fitted noise term, so it smooths
simulation noise instead of interpolating it. Always check
``loo_r2()`` (leave-one-out accuracy) before trusting predictions, and
never extrapolate outside the sampled ranges.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from simulsi.core.model import Model
from simulsi.randomness.stream import derive_seed

Array = npt.NDArray[np.float64]


class GaussianProcess:
    """GP regression with an ARD squared-exponential kernel and Gaussian noise.

    Inputs are rescaled to [0, 1] and outputs standardised internally;
    hyperparameters maximise the log marginal likelihood (L-BFGS-B with
    ``restarts`` random starts).
    """

    def __init__(self, *, noise: bool = True, restarts: int = 4, seed: int = 0) -> None:
        self.noise = noise
        self.restarts = restarts
        self.seed = seed
        self._fitted = False

    # -- fitting ---------------------------------------------------------------

    def _kernel(self, a: Array, b: Array, ls: Array, var: float) -> Array:
        d = (a[:, None, :] - b[None, :, :]) / ls
        out: Array = var * np.exp(-0.5 * np.sum(d * d, axis=2))
        return out

    def _nll(self, theta: Array) -> float:
        d = self._X.shape[1]
        ls, var = np.exp(theta[:d]), math.exp(theta[d])
        noise = math.exp(theta[d + 1]) if self.noise else 0.0
        K = self._kernel(self._X, self._X, ls, var)
        K[np.diag_indices_from(K)] += noise + 1e-8
        try:
            L = linalg.cholesky(K, lower=True)
        except linalg.LinAlgError:
            return 1e25
        alpha = linalg.cho_solve((L, True), self._y)
        n = len(self._y)
        return float(
            0.5 * self._y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * math.log(2 * math.pi)
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> GaussianProcess:
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float).ravel()
        if Xa.shape[0] != len(ya):
            raise ValueError("X and y have different numbers of rows")
        if len(ya) < 3:
            raise ValueError("need at least 3 observations")
        if not (np.all(np.isfinite(Xa)) and np.all(np.isfinite(ya))):
            raise ValueError("X and y must be finite")
        self._lo = Xa.min(axis=0)
        span = Xa.max(axis=0) - self._lo
        self._span = np.where(span > 0, span, 1.0)
        self._ymean = float(ya.mean())
        self._ystd = float(ya.std()) or 1.0
        self._X = (Xa - self._lo) / self._span
        self._y = (ya - self._ymean) / self._ystd
        d = Xa.shape[1]
        bounds = [(math.log(0.01), math.log(10.0))] * d + [(math.log(0.05), math.log(20.0))]
        if self.noise:
            bounds.append((math.log(1e-6), math.log(1.0)))
        rng = np.random.default_rng(self.seed)
        best: Any = None
        starts = [np.array([math.log(0.3)] * d + [0.0] + ([math.log(0.1)] if self.noise else []))]
        starts += [
            np.array([rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(self.restarts)
        ]
        for x0 in starts:
            res = optimize.minimize(self._nll, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x
        self.length_scales = np.exp(theta[:d]) * self._span
        self._ls = np.exp(theta[:d])
        self._var = math.exp(theta[d])
        self._noise = math.exp(theta[d + 1]) if self.noise else 0.0
        K = self._kernel(self._X, self._X, self._ls, self._var)
        K[np.diag_indices_from(K)] += self._noise + 1e-8
        self._L = linalg.cholesky(K, lower=True)
        self._alpha = linalg.cho_solve((self._L, True), self._y)
        self.log_marginal_likelihood = -float(best.fun)
        self._fitted = True
        return self

    # -- prediction --------------------------------------------------------------

    def predict(self, X: npt.ArrayLike, *, return_std: bool = False) -> Any:
        """Mean prediction (and the standard deviation of the *mean* if ``return_std``)."""
        if not self._fitted:
            raise RuntimeError("fit the surrogate first")
        Xs = (np.atleast_2d(np.asarray(X, dtype=float)) - self._lo) / self._span
        Ks = self._kernel(Xs, self._X, self._ls, self._var)
        mean = Ks @ self._alpha * self._ystd + self._ymean
        if not return_std:
            return mean
        v = linalg.solve_triangular(self._L, Ks.T, lower=True)
        var = np.maximum(self._var - np.sum(v * v, axis=0), 0.0)
        return mean, np.sqrt(var) * self._ystd

    def loo_residuals(self) -> Array:
        """Leave-one-out prediction errors, in closed form (no refitting)."""
        if not self._fitted:
            raise RuntimeError("fit the surrogate first")
        Kinv = linalg.cho_solve((self._L, True), np.eye(len(self._y)))
        res: Array = self._alpha / np.diag(Kinv) * self._ystd
        return res

    def loo_r2(self) -> float:
        """Leave-one-out R^2: share of output variance the surrogate predicts out of sample."""
        e = self.loo_residuals()
        y = self._y * self._ystd
        return float(1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2))

    @property
    def noise_std(self) -> float:
        """Estimated noise standard deviation, in output units."""
        return math.sqrt(self._noise) * self._ystd


def latin_hypercube(n: int, d: int, seed: int) -> Array:
    """``n`` points in [0, 1]^d, one per stratum in every dimension."""
    rng = np.random.default_rng(seed)
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def _scale(
    unit: Array, ranges: Mapping[str, tuple[float, float]], ints: Sequence[bool]
) -> list[dict[str, Any]]:
    names = list(ranges)
    out = []
    for row in unit:
        p: dict[str, Any] = {}
        for j, k in enumerate(names):
            lo, hi = ranges[k]
            v = lo + row[j] * (hi - lo)
            p[k] = round(v) if ints[j] else float(v)
        out.append(p)
    return out


def _int_params(model: Model, names: Sequence[str]) -> list[bool]:
    return [k in model.parameters and model.parameters[k].kind == "int" for k in names]


@dataclass
class Surrogate:
    """GP surrogates of one or more simulation outputs over a parameter box."""

    parameters: list[str]
    ranges: dict[str, tuple[float, float]]
    outputs: dict[str, GaussianProcess]
    design: list[dict[str, Any]]
    data: dict[str, Array] = field(repr=False)

    def _matrix(self, points: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> Array:
        rows = [points] if isinstance(points, Mapping) else list(points)
        return np.array([[float(r[k]) for k in self.parameters] for r in rows])

    def predict(
        self,
        points: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        output: str | None = None,
        *,
        return_std: bool = False,
    ) -> Any:
        """Predict ``output`` at one parameter mapping or a list of them."""
        name = output or next(iter(self.outputs))
        if name not in self.outputs:
            raise KeyError(f"no surrogate for {name!r}; have {list(self.outputs)}")
        X = self._matrix(points)
        for j, k in enumerate(self.parameters):
            lo, hi = self.ranges[k]
            if np.any(X[:, j] < lo) or np.any(X[:, j] > hi):
                raise ValueError(f"{k} outside the fitted range [{lo}, {hi}]: no extrapolation")
        return self.outputs[name].predict(X, return_std=return_std)

    def accuracy(self) -> dict[str, float]:
        """Leave-one-out R^2 per output (1 = perfect; below ~0.8, add runs)."""
        return {k: gp.loo_r2() for k, gp in self.outputs.items()}


def fit_surrogate(
    model: Model | Callable[..., Any],
    ranges: Mapping[str, tuple[float, float]],
    outputs: Sequence[str],
    n: int = 40,
    *,
    replications: int = 1,
    seed: int = 0,
    fixed: Mapping[str, Any] | None = None,
) -> Surrogate:
    """Run a Latin hypercube design of ``n`` points and fit a GP per output.

    Each point averages ``replications`` runs; all points share the same
    seeds (common random numbers), which makes the surface smoother.
    """
    names = list(ranges)
    if not names:
        raise ValueError("need at least one parameter range")
    for k, (lo, hi) in ranges.items():
        if not hi > lo:
            raise ValueError(f"range for {k!r} needs low < high")
    ints = _int_params(model, names) if isinstance(model, Model) else [False] * len(names)
    design = _scale(latin_hypercube(n, len(names), derive_seed(seed, "surrogate")), ranges, ints)
    values: dict[str, list[float]] = {o: [] for o in outputs}
    for p in design:
        sums = dict.fromkeys(outputs, 0.0)
        for r in range(replications):
            if isinstance(model, Model):
                m: Mapping[str, float] = model.simulate(
                    {**(fixed or {}), **p}, seed=derive_seed(seed, "replication", r)
                ).metrics
            else:
                res = model(**(fixed or {}), **p)
                m = res if isinstance(res, Mapping) else {"value": float(res)}
            for o in outputs:
                if o not in m:
                    raise KeyError(f"model produced no output {o!r}")
                sums[o] += float(m[o])
        for o in outputs:
            values[o].append(sums[o] / replications)
    X = np.array([[float(p[k]) for k in names] for p in design])
    gps = {o: GaussianProcess(seed=seed).fit(X, values[o]) for o in outputs}
    return Surrogate(
        parameters=names,
        ranges={k: (float(lo), float(hi)) for k, (lo, hi) in ranges.items()},
        outputs=gps,
        design=design,
        data={o: np.asarray(v) for o, v in values.items()},
    )
