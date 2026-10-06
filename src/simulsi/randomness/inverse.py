"""Inverse-transform sampling, the basis for antithetic variates.

NumPy's fast samplers (ziggurat, rejection) consume a variable number of
uniforms per variate, so a run fed ``1 - U`` instead of ``U`` would not be the
mirror image of the original run. :class:`InverseTransformGenerator` exposes
the subset of :class:`numpy.random.Generator` that SimulSI uses, but produces
every variate from exactly one uniform by inverting its CDF. With
``flip=True`` it uses ``1 - U``: two runs with the same seed, one flipped, form
an *antithetic pair* whose outputs are negatively correlated when the model
responds monotonically to its inputs.

Inverse sampling is slower than NumPy's native samplers (noticeably so for
Poisson and binomial, which use SciPy's discrete quantile functions).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import special as _sp
from scipy import stats as _st


def _out(x: npt.NDArray[Any], size: Any) -> Any:
    return x if size is not None else x.item()


class InverseTransformGenerator:
    """A drop-in for the parts of ``numpy.random.Generator`` SimulSI uses."""

    __slots__ = ("base", "flip")

    def __init__(self, base: np.random.Generator, *, flip: bool = False) -> None:
        self.base = base
        self.flip = flip

    def _u(self, size: Any = None) -> npt.NDArray[np.float64]:
        u = np.asarray(self.base.random(size), dtype=float)
        u = 1.0 - u if self.flip else u
        # keep strictly inside (0, 1) so quantile functions stay finite
        return np.clip(u, 1e-16, 1.0 - 1e-16)

    def random(self, size: Any = None) -> Any:
        return _out(self._u(size), size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any = None) -> Any:
        return _out(low + (high - low) * self._u(size), size)

    def integers(self, low: int, high: int | None = None, size: Any = None) -> Any:
        if high is None:
            low, high = 0, low
        x = np.floor(low + (high - low) * self._u(size)).astype(np.int64)
        return _out(np.minimum(x, high - 1), size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any = None) -> Any:
        return _out(loc + scale * _sp.ndtri(self._u(size)), size)

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: Any = None) -> Any:
        return _out(np.exp(mean + sigma * _sp.ndtri(self._u(size))), size)

    def exponential(self, scale: float = 1.0, size: Any = None) -> Any:
        return _out(-scale * np.log1p(-self._u(size)), size)

    def gamma(self, shape: float, scale: float = 1.0, size: Any = None) -> Any:
        return _out(scale * _sp.gammaincinv(shape, self._u(size)), size)

    def triangular(self, left: float, mode: float, right: float, size: Any = None) -> Any:
        u = self._u(size)
        c = (mode - left) / (right - left)
        lower = left + np.sqrt(u * (right - left) * (mode - left))
        upper = right - np.sqrt((1 - u) * (right - left) * (right - mode))
        return _out(np.where(u < c, lower, upper), size)

    def poisson(self, lam: float, size: Any = None) -> Any:
        return _out(_st.poisson.ppf(self._u(size), lam).astype(np.int64), size)

    def binomial(self, n: int, p: float, size: Any = None) -> Any:
        return _out(_st.binom.ppf(self._u(size), n, p).astype(np.int64), size)

    def choice(self, a: int | Sequence[Any], size: Any = None, p: Any = None) -> Any:
        n = a if isinstance(a, int) else len(a)
        probs = np.full(n, 1.0 / n) if p is None else np.asarray(p, dtype=float)
        cum = np.cumsum(probs)
        cum[-1] = 1.0
        idx = np.minimum(np.searchsorted(cum, self._u(size), side="right"), n - 1)
        if isinstance(a, int):
            return _out(np.asarray(idx), size)
        values = np.asarray(a, dtype=object)[idx]
        return values if size is not None else values.item() if hasattr(values, "item") else values

    def permutation(self, n: int) -> npt.NDArray[np.int64]:
        return np.argsort(self._u(n), kind="stable").astype(np.int64)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"{name!r} is not available with inverse-transform (antithetic) sampling"
        )
