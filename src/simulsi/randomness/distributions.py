"""Probability distributions.

Distributions are small immutable objects that know how to draw from a
:class:`~simulsi.randomness.stream.RandomStream`. They validate their
parameters eagerly, expose analytic ``mean``/``variance`` where they exist,
and round-trip through plain dictionaries (``to_spec`` / :func:`from_spec`)
so they can appear in YAML configuration without any code execution.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from simulsi.errors import ConfigError
from simulsi.randomness.stream import RandomStream

T = TypeVar("T")

_REGISTRY: dict[str, type[Distribution[Any]]] = {}


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _finite(name: str, value: float) -> None:
    _check(math.isfinite(value), f"{name} must be finite, got {value}")


class Distribution(ABC, Generic[T]):
    """Base class. Subclasses implement :meth:`sample`."""

    kind: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.kind:
            _REGISTRY[cls.kind] = cls

    @abstractmethod
    def sample(self, stream: RandomStream) -> T: ...

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        """Draw ``n`` values. Subclasses override with vectorised NumPy calls."""
        return np.asarray([self.sample(stream) for _ in range(n)])

    @property
    def mean(self) -> float | None:
        return None

    @property
    def variance(self) -> float | None:
        return None

    def to_spec(self) -> dict[str, Any]:
        """Serialisable description, accepted by :func:`from_spec`."""
        if not self.kind:
            raise TypeError(f"{type(self).__name__} cannot be serialised")
        spec: dict[str, Any] = {"distribution": self.kind}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            spec[f.name] = list(value) if isinstance(value, tuple) else value
        return spec


@dataclass(frozen=True)
class Constant(Distribution[float]):
    value: float
    kind: ClassVar[str] = "constant"

    def sample(self, stream: RandomStream) -> float:
        return self.value

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return np.full(n, self.value, dtype=float)

    @property
    def mean(self) -> float:
        return self.value

    @property
    def variance(self) -> float:
        return 0.0


@dataclass(frozen=True)
class Uniform(Distribution[float]):
    low: float = 0.0
    high: float = 1.0
    kind: ClassVar[str] = "uniform"

    def __post_init__(self) -> None:
        _finite("low", self.low)
        _finite("high", self.high)
        _check(self.low <= self.high, "uniform requires low <= high")

    def sample(self, stream: RandomStream) -> float:
        return float(stream.generator.uniform(self.low, self.high))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.uniform(self.low, self.high, n)

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2

    @property
    def variance(self) -> float:
        return (self.high - self.low) ** 2 / 12


@dataclass(frozen=True)
class Normal(Distribution[float]):
    mean_: float = field(default=0.0, metadata={"alias": "mean"})
    std: float = 1.0
    kind: ClassVar[str] = "normal"

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        object.__setattr__(self, "mean_", float(mean))
        object.__setattr__(self, "std", float(std))
        _finite("mean", self.mean_)
        _finite("std", self.std)
        _check(self.std >= 0, "normal std must be >= 0")

    def sample(self, stream: RandomStream) -> float:
        return float(stream.generator.normal(self.mean_, self.std))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.normal(self.mean_, self.std, n)

    @property
    def mean(self) -> float:
        return self.mean_

    @property
    def variance(self) -> float:
        return self.std**2

    def to_spec(self) -> dict[str, Any]:
        return {"distribution": self.kind, "mean": self.mean_, "std": self.std}


@dataclass(frozen=True)
class Exponential(Distribution[float]):
    """Exponential distribution. Give exactly one of ``rate`` or ``mean``."""

    rate: float
    kind: ClassVar[str] = "exponential"

    def __init__(self, rate: float | None = None, *, mean: float | None = None) -> None:
        if (rate is None) == (mean is None):
            raise ValueError("exponential needs exactly one of rate= or mean=")
        r = float(rate) if rate is not None else 1.0 / float(mean)  # type: ignore[arg-type]
        _check(math.isfinite(r) and r > 0, "exponential rate must be finite and > 0")
        object.__setattr__(self, "rate", r)

    def sample(self, stream: RandomStream) -> float:
        return float(stream.generator.exponential(1.0 / self.rate))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.exponential(1.0 / self.rate, n)

    @property
    def mean(self) -> float:
        return 1.0 / self.rate

    @property
    def variance(self) -> float:
        return 1.0 / self.rate**2


@dataclass(frozen=True)
class Poisson(Distribution[int]):
    lam: float
    kind: ClassVar[str] = "poisson"

    def __post_init__(self) -> None:
        _check(math.isfinite(self.lam) and self.lam >= 0, "poisson lam must be >= 0")

    def sample(self, stream: RandomStream) -> int:
        return int(stream.generator.poisson(self.lam))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.poisson(self.lam, n)

    @property
    def mean(self) -> float:
        return self.lam

    @property
    def variance(self) -> float:
        return self.lam


@dataclass(frozen=True)
class Binomial(Distribution[int]):
    n: int
    p: float
    kind: ClassVar[str] = "binomial"

    def __post_init__(self) -> None:
        _check(int(self.n) == self.n and self.n >= 0, "binomial n must be a non-negative integer")
        _check(0.0 <= self.p <= 1.0, f"binomial p must be in [0, 1], got {self.p}")

    def sample(self, stream: RandomStream) -> int:
        return int(stream.generator.binomial(self.n, self.p))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.binomial(self.n, self.p, n)

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)


@dataclass(frozen=True)
class Gamma(Distribution[float]):
    shape: float
    scale: float = 1.0
    kind: ClassVar[str] = "gamma"

    def __post_init__(self) -> None:
        _check(math.isfinite(self.shape) and self.shape > 0, "gamma shape must be > 0")
        _check(math.isfinite(self.scale) and self.scale > 0, "gamma scale must be > 0")

    def sample(self, stream: RandomStream) -> float:
        return float(stream.generator.gamma(self.shape, self.scale))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.gamma(self.shape, self.scale, n)

    @property
    def mean(self) -> float:
        return self.shape * self.scale

    @property
    def variance(self) -> float:
        return self.shape * self.scale**2


@dataclass(frozen=True)
class LogNormal(Distribution[float]):
    """Log-normal; ``mu``/``sigma`` parameterise the underlying normal distribution.

    Use :meth:`from_moments` to specify the mean and standard deviation of the
    log-normal itself, which is usually what data gives you.
    """

    mu: float = 0.0
    sigma: float = 1.0
    kind: ClassVar[str] = "lognormal"

    def __post_init__(self) -> None:
        _finite("mu", self.mu)
        _check(math.isfinite(self.sigma) and self.sigma >= 0, "lognormal sigma must be >= 0")

    @classmethod
    def from_moments(cls, mean: float, std: float) -> LogNormal:
        _check(mean > 0 and std >= 0, "lognormal needs mean > 0 and std >= 0")
        sigma2 = math.log1p((std / mean) ** 2)
        return cls(mu=math.log(mean) - sigma2 / 2, sigma=math.sqrt(sigma2))

    def sample(self, stream: RandomStream) -> float:
        return float(stream.generator.lognormal(self.mu, self.sigma))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.lognormal(self.mu, self.sigma, n)

    @property
    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)

    @property
    def variance(self) -> float:
        return (math.exp(self.sigma**2) - 1) * math.exp(2 * self.mu + self.sigma**2)


@dataclass(frozen=True)
class Triangular(Distribution[float]):
    low: float
    mode: float
    high: float
    kind: ClassVar[str] = "triangular"

    def __post_init__(self) -> None:
        for name in ("low", "mode", "high"):
            _finite(name, getattr(self, name))
        _check(self.low <= self.mode <= self.high, "triangular requires low <= mode <= high")
        _check(self.low < self.high, "triangular requires low < high")

    def sample(self, stream: RandomStream) -> float:
        return float(stream.generator.triangular(self.low, self.mode, self.high))

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        return stream.generator.triangular(self.low, self.mode, self.high, n)

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    @property
    def variance(self) -> float:
        a, c, b = self.low, self.mode, self.high
        return (a * a + b * b + c * c - a * b - a * c - b * c) / 18


@dataclass(frozen=True)
class Empirical(Distribution[float]):
    """Resamples observed data (with replacement), optionally weighted."""

    values: tuple[float, ...]
    weights: tuple[float, ...] | None = None
    kind: ClassVar[str] = "empirical"

    def __init__(self, values: Sequence[float], weights: Sequence[float] | None = None) -> None:
        vals = tuple(float(v) for v in values)
        _check(len(vals) > 0, "empirical distribution needs at least one value")
        _check(all(math.isfinite(v) for v in vals), "empirical values must be finite")
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "weights", None)
        if weights is not None:
            w = tuple(float(x) for x in weights)
            _check(len(w) == len(vals), "empirical weights must match values")
            _check(all(x >= 0 for x in w) and sum(w) > 0, "empirical weights must be >= 0")
            total = sum(w)
            object.__setattr__(self, "weights", tuple(x / total for x in w))

    def sample(self, stream: RandomStream) -> float:
        return self.values[int(stream.generator.choice(len(self.values), p=self.weights))]

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        idx = stream.generator.choice(len(self.values), size=n, p=self.weights)
        return np.asarray(self.values)[idx]

    @property
    def mean(self) -> float:
        return float(np.average(self.values, weights=self.weights))

    @property
    def variance(self) -> float:
        m = self.mean
        return float(np.average((np.asarray(self.values) - m) ** 2, weights=self.weights))


@dataclass(frozen=True)
class Categorical(Distribution[Any]):
    """Draws one of ``categories`` with the given probabilities."""

    categories: tuple[Any, ...]
    probabilities: tuple[float, ...]
    kind: ClassVar[str] = "categorical"

    def __init__(
        self,
        categories: Sequence[Any] | Mapping[Any, float],
        probabilities: Sequence[float] | None = None,
    ) -> None:
        if isinstance(categories, Mapping):
            _check(
                probabilities is None, "pass probabilities via the mapping or the list, not both"
            )
            cats = tuple(categories.keys())
            probs = tuple(float(p) for p in categories.values())
        else:
            cats = tuple(categories)
            if probabilities is None:
                probs = tuple(1.0 / len(cats) for _ in cats) if cats else ()
            else:
                probs = tuple(float(p) for p in probabilities)
        _check(len(cats) > 0, "categorical needs at least one category")
        _check(len(cats) == len(probs), "categories and probabilities must have equal length")
        _check(all(0.0 <= p <= 1.0 for p in probs), "probabilities must be in [0, 1]")
        _check(abs(sum(probs) - 1.0) < 1e-9, f"probabilities must sum to 1, got {sum(probs)}")
        object.__setattr__(self, "categories", cats)
        object.__setattr__(self, "probabilities", probs)

    def sample(self, stream: RandomStream) -> Any:
        return self.categories[
            int(stream.generator.choice(len(self.categories), p=self.probabilities))
        ]

    def sample_n(self, stream: RandomStream, n: int) -> npt.NDArray[Any]:
        idx = stream.generator.choice(len(self.categories), size=n, p=self.probabilities)
        return np.asarray(self.categories, dtype=object)[idx]


@dataclass(frozen=True)
class Custom(Distribution[T]):
    """Wraps any function ``stream -> value``. Not serialisable to configuration."""

    fn: Callable[[RandomStream], T]
    name: str = "custom"

    def sample(self, stream: RandomStream) -> T:
        return self.fn(stream)


DistributionLike = Distribution[Any] | float | int


def as_distribution(value: DistributionLike | Mapping[str, Any]) -> Distribution[Any]:
    """Coerce numbers to :class:`Constant` and spec dicts via :func:`from_spec`."""
    if isinstance(value, Distribution):
        return value
    if isinstance(value, Mapping):
        return from_spec(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"cannot interpret {value!r} as a distribution")
    return Constant(float(value))


def from_spec(spec: Mapping[str, Any]) -> Distribution[Any]:
    """Build a distribution from ``{"distribution": "<kind>", **params}``.

    Only registered distribution classes can be created, so configuration
    files can never trigger arbitrary code.
    """
    if "distribution" not in spec:
        raise ConfigError(f"distribution spec needs a 'distribution' key: {dict(spec)!r}")
    kind = str(spec["distribution"]).lower()
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ConfigError(f"unknown distribution {kind!r}; known: {', '.join(sorted(_REGISTRY))}")
    params = {k: v for k, v in spec.items() if k != "distribution"}
    try:
        return cls(**params)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid parameters for {kind} distribution: {exc}") from exc


def is_distribution_spec(value: Any) -> bool:
    return isinstance(value, Mapping) and "distribution" in value


def registered_distributions() -> list[str]:
    return sorted(_REGISTRY)
