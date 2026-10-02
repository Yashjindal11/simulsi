"""Seeded random number streams.

SimulSI never touches global random state. Every simulation owns a root
:class:`RandomStream` built from a seed, and named sub-streams are derived
from ``(seed, name)`` with NumPy's ``SeedSequence``. A sub-stream therefore
produces the same numbers no matter how many other streams exist or in which
order they were created, which keeps *common random numbers* intact when a
model changes (e.g. adding a resource does not shift the arrival stream).
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from simulsi.randomness.distributions import Distribution

T = TypeVar("T")

_SEED_BITS = 63


def _name_key(name: str) -> int:
    # crc32 is stable across processes and Python versions (unlike hash()).
    return zlib.crc32(name.encode("utf-8"))


def derive_seed(seed: int, *keys: int | str) -> int:
    """Deterministically derive an independent 63-bit seed from ``seed`` and ``keys``."""
    spawn_key = tuple(_name_key(k) if isinstance(k, str) else int(k) for k in keys)
    state = np.random.SeedSequence(entropy=seed, spawn_key=spawn_key).generate_state(
        2, dtype=np.uint32
    )
    return (int(state[0]) << 32 | int(state[1])) & ((1 << _SEED_BITS) - 1)


def fresh_seed() -> int:
    """A new seed from OS entropy, for runs where the user did not supply one."""
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0] >> 1)


class RandomStream:
    """A reproducible stream of random numbers (PCG64 under the hood)."""

    __slots__ = ("_children", "generator", "name", "seed")

    def __init__(self, seed: int | None = None, *, name: str = "root") -> None:
        self.seed = fresh_seed() if seed is None else int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.name = name
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self._children: dict[str, RandomStream] = {}

    def stream(self, name: str) -> RandomStream:
        """Return the named sub-stream (created on first use, then cached)."""
        child = self._children.get(name)
        if child is None:
            child = RandomStream(derive_seed(self.seed, name), name=f"{self.name}/{name}")
            self._children[name] = child
        return child

    # -- primitives --------------------------------------------------------

    def random(self) -> float:
        return float(self.generator.random())

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self.generator.uniform(low, high))

    def integers(self, low: int, high: int) -> int:
        """Uniform integer in ``[low, high)``."""
        return int(self.generator.integers(low, high))

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        return float(self.generator.normal(mean, std))

    def exponential(self, mean: float) -> float:
        """Exponential variate with the given *mean* (= 1 / rate)."""
        return float(self.generator.exponential(mean))

    def poisson(self, lam: float) -> int:
        return int(self.generator.poisson(lam))

    def binomial(self, n: int, p: float) -> int:
        return int(self.generator.binomial(n, p))

    def gamma(self, shape: float, scale: float = 1.0) -> float:
        return float(self.generator.gamma(shape, scale))

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        """Log-normal variate; ``mean``/``sigma`` are those of the underlying normal."""
        return float(self.generator.lognormal(mean, sigma))

    def triangular(self, low: float, mode: float, high: float) -> float:
        return float(self.generator.triangular(low, mode, high))

    def bernoulli(self, p: float) -> bool:
        return bool(self.generator.random() < p)

    def choice(self, options: Sequence[T], p: Sequence[float] | None = None) -> T:
        idx = int(self.generator.choice(len(options), p=None if p is None else np.asarray(p)))
        return options[idx]

    def shuffle(self, items: list[Any]) -> None:
        order = self.generator.permutation(len(items))
        items[:] = [items[i] for i in order]

    def sample(self, distribution: Distribution[T]) -> T:
        return distribution.sample(self)

    def __repr__(self) -> str:
        return f"RandomStream(name={self.name!r}, seed={self.seed})"
