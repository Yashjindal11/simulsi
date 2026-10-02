from __future__ import annotations

import math

import numpy as np
import pytest

from simulsi import ConfigError, RandomStream, Simulation
from simulsi.randomness import (
    Binomial,
    Categorical,
    Constant,
    Custom,
    Empirical,
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    Poisson,
    Triangular,
    Uniform,
    as_distribution,
    derive_seed,
    from_spec,
    registered_distributions,
)


def test_same_seed_same_numbers() -> None:
    a, b = RandomStream(7), RandomStream(7)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_different_seed_different_numbers() -> None:
    assert RandomStream(1).random() != RandomStream(2).random()


def test_named_streams_do_not_depend_on_creation_order() -> None:
    a = RandomStream(11)
    a.stream("service")
    x = a.stream("arrivals").random()
    b = RandomStream(11)
    y = b.stream("arrivals").random()
    assert x == y
    assert a.stream("arrivals") is a.stream("arrivals")


def test_named_streams_are_distinct() -> None:
    root = RandomStream(5)
    assert root.stream("a").random() != root.stream("b").random()


def test_derive_seed_is_stable_and_bounded() -> None:
    s = derive_seed(42, "x", 3)
    assert s == derive_seed(42, "x", 3)
    assert 0 <= s < 2**63
    assert s != derive_seed(42, "x", 4)


def test_unseeded_simulation_records_its_seed() -> None:
    sim = Simulation()
    replay = Simulation(seed=sim.seed)
    assert sim.rng.random() == replay.rng.random()


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        RandomStream(-1)


DISTS = [
    Uniform(2, 6),
    Normal(10, 2),
    Exponential(rate=0.5),
    Exponential(mean=4.0),
    Poisson(3.0),
    Binomial(10, 0.3),
    Gamma(2.0, 1.5),
    LogNormal(0.5, 0.4),
    Triangular(1, 2, 6),
    Empirical([1, 2, 3, 10]),
    Empirical([1, 2], weights=[3, 1]),
    Constant(4.0),
]


@pytest.mark.parametrize("dist", DISTS, ids=lambda d: type(d).__name__)
def test_sample_mean_matches_analytic_mean(dist: object) -> None:
    from simulsi.randomness import Distribution

    assert isinstance(dist, Distribution)
    stream = RandomStream(123)
    xs = dist.sample_n(stream, 40_000).astype(float)
    assert dist.mean is not None and dist.variance is not None
    se = math.sqrt(dist.variance / len(xs)) if dist.variance > 0 else 0.0
    assert abs(xs.mean() - dist.mean) <= 5 * se + 1e-12
    assert abs(xs.var() - dist.variance) <= 0.1 * dist.variance + 1e-12


@pytest.mark.parametrize("dist", DISTS, ids=lambda d: type(d).__name__)
def test_spec_round_trip(dist: object) -> None:
    from simulsi.randomness import Distribution

    assert isinstance(dist, Distribution)
    again = from_spec(dist.to_spec())
    assert again == dist
    s1, s2 = RandomStream(9), RandomStream(9)
    assert dist.sample(s1) == again.sample(s2)


def test_lognormal_from_moments() -> None:
    d = LogNormal.from_moments(10.0, 3.0)
    assert d.mean == pytest.approx(10.0)
    assert math.sqrt(d.variance) == pytest.approx(3.0)


def test_categorical() -> None:
    d = Categorical({"a": 0.2, "b": 0.8})
    xs = d.sample_n(RandomStream(1), 10_000)
    assert set(xs) == {"a", "b"}
    assert abs(np.mean(xs == "b") - 0.8) < 0.03
    assert Categorical(["x", "y"]).probabilities == (0.5, 0.5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Uniform(3, 1),
        lambda: Normal(0, -1),
        lambda: Exponential(rate=0),
        lambda: Exponential(),
        lambda: Exponential(rate=1, mean=1),
        lambda: Poisson(-1),
        lambda: Binomial(5, 1.5),
        lambda: Gamma(0),
        lambda: Triangular(1, 5, 3),
        lambda: Empirical([]),
        lambda: Empirical([1, 2], weights=[1]),
        lambda: Categorical(["a", "b"], [0.5, 0.6]),
        lambda: Categorical(["a"], [1.2]),
    ],
)
def test_invalid_parameters_raise(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_from_spec_rejects_unknown_and_bad_params() -> None:
    with pytest.raises(ConfigError):
        from_spec({"distribution": "os.system"})
    with pytest.raises(ConfigError):
        from_spec({"distribution": "uniform", "low": 5, "high": 1})
    with pytest.raises(ConfigError):
        from_spec({"low": 1})
    assert "exponential" in registered_distributions()


def test_as_distribution() -> None:
    assert as_distribution(3) == Constant(3.0)
    assert as_distribution({"distribution": "exponential", "mean": 2}) == Exponential(rate=0.5)
    with pytest.raises(TypeError):
        as_distribution("x")  # type: ignore[arg-type]


def test_custom_distribution() -> None:
    d = Custom(lambda s: s.integers(0, 3) * 10)
    assert d.sample(RandomStream(1)) in {0, 10, 20}
    with pytest.raises(TypeError):
        d.to_spec()
