from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from simulsi import Experiment, Model, Scenario, Simulation
from simulsi.models import mmc
from simulsi.randomness import (
    Binomial,
    Categorical,
    Empirical,
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    Poisson,
    RandomStream,
    Triangular,
    Uniform,
)
from simulsi.statistics import control_variate

DISTS = [
    Uniform(2, 6),
    Normal(10, 2),
    Exponential(rate=0.5),
    Gamma(2.0, 1.5),
    LogNormal(0.5, 0.4),
    Triangular(1, 2, 6),
    Poisson(3.0),
    Binomial(10, 0.3),
    Empirical([1, 2, 3, 10]),
]


@pytest.mark.parametrize("mode", ["inverse", "antithetic"])
@pytest.mark.parametrize("dist", DISTS, ids=lambda d: type(d).__name__)
def test_inverse_sampling_preserves_distributions(dist: Any, mode: str) -> None:
    xs = dist.sample_n(RandomStream(7, mode=mode), 20_000).astype(float)  # type: ignore[arg-type]
    se = (dist.variance / len(xs)) ** 0.5
    assert abs(xs.mean() - dist.mean) <= 5 * se
    single = dist.sample(RandomStream(7, mode=mode))  # type: ignore[arg-type]
    assert np.isfinite(float(single))


def test_antithetic_streams_mirror_uniforms() -> None:
    a, b = RandomStream(3, mode="inverse"), RandomStream(3, mode="antithetic")
    ua, ub = a.stream("x").random(), b.stream("x").random()
    assert ua + ub == pytest.approx(1.0)
    ea, eb = Exponential(mean=1).sample(a), Exponential(mean=1).sample(b)
    assert (ea - 1) * (eb - 1) < 0 or ea == pytest.approx(
        eb
    )  # opposite sides of the median, typically
    assert Categorical(["x", "y"], [0.3, 0.7]).sample(a) in {"x", "y"}
    with pytest.raises(ValueError):
        RandomStream(1, mode="magic")  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        a.generator.weibull(1.0)


def test_antithetic_experiment_reduces_variance() -> None:
    model = mmc.with_options(duration=500, warmup=50)
    sc = Scenario("baseline", {"arrival_rate": 0.7})
    anti = Experiment(model, sc, replications=40, seed=5, antithetic=True).run()
    plain = Experiment(model, sc, replications=40, seed=5).run()
    metric = "resource.server.utilization"
    assert len(anti.values(metric)) == 20 and len(anti.raw_values(metric)) == 40
    raw = anti.raw_values(metric)
    # the two members of each pair are strongly negatively correlated ...
    assert np.corrcoef(raw[0::2], raw[1::2])[0, 1] < -0.3
    # ... so the pair means estimate the mean more precisely than independent runs
    se_anti = anti.values(metric).std(ddof=1) / np.sqrt(20)
    se_plain = plain.values(metric).std(ddof=1) / np.sqrt(40)
    assert se_anti < se_plain, (se_anti, se_plain)
    assert abs(anti.summary([metric])[0]["mean"] - 0.7) < 0.03
    with pytest.raises(Exception, match="even number"):
        Experiment(model, sc, replications=3, antithetic=True).run()


def _mm1_with_service_tally(sim: Simulation, p: Any) -> None:
    server = sim.resource("server", 1)
    arr, svc = sim.stream("arrivals"), sim.stream("service")

    def customer(sim: Simulation) -> Any:
        s = Exponential(rate=p.service_rate).sample(svc)
        sim.metrics.observe("service", s)
        yield from sim.use(server, s)

    def source(sim: Simulation) -> Any:
        while True:
            yield Exponential(rate=p.arrival_rate).sample(arr)
            sim.process(customer(sim))

    sim.process(source(sim))


def test_control_variates_reduce_half_width() -> None:
    model = Model(
        _mm1_with_service_tally,
        name="cv",
        duration=800.0,
        warmup=50.0,
        parameters={"arrival_rate": 0.8, "service_rate": 1.0},
    )
    res = Experiment(model, replications=30, seed=2).run()
    est = res.control_variate("resource.server.wait.mean", {"service.mean": 1.0})
    assert est.n == 30 and est.coefficients["service.mean"] > 0
    assert est.variance_reduction > 0.2, est.to_dict()
    assert est.ci_low < est.mean < est.ci_high
    with pytest.raises(ValueError):
        control_variate([1.0, 2.0], {"x": [1.0, 2.0]}, {})
