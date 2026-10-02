from __future__ import annotations

import math

import numpy as np
import pytest

from simulsi import Simulation
from simulsi.core.clock import Clock
from simulsi.metrics import Metrics, Tally, TimeWeighted


def test_tally_matches_numpy() -> None:
    xs = np.random.default_rng(0).normal(5, 2, 1000)
    t = Tally("x")
    for x in xs:
        t.observe(x)
    s = t.summary()
    assert s["mean"] == pytest.approx(xs.mean())
    assert s["std"] == pytest.approx(xs.std(ddof=1))
    assert s["p95"] == pytest.approx(np.quantile(xs, 0.95))
    assert s["min"] == xs.min() and s["max"] == xs.max()


def test_empty_tally_is_nan() -> None:
    s = Tally("x").summary()
    assert s["count"] == 0 and math.isnan(s["mean"]) and math.isnan(s["min"])


def test_time_weighted_average() -> None:
    clock = Clock()
    g = TimeWeighted("wip", clock)
    clock.advance_to(2)
    g.record(4)
    clock.advance_to(5)
    g.record(1)
    clock.advance_to(10)
    # 0*2 + 4*3 + 1*5 = 17 over 10
    assert g.mean == pytest.approx(1.7)
    assert g.series == [(0.0, 0.0), (2.0, 4.0), (5.0, 1.0)]
    assert g.max == 4


def test_metrics_registry_and_flat_names() -> None:
    sim = Simulation(seed=1)
    m = sim.metrics
    sim.call_at(lambda: (m.observe("wait", 2), m.increment("served"), m.record("wip", 3)), time=1)
    sim.call_at(lambda: (m.observe("wait", 4), m.increment("served"), m.record("wip", 1)), time=3)
    sim.call_at(lambda: m.histogram("size", 7, bins=[0, 5, 10]), time=4)
    sim.on_finish(lambda s: s.metrics.set("cost", 99.5))
    flat = sim.run(until=4).metrics
    assert flat["wait.mean"] == 3 and flat["wait.count"] == 2
    assert flat["served"] == 2 and flat["served.rate"] == 0.5
    assert flat["wip.mean"] == pytest.approx((0 * 1 + 3 * 2 + 1 * 1) / 4)
    assert flat["cost"] == 99.5
    assert sim.details()["metrics"]["tallies"]["size"]["histogram"]["counts"] == [0.0, 1.0]


def test_reset_statistics_for_warmup() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("r", 1)

    def job(sim: Simulation):  # type: ignore[no-untyped-def]
        yield from sim.use(r, 10)

    sim.process(job(sim))
    sim.warmup(5)
    m = sim.run(until=20).metrics
    # after warm-up at t=5: busy [5,10], idle [10,20]
    assert m["resource.r.utilization"] == pytest.approx(5 / 15)
    assert m["sim.observed_time"] == 15


def test_metrics_without_values_cannot_give_quantiles() -> None:
    m = Metrics(Clock(), keep_values=False)
    m.observe("x", 1)
    with pytest.raises(ValueError):
        m.tally("x").quantile(0.5)
    assert "x.p50" not in m.flat()
