"""Check simulated queueing output against closed-form M/M/c results."""

from __future__ import annotations

import math
from typing import Any

import pytest

from simulsi import Simulation
from simulsi.randomness import Exponential

pytestmark = pytest.mark.statistical


def erlang_c_wait(lam: float, mu: float, c: int) -> tuple[float, float]:
    """Mean wait in queue Wq and utilization for an M/M/c queue."""
    a = lam / mu
    rho = a / c
    s = sum(a**k / math.factorial(k) for k in range(c))
    top = a**c / (math.factorial(c) * (1 - rho))
    p_wait = top / (s + top)
    return p_wait / (c * mu - lam), rho


def simulate_mmc(lam: float, mu: float, c: int, seed: int, horizon: float) -> dict[str, float]:
    sim = Simulation(seed=seed, keep_values=False, record_series=False, keep_entity_history=False)
    server = sim.resource("server", capacity=c)
    arrivals = sim.stream("arrivals")
    service = sim.stream("service")
    inter, svc = Exponential(rate=lam), Exponential(rate=mu)

    def customer(sim: Simulation) -> Any:
        yield from sim.use(server, svc.sample(service))

    def source(sim: Simulation) -> Any:
        while True:
            yield inter.sample(arrivals)
            sim.process(customer(sim))

    sim.process(source(sim))
    sim.warmup(horizon * 0.1)
    return sim.run(until=horizon).metrics


@pytest.mark.parametrize(("lam", "mu", "c"), [(0.8, 1.0, 1), (1.5, 1.0, 2), (4.0, 1.5, 3)])
def test_mmc_matches_erlang_c(lam: float, mu: float, c: int) -> None:
    wq, rho = erlang_c_wait(lam, mu, c)
    waits, utils = [], []
    for seed in range(8):
        m = simulate_mmc(lam, mu, c, seed, horizon=20_000)
        waits.append(m["resource.server.wait.mean"])
        utils.append(m["resource.server.utilization"])
    mean_w = sum(waits) / len(waits)
    sd_w = math.sqrt(sum((w - mean_w) ** 2 for w in waits) / (len(waits) - 1))
    half = 4.03 * sd_w / math.sqrt(len(waits))  # ~99.5% t-interval with 7 dof, generous
    assert abs(mean_w - wq) < max(half, 0.03 * wq), (mean_w, wq, half)
    assert sum(utils) / len(utils) == pytest.approx(rho, rel=0.02)


def test_reproducibility_of_full_model() -> None:
    a = simulate_mmc(0.9, 1.0, 1, seed=99, horizon=2_000)
    b = simulate_mmc(0.9, 1.0, 1, seed=99, horizon=2_000)
    c = simulate_mmc(0.9, 1.0, 1, seed=100, horizon=2_000)
    assert a == b
    assert a["resource.server.wait.mean"] != c["resource.server.wait.mean"]
