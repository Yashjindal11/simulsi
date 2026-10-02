"""Built-in reference models.

These are small, well-understood models with known analytic results. They
serve as quick-start examples, CLI defaults (``simulsi run builtin:mmc``)
and as correctness checks for the engine.
"""

from __future__ import annotations

import math
from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.randomness.distributions import Exponential


def erlang_c(arrival_rate: float, service_rate: float, servers: int) -> dict[str, float]:
    """Steady-state M/M/c results (Erlang C). Requires utilization < 1."""
    lam, mu, c = arrival_rate, service_rate, servers
    a = lam / mu
    rho = a / c
    if rho >= 1:
        raise ValueError(f"unstable queue: utilization {rho:.3f} >= 1")
    head = sum(a**k / math.factorial(k) for k in range(c))
    tail = a**c / (math.factorial(c) * (1 - rho))
    p_wait = tail / (head + tail)
    wq = p_wait / (c * mu - lam)
    return {
        "utilization": rho,
        "p_wait": p_wait,
        "mean_wait": wq,
        "mean_queue_length": lam * wq,
        "mean_time_in_system": wq + 1 / mu,
    }


def _mmc_build(sim: Simulation, p: Params) -> None:
    server = sim.resource("server", capacity=p.servers)
    inter = Exponential(rate=p.arrival_rate)
    service = Exponential(rate=p.service_rate)
    arrivals, services = sim.stream("arrivals"), sim.stream("service")

    def customer(sim: Simulation) -> Any:
        yield from sim.use(server, service.sample(services))

    def source(sim: Simulation) -> Any:
        while True:
            yield inter.sample(arrivals)
            sim.process(customer(sim))

    sim.process(source(sim), name="arrivals")


mmc = Model(
    _mmc_build,
    name="mmc",
    duration=10_000.0,
    warmup=1_000.0,
    version="1",
    description="M/M/c queue: Poisson arrivals, exponential service, c identical servers.",
    parameters=[
        Parameter("arrival_rate", 0.9, "float", low=0.0, description="customers per time unit"),
        Parameter(
            "service_rate", 1.0, "float", low=0.0, description="services per server per time unit"
        ),
        Parameter("servers", 1, "int", low=1, description="number of servers"),
    ],
    outputs=["resource.server.wait.mean", "resource.server.utilization"],
    sim_options={"keep_values": False, "record_series": False, "keep_entity_history": False},
)

BUILTIN_MODELS: dict[str, Model] = {"mmc": mmc}
