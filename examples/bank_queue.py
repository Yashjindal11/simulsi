"""Example 1 - Bank queue.

Customers arrive at random and wait for one of several tellers. Impatient
customers leave (renege) if they wait too long. We compare staffing levels
and a demand surge, and ask whether 30 replications are enough.

    python examples/bank_queue.py
"""

from __future__ import annotations

from typing import Any

from simulsi import Experiment, Parameter, Scenario, Simulation, compare, model
from simulsi.randomness import Exponential, LogNormal


@model(
    duration=8 * 60.0,  # one 8-hour day, in minutes
    version="1",
    parameters=[
        Parameter("arrival_rate", 1.0, "float", low=0.0, unit="customers/min"),
        Parameter("tellers", 3, "int", low=1),
        Parameter("mean_service", 2.6, "float", low=0.1, unit="min"),
        Parameter(
            "service_cv", 0.6, "float", low=0.0, description="coefficient of variation of service"
        ),
        Parameter("patience", 15.0, "float", low=0.0, unit="min"),
    ],
)
def bank(sim: Simulation, p: Any) -> None:
    tellers = sim.resource("teller", capacity=p.tellers)
    inter = Exponential(rate=p.arrival_rate)
    service = LogNormal.from_moments(p.mean_service, p.service_cv * p.mean_service)
    arrivals, services = sim.stream("arrivals"), sim.stream("service")

    def customer(sim: Simulation, c: Any) -> Any:
        c.set_state("queued")
        req = yield sim.request(tellers, patience=p.patience)
        if not req.granted:
            sim.metrics.increment("customers.reneged")
            sim.dispose(c, "reneged")
            return
        c.set_state("served")
        yield service.sample(services)
        sim.release(req)
        sim.metrics.increment("customers.served")
        sim.dispose(c)

    def source(sim: Simulation) -> Any:
        while True:
            yield inter.sample(arrivals)
            c = sim.entity("customer")
            sim.process(customer(sim, c), entity=c)

    sim.process(source(sim), name="arrivals")

    def finish(sim: Simulation) -> None:
        served = sim.metrics.counter("customers.served").value
        reneged = sim.metrics.counter("customers.reneged").value
        sim.metrics.set("customers.renege_fraction", reneged / max(1.0, served + reneged))

    sim.on_finish(finish)


model = bank

METRICS = [
    "resource.teller.wait.mean",
    "resource.teller.mean_queue_length",
    "resource.teller.utilization",
    "customers.renege_fraction",
]


def main(replications: int = 30) -> None:
    base = Scenario("baseline", {})
    scenarios = [
        base,
        base.derive("four_tellers", tellers=4),
        base.derive("lunch_rush", arrival_rate=1.25),
        base.derive("lunch_rush_four_tellers", arrival_rate=1.25, tellers=4),
    ]
    res = Experiment(bank, scenarios, replications=replications, seed=2026).run()
    print(res.format_summary(METRICS))
    print("\nDifferences vs baseline (paired, common random numbers):")
    print(compare(res, "baseline", metrics=METRICS).format())
    adv = res.replication_advice("resource.teller.wait.mean", relative_precision=0.05)
    print(
        f"\nWait-time estimate: ±{adv.relative_half_width:.1%} with n={adv.n}; "
        f"~{adv.required_n} replications for ±5%."
    )


if __name__ == "__main__":
    main()
