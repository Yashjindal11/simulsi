"""The smallest useful SimulSI model: an M/M/c queue.

simulsi run examples/queue.py
simulsi run examples/queue.py --param servers=3 --param arrival_rate=2.5
python examples/queue.py
"""

from __future__ import annotations

from typing import Any

from simulsi import Parameter, Simulation, model
from simulsi.models import erlang_c
from simulsi.randomness import Exponential


@model(
    duration=10_000.0,
    warmup=1_000.0,
    parameters=[
        Parameter("arrival_rate", 1.8, "float", low=0.0, description="arrivals per time unit"),
        Parameter(
            "service_rate", 1.0, "float", low=0.0, description="services per server per time unit"
        ),
        Parameter("servers", 2, "int", low=1),
    ],
)
def queue(sim: Simulation, p: Any) -> None:
    server = sim.resource("server", capacity=p.servers)
    inter = Exponential(rate=p.arrival_rate)
    service = Exponential(rate=p.service_rate)
    arrivals, services = sim.stream("arrivals"), sim.stream("service")

    def customer(sim: Simulation) -> Any:
        yield sim.request(server)
        yield service.sample(services)
        sim.release(server)

    def source(sim: Simulation) -> Any:
        while True:
            yield inter.sample(arrivals)
            sim.process(customer(sim))

    sim.process(source(sim), name="arrivals")


model = queue


def main(replications: int = 20) -> None:
    from simulsi import Experiment

    res = Experiment(queue, replications=replications, seed=42).run()
    theory = erlang_c(1.8, 1.0, 2)
    print(res.format_summary(["resource.server.wait.mean", "resource.server.utilization"]))
    print(
        f"\nErlang C: mean wait {theory['mean_wait']:.3f}, utilization {theory['utilization']:.3f}"
    )


if __name__ == "__main__":
    main()
