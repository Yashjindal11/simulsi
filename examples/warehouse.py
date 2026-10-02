"""Example 3 - Warehouse order fulfilment.

Orders arrive through the day (busier mid-day), each with a random number of
lines. Pickers walk the aisles to pick an order, then hand it to a packing
station through a buffer. We track throughput, the open-order backlog and
worker utilization, and compare adding a picker against faster picking
(e.g. better slotting).

    python examples/warehouse.py
"""

from __future__ import annotations

import math
from typing import Any

from simulsi import Experiment, Parameter, Scenario, Simulation, compare, model
from simulsi.randomness import Exponential, Gamma, Poisson


@model(
    duration=16 * 60.0,  # two 8-hour shifts, minutes
    version="1",
    parameters=[
        Parameter("orders_per_hour", 40.0, "float", low=0.0, description="average over the day"),
        Parameter("pickers", 6, "int", low=1),
        Parameter("packers", 2, "int", low=1),
        Parameter("minutes_per_line", 1.2, "float", low=0.0),
        Parameter("travel_minutes", 4.0, "float", low=0.0),
        Parameter("pack_minutes", 2.5, "float", low=0.0),
    ],
)
def warehouse(sim: Simulation, p: Any) -> None:
    pickers = sim.resource("picker", p.pickers)
    packers = sim.resource("packer", p.packers)
    to_pack = sim.queue("pack_buffer")
    s_arr, s_lines, s_work = sim.stream("arrivals"), sim.stream("lines"), sim.stream("work")
    lines = Poisson(3.0)
    travel = Gamma(4.0, p.travel_minutes / 4.0)
    pack = Exponential(mean=p.pack_minutes)
    peak = p.orders_per_hour / 60.0

    def rate(t: float) -> float:
        # sinusoidal day profile, peaking mid-shift; average equals `peak`
        return peak * (1 + 0.5 * math.sin(2 * math.pi * (t % 480) / 480 - math.pi / 2))

    def order_source(sim: Simulation) -> Any:
        # non-homogeneous Poisson arrivals by thinning
        max_rate = peak * 1.5
        while True:
            yield s_arr.exponential(1 / max_rate)
            if s_arr.random() <= rate(sim.now) / max_rate:
                o = sim.entity("order", lines=1 + lines.sample(s_lines))
                sim.metrics.record("backlog", len(sim.active_entities("order")))
                sim.process(pick(sim, o), entity=o)

    def pick(sim: Simulation, o: Any) -> Any:
        o.set_state("waiting_pick")
        req = yield sim.request(pickers)
        o.set_state("picking")
        yield travel.sample(s_work) + o["lines"] * p.minutes_per_line
        sim.release(req)
        o.set_state("waiting_pack")
        yield to_pack.put(o)

    def packer_worker(sim: Simulation) -> Any:
        while True:
            o = yield to_pack.get()
            yield from sim.use(packers, pack.sample(s_work))
            sim.metrics.increment("orders.shipped")
            sim.dispose(o)
            sim.metrics.record("backlog", len(sim.active_entities("order")))

    sim.process(order_source(sim), name="orders")
    for i in range(p.packers):
        sim.process(packer_worker(sim), name=f"packer-{i}")


model = warehouse

METRICS = [
    "orders.shipped",
    "backlog.mean",
    "backlog.max",
    "entity.order.time_in_system.mean",
    "resource.picker.utilization",
    "resource.packer.utilization",
]


def main(replications: int = 15) -> None:
    base = Scenario("baseline", {})
    scenarios = [
        base,
        base.derive("seventh_picker", pickers=7),
        base.derive("better_slotting", minutes_per_line=0.9),
    ]
    res = Experiment(warehouse, scenarios, replications=replications, seed=11).run()
    print(res.format_summary(METRICS))
    print()
    print(
        compare(
            res, "baseline", metrics=["entity.order.time_in_system.mean", "backlog.mean"]
        ).format()
    )


if __name__ == "__main__":
    main()
