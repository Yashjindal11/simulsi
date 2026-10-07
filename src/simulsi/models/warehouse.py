"""Warehouse order fulfilment: picking strategy, packing and truck cut-offs.

Orders arrive through the shift. With ``discrete`` picking a picker walks
the aisles for one order at a time; with ``batch`` picking a picker
collects ``batch_size`` orders in one tour, sharing the walking time but
making early orders wait for the batch to fill. Picked orders are packed at
packing stations and must be ready before the next truck departure (every
``truck_interval`` minutes) to ship on time.

Try: ``picking`` discrete vs batch, ``batch_size`` 2 vs 8, ``pickers`` 10
vs 14, ``packers`` 4 vs 8, ``orders_per_hour`` 90 vs 130.

Time unit: minutes over an 8-hour shift. Synthetic data only.
"""

from __future__ import annotations

import math
from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.flow import batch
from simulsi.randomness.distributions import Triangular


def _build(sim: Simulation, p: Params) -> None:
    pickers = sim.resource("picker", p.pickers)
    packers = sim.resource("packer", p.packers)
    inbox = sim.queue("orders")
    rs = sim.stream("orders")
    pack = Triangular(1.0, 2.0, 4.0)
    m = sim.metrics
    for name in ("orders", "shipped_on_time", "shipped_late", "tours"):
        m.counter(name)

    def next_truck(t: float) -> float:
        return float(math.ceil(max(t, 1e-9) / p.truck_interval) * p.truck_interval)

    def finish_order(arrived: float, lines: int) -> Any:
        req = yield sim.request(packers)
        yield pack.sample(rs) + 0.2 * lines
        packers.release(req)
        m.observe("cycle_time", sim.now - arrived)
        due = next_truck(arrived + p.min_processing)
        m.increment("shipped_on_time" if sim.now <= due else "shipped_late")

    def tour(orders: list[tuple[float, int]]) -> Any:
        req = yield sim.request(pickers)
        m.increment("tours")
        lines = sum(n for _, n in orders)
        # one walk through the aisles per tour, plus per-line search and a sorting step for batches
        sort = 0.3 * lines if len(orders) > 1 else 0.0
        yield rs.triangular(0.8, 1.0, 1.3) * p.walk_minutes + p.minutes_per_line * lines + sort
        pickers.release(req)
        for arrived, n in orders:
            sim.process(finish_order(arrived, n), name="pack")

    def source() -> Any:
        while True:
            yield rs.exponential(60 / p.orders_per_hour)
            m.increment("orders")
            yield inbox.put((sim.now, 1 + rs.poisson(p.mean_lines - 1)))

    def dispatcher() -> Any:
        while True:
            if p.picking == "batch":
                orders = yield from batch(inbox, p.batch_size, timeout=p.batch_timeout)
                if not orders:
                    continue
            else:
                orders = [(yield inbox.get())]
            if pickers.available == 0:
                yield sim.wait_until(lambda: pickers.available > 0)
            sim.process(tour(orders), name="tour")

    sim.process(source(), name="arrivals")
    sim.process(dispatcher(), name="dispatcher")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        done = c["shipped_on_time"].value + c["shipped_late"].value
        sim.metrics.set("on_time_rate", c["shipped_on_time"].value / done if done else float("nan"))

    sim.on_finish(finish)


warehouse = Model(
    _build,
    name="warehouse",
    duration=480.0,
    warmup=30.0,
    version="1",
    description="Warehouse fulfilment: discrete vs batch picking, packing capacity and truck cut-offs.",
    parameters=[
        Parameter("orders_per_hour", 90.0, "float", low=0),
        Parameter("mean_lines", 3.0, "float", low=1, description="items per order"),
        Parameter("picking", "discrete", "str", choices=("discrete", "batch")),
        Parameter("batch_size", 4, "int", low=1),
        Parameter(
            "batch_timeout",
            10.0,
            "float",
            low=0,
            unit="min",
            description="start a partial batch after this long",
        ),
        Parameter("pickers", 12, "int", low=1),
        Parameter("packers", 6, "int", low=1),
        Parameter(
            "walk_minutes", 6.0, "float", low=0, unit="min", description="aisle walk per tour"
        ),
        Parameter("minutes_per_line", 0.5, "float", low=0, unit="min"),
        Parameter("truck_interval", 60.0, "float", low=1, unit="min"),
        Parameter(
            "min_processing",
            20.0,
            "float",
            low=0,
            unit="min",
            description="orders arriving less than this before a truck catch the next one",
        ),
    ],
    outputs=[
        "on_time_rate",
        "cycle_time.mean",
        "tours",
        "resource.picker.utilization",
        "resource.packer.wait.mean",
    ],
    presets={
        "batch_picking": {"picking": "batch"},
        "big_batches": {"picking": "batch", "batch_size": 8},
        "peak_day": {"orders_per_hour": 130.0},
        "peak_day_batch": {"orders_per_hour": 130.0, "picking": "batch"},
        "extra_packers": {"packers": 8},
        "fewer_pickers": {"pickers": 10},
    },
)
