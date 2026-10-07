"""Inventory under an (s, S) policy: when to reorder and how much.

A single item faces random daily demand. Whenever the *inventory position*
(stock on hand plus stock on order) falls to the reorder point ``s`` or
below, an order is placed to bring it back up to ``S``; it arrives after a
random lead time. Demand that cannot be met is lost. Each order has a fixed
cost, stock costs money to hold, and lost sales cost lost margin. This is
the classic setting for simulation optimisation, for example::

    optimize(inventory, "cost_per_day", {"s": (20, 200), "S": (100, 500)}, method="bayes")

Try: ``s`` 40 vs 120, ``S`` 150 vs 400, ``mean_lead_time`` 2 vs 8,
``demand_cv`` 0.2 vs 0.8, ``order_cost`` 50 vs 500.

Time unit: days. Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.randomness.distributions import Gamma


def _build(sim: Simulation, p: Params) -> None:
    rs, rl = sim.stream("demand"), sim.stream("lead_time")
    shape = 1 / max(p.demand_cv, 1e-6) ** 2
    demand = Gamma(shape, p.mean_demand / shape)
    lead_shape = 4.0
    lead = Gamma(lead_shape, p.mean_lead_time / lead_shape)
    state = {"on_hand": float(p.S), "on_order": 0.0}
    m = sim.metrics
    for name in ("orders", "demand", "sold", "lost"):
        m.counter(name)
    if not p.s < p.S:
        raise ValueError(f"S ({p.S}) must be larger than s ({p.s})")

    def delivery(qty: float) -> Any:
        yield lead.sample(rl)
        state["on_hand"] += qty
        state["on_order"] -= qty
        m.record("on_hand", state["on_hand"])

    def days() -> Any:
        m.record("on_hand", state["on_hand"])
        while True:
            d = round(demand.sample(rs))
            sold = min(d, state["on_hand"])
            state["on_hand"] -= sold
            m.increment("demand", d)
            m.increment("sold", sold)
            m.increment("lost", d - sold)
            m.record("on_hand", state["on_hand"])
            position = state["on_hand"] + state["on_order"]
            cost = p.holding_cost * state["on_hand"] + p.lost_sale_cost * (d - sold)
            if position <= p.s:
                qty = p.S - position
                state["on_order"] += qty
                m.increment("orders")
                cost += p.order_cost
                sim.process(delivery(qty), name="delivery")
            m.observe("daily_cost", cost)
            yield 1.0

    sim.process(days(), name="days")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        dem = c["demand"].value
        sim.metrics.set("fill_rate", c["sold"].value / dem if dem else float("nan"))
        cost = sim.metrics.tallies.get("daily_cost")
        sim.metrics.set("cost_per_day", cost.mean if cost is not None else float("nan"))

    sim.on_finish(finish)


inventory = Model(
    _build,
    name="inventory",
    duration=365.0,
    warmup=30.0,
    version="1",
    description="Single-item (s, S) inventory policy: reorder point, order-up-to level, lost sales and costs.",
    parameters=[
        Parameter("s", 80, "int", low=0, description="reorder point"),
        Parameter("S", 250, "int", low=1, description="order-up-to level"),
        Parameter("mean_demand", 20.0, "float", low=0, unit="units/day"),
        Parameter("demand_cv", 0.4, "float", low=0),
        Parameter("mean_lead_time", 4.0, "float", low=0.01, unit="days"),
        Parameter("order_cost", 150.0, "float", low=0),
        Parameter("holding_cost", 0.2, "float", low=0, description="per unit per day"),
        Parameter("lost_sale_cost", 8.0, "float", low=0, description="margin lost per unit"),
    ],
    outputs=["fill_rate", "cost_per_day", "on_hand.mean", "orders", "lost"],
    presets={
        "low_reorder_point": {"s": 40},
        "high_reorder_point": {"s": 120},
        "small_orders": {"S": 150},
        "big_orders": {"S": 400},
        "slow_supplier": {"mean_lead_time": 8.0},
        "volatile_demand": {"demand_cv": 0.8},
    },
)
