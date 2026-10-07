"""Four-stage supply chain and the bullwhip effect (the "beer game").

Customers buy from a retailer, which orders from a wholesaler, which orders
from a distributor, which orders from a factory. Every day each stage ships
what it can (unfilled orders are backlogged), and orders up to a target
based on its own demand forecast (exponential smoothing). Shipments take
``lead_time`` days. Small swings in customer demand become large swings in
orders upstream - the bullwhip effect - because each stage reacts to the
orders of the stage below, not to real demand. Sharing point-of-sale data
(``information_sharing``) lets every stage forecast from customer demand.

Try: ``lead_time`` 1 vs 5, ``smoothing`` 0.1 vs 0.8, ``information_sharing``
false vs true, ``demand_shock`` 0 vs 0.3, ``safety_factor`` 0 vs 2.

Time unit: days. Synthetic data only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation

STAGES = ("retailer", "wholesaler", "distributor", "factory")


def _build(sim: Simulation, p: Params) -> None:
    rs = sim.stream("demand")
    m = sim.metrics
    k = len(STAGES)
    start = p.mean_demand * (p.lead_time + 1)
    on_hand = [start] * k
    backlog = [0.0] * k
    # pipeline[i]: deliveries on their way *to* stage i, as (arrival_day, qty)
    pipeline: list[deque[tuple[float, float]]] = [deque() for _ in range(k)]
    on_order = [0.0] * k
    forecast = [float(p.mean_demand)] * k
    variance = [(p.demand_cv * p.mean_demand) ** 2] * k
    served = {"immediate": 0.0, "demand": 0.0}
    cover = p.lead_time + 1  # days of demand to cover: lead time + review period

    def day() -> Any:
        while True:
            t = sim.now
            mean = p.mean_demand * (1 + (p.demand_shock if t >= p.shock_day else 0.0))
            demand = max(0.0, round(rs.normal(mean, p.demand_cv * mean)))
            m.observe("demand", demand)
            # 1. receive deliveries
            for i in range(k):
                while pipeline[i] and pipeline[i][0][0] <= t:
                    _, q = pipeline[i].popleft()
                    on_hand[i] += q
                    on_order[i] -= q
            # 2. each stage serves its incoming order (customer demand or downstream order)
            incoming = demand
            for i in range(k):
                due = backlog[i] + incoming
                ship = min(on_hand[i], due)
                if i == 0:
                    served["demand"] += incoming
                    served["immediate"] += min(incoming, max(0.0, on_hand[0] - backlog[0]))
                on_hand[i] -= ship
                backlog[i] = due - ship
                if i > 0:
                    pipeline[i - 1].append((t + p.lead_time, ship))
                # 3. forecast and order up to target
                signal = demand if p.information_sharing else incoming
                err = signal - forecast[i]
                forecast[i] += p.smoothing * err
                variance[i] = (1 - p.smoothing) * variance[i] + p.smoothing * err * err
                target = forecast[i] * cover + p.safety_factor * math.sqrt(variance[i] * cover)
                position = on_hand[i] - backlog[i] + on_order[i]
                order = max(0.0, round(target - position))
                if i == k - 1:
                    order = min(order, p.production_capacity)
                    pipeline[i].append((t + p.lead_time, order))
                on_order[i] += order
                m.observe(f"orders.{STAGES[i]}", order)
                m.record(f"inventory.{STAGES[i]}", on_hand[i])
                m.record(f"backlog.{STAGES[i]}", backlog[i])
                m.observe("cost", p.holding_cost * on_hand[i] + p.backorder_cost * backlog[i])
                incoming = order
            yield 1.0

    sim.process(day(), name="days")

    def finish(sim: Simulation) -> None:
        tallies = sim.metrics.tallies
        d = tallies["demand"].std if "demand" in tallies else float("nan")
        for s in STAGES:
            o = tallies.get(f"orders.{s}")
            if o is not None and d and d > 0:
                sim.metrics.set(f"bullwhip.{s}", (o.std / d) ** 2)
        if served["demand"]:
            sim.metrics.set("fill_rate", served["immediate"] / served["demand"])
        c = tallies.get("cost")
        if c is not None and c.count:
            sim.metrics.set("cost_per_day", c.total / (c.count / k))

    sim.on_finish(finish)


supply_chain = Model(
    _build,
    name="supply_chain",
    duration=200.0,
    warmup=20.0,
    version="1",
    description="Four-stage supply chain (beer game): forecasting, lead times and the bullwhip effect.",
    parameters=[
        Parameter("mean_demand", 100.0, "float", low=1, unit="units/day"),
        Parameter(
            "demand_cv", 0.2, "float", low=0, description="coefficient of variation of daily demand"
        ),
        Parameter(
            "demand_shock", 0.0, "float", low=-0.9, description="permanent demand step (fraction)"
        ),
        Parameter("shock_day", 60.0, "float", low=0, unit="days"),
        Parameter("lead_time", 2, "int", low=1, high=30, unit="days"),
        Parameter("smoothing", 0.3, "probability", description="forecast reaction speed (alpha)"),
        Parameter("safety_factor", 1.0, "float", low=0),
        Parameter(
            "information_sharing",
            False,
            "bool",
            description="every stage forecasts from customer demand",
        ),
        Parameter("production_capacity", 250.0, "float", low=1, unit="units/day"),
        Parameter("holding_cost", 1.0, "float", low=0, description="per unit per day"),
        Parameter("backorder_cost", 5.0, "float", low=0, description="per unit per day"),
    ],
    outputs=[
        "bullwhip.retailer",
        "bullwhip.factory",
        "fill_rate",
        "cost_per_day",
        "inventory.factory.mean",
        "backlog.retailer.mean",
    ],
    presets={
        "long_lead_times": {"lead_time": 5},
        "jumpy_forecasts": {"smoothing": 0.8},
        "shared_pos_data": {"information_sharing": True},
        "demand_shock": {"demand_shock": 0.3},
        "lean_no_safety": {"safety_factor": 0.0},
        "tight_factory": {"production_capacity": 110.0, "demand_shock": 0.2},
    },
)
