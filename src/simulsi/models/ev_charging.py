"""EV charging hub: chargers, a site power limit, queues and balking.

Electric vehicles arrive through the day (morning and evening peaks), need a
random amount of energy, and use a fast (DC) charger if one is free -
otherwise a slow (AC) charger, or they queue for a fast charger and give up
after their patience. All chargers share the site's grid connection: when
too many sessions run at once, each one gets ``grid_limit_kw`` divided by
the number of active sessions, so charging slows down for everyone.

Try: ``fast_chargers`` 4 vs 10, ``grid_limit_kw`` 300 vs 1500,
``arrivals_per_hour`` 8 vs 16, ``patience`` 10 vs 40.

Time unit: minutes over one day. Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.schedules import PiecewiseRate, arrivals
from simulsi.randomness.distributions import LogNormal

DAY = 24 * 60.0


def _build(sim: Simulation, p: Params) -> None:
    fast = sim.resource("fast_charger", p.fast_chargers)
    slow = sim.resource("slow_charger", p.slow_chargers)
    rs = sim.stream("vehicles")
    energy = LogNormal.from_moments(p.mean_energy_kwh, 0.5 * p.mean_energy_kwh)
    base = p.arrivals_per_hour / 60
    profile = PiecewiseRate(
        [
            (0, 0.2 * base),
            (7 * 60, 1.4 * base),
            (10 * 60, base),
            (16 * 60, 1.6 * base),
            (20 * 60, 0.6 * base),
        ],
        period=DAY,
    )
    state = {"active": 0}
    m = sim.metrics
    for name in ("vehicles", "balked", "sessions.fast", "sessions.slow"):
        m.counter(name)
    m.record("site_power_kw", 0.0)

    def charge(kind: str, kwh: float, charger_kw: float) -> Any:
        state["active"] += 1
        power = min(charger_kw, p.grid_limit_kw / state["active"])
        m.record("site_power_kw", min(p.grid_limit_kw, state["active"] * charger_kw))
        m.observe("charge_power_kw", power)
        yield kwh / power * 60
        state["active"] -= 1
        m.record("site_power_kw", min(p.grid_limit_kw, state["active"] * charger_kw))
        m.increment(f"sessions.{kind}")
        m.observe("energy_kwh", kwh)
        m.observe("revenue", kwh * (p.fast_price if kind == "fast" else p.slow_price))

    def vehicle(sim: Simulation, i: int) -> Any:
        m.increment("vehicles")
        kwh = energy.sample(rs)
        t0 = sim.now
        if fast.available == 0 and slow.available > 0 and kwh <= p.slow_ok_kwh:
            req = yield sim.request(slow)
            m.observe("wait", 0.0)
            yield from charge("slow", kwh, p.slow_kw)
            slow.release(req)
            return
        req = yield sim.request(fast, patience=rs.exponential(p.patience))
        if not req.granted:
            m.increment("balked")
            return
        m.observe("wait", sim.now - t0)
        yield from charge("fast", kwh, p.fast_kw)
        fast.release(req)

    arrivals(sim, profile, vehicle, stream="arrivals")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        n = c["vehicles"].value
        sim.metrics.set("balk_rate", c["balked"].value / n if n else 0.0)
        rev = sim.metrics.tallies.get("revenue")
        sim.metrics.set("revenue_total", rev.total if rev is not None else 0.0)

    sim.on_finish(finish)


ev_charging = Model(
    _build,
    name="ev_charging",
    duration=DAY,
    version="1",
    description="EV charging hub: fast and slow chargers, a shared grid limit, queues and balking drivers.",
    parameters=[
        Parameter("fast_chargers", 6, "int", low=0),
        Parameter("slow_chargers", 8, "int", low=0),
        Parameter("fast_kw", 150.0, "float", low=1, unit="kW"),
        Parameter("slow_kw", 22.0, "float", low=1, unit="kW"),
        Parameter("grid_limit_kw", 600.0, "float", low=1, unit="kW", description="site connection"),
        Parameter("arrivals_per_hour", 10.0, "float", low=0),
        Parameter("mean_energy_kwh", 35.0, "float", low=1, unit="kWh"),
        Parameter(
            "slow_ok_kwh",
            25.0,
            "float",
            low=0,
            unit="kWh",
            description="drivers needing less than this accept a slow charger",
        ),
        Parameter("patience", 20.0, "float", low=0.1, unit="min"),
        Parameter("fast_price", 0.55, "float", low=0, unit="/kWh"),
        Parameter("slow_price", 0.35, "float", low=0, unit="/kWh"),
    ],
    outputs=[
        "balk_rate",
        "wait.mean",
        "charge_power_kw.mean",
        "site_power_kw.max",
        "revenue_total",
        "resource.fast_charger.utilization",
    ],
    presets={
        "more_fast_chargers": {"fast_chargers": 10},
        "busy_day_more_chargers": {"arrivals_per_hour": 16.0, "fast_chargers": 10},
        "busy_day_more_chargers_and_grid": {
            "arrivals_per_hour": 16.0,
            "fast_chargers": 10,
            "grid_limit_kw": 1500.0,
        },
        "grid_upgrade": {"fast_chargers": 10, "grid_limit_kw": 1500.0},
        "weak_grid": {"grid_limit_kw": 300.0},
        "busy_day": {"arrivals_per_hour": 16.0},
        "impatient_drivers": {"patience": 8.0},
    },
)
