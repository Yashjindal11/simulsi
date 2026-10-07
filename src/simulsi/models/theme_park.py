"""Theme park: ride choice, queues, express passes and smart routing.

Visitors arrive through the morning, stay a few hours and ride as much as
they can. Each ride carries a fixed number of riders per cycle. Visitors
pick rides by popularity, or with ``smart_routing`` by the shortest posted
wait (like a park app), and skip a ride whose posted wait is longer than
their tolerance. A share of visitors hold an express pass and board ahead of
everyone else.

Try: ``express_share`` 0 vs 0.4, ``smart_routing`` false vs true,
``visitors`` 3000 vs 5000, ``coaster_seats`` 24 vs 36.

Time unit: minutes from opening (10 hours). Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.schedules import PiecewiseRate, arrivals
from simulsi.randomness.distributions import Categorical

RIDES = {  # name: (seats per cycle, cycle minutes, popularity)
    "coaster": (24, 3.0, 0.30),
    "drop_tower": (16, 2.5, 0.15),
    "log_flume": (20, 4.0, 0.20),
    "dark_ride": (30, 3.0, 0.20),
    "carousel": (40, 4.0, 0.15),
}


def _build(sim: Simulation, p: Params) -> None:
    seats = {**{k: v[0] for k, v in RIDES.items()}, "coaster": p.coaster_seats}
    rides = {k: sim.resource(k, seats[k], discipline="priority") for k in RIDES}
    cycle = {k: v[1] for k, v in RIDES.items()}
    popularity = Categorical({k: v[2] for k, v in RIDES.items()})
    rs = sim.stream("visitors")
    rate = p.visitors / 300  # arrivals spread over the first five hours, peaking early
    profile = PiecewiseRate([(0, 1.6 * rate), (90, 1.2 * rate), (180, 0.8 * rate), (300, 0.0)])
    m = sim.metrics
    for name in ("visitors", "rides", "skipped", "express_rides"):
        m.counter(name)

    def posted_wait(name: str) -> float:
        return rides[name].queue_size / rides[name].capacity * cycle[name]

    def visitor(sim: Simulation, i: int) -> Any:
        m.increment("visitors")
        express = rs.random() < p.express_share
        leave_at = sim.now + rs.uniform(180, 360)
        count = 0
        while sim.now < leave_at:
            name = min(RIDES, key=posted_wait) if p.smart_routing else popularity.sample(rs)
            if not express and posted_wait(name) > p.wait_tolerance:
                m.increment("skipped")
                yield rs.uniform(10, 25)  # food, shops, shows
                continue
            t0 = sim.now
            req = yield sim.request(rides[name], priority=0 if express else 1)
            m.observe("wait", sim.now - t0)
            m.observe(f"wait.{'express' if express else 'regular'}", sim.now - t0)
            yield cycle[name]
            rides[name].release(req)
            m.increment("rides")
            if express:
                m.increment("express_rides")
            count += 1
            yield rs.uniform(5, 15)  # walk to the next ride
        m.observe("rides_per_visitor", count)

    arrivals(sim, profile, visitor, stream="arrivals")


theme_park = Model(
    _build,
    name="theme_park",
    duration=600.0,
    version="1",
    description="Theme park: ride capacity, posted waits, express passes and app-based smart routing.",
    parameters=[
        Parameter("visitors", 4000, "int", low=0),
        Parameter("express_share", 0.1, "probability", description="visitors with an express pass"),
        Parameter(
            "smart_routing", False, "bool", description="visitors pick the shortest posted wait"
        ),
        Parameter("wait_tolerance", 45.0, "float", low=0, unit="min"),
        Parameter("coaster_seats", 24, "int", low=1),
    ],
    outputs=[
        "rides_per_visitor.mean",
        "wait.mean",
        "wait.regular.mean",
        "wait.express.mean",
        "skipped",
        "resource.coaster.utilization",
    ],
    sim_options={"keep_values": False},
    presets={
        "no_express": {"express_share": 0.0},
        "lots_of_express": {"express_share": 0.4},
        "smart_routing": {"smart_routing": True},
        "crowd_day": {"visitors": 5500},
        "crowd_day_smart": {"visitors": 5500, "smart_routing": True},
        "bigger_coaster": {"coaster_seats": 36},
    },
)
