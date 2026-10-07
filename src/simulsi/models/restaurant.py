"""Restaurant evening: tables by size, seating rules, kitchen and walk-aways.

Parties of 1-6 arrive through the evening (reservations at fixed slots plus
walk-ins with a dinner rush). Parties of up to two take a two-top, three or
four a four-top, five or six a six-top. With ``flexible_seating`` a small
party may take a free larger table when its own size is full - faster
seating now, but it can block a bigger party later. Walk-ins leave if no
table frees up within their patience. Seated parties order; the kitchen
(``cooks``) prepares each party's food, then they dine and leave.

Try: ``four_tops`` 6 vs 12, ``cooks`` 2 vs 4, ``flexible_seating`` false
vs true, ``reservation_share`` 0 vs 0.7, ``rush_multiplier`` 1 vs 2.

Time unit: minutes from 17:00 to 23:00. Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.schedules import PiecewiseRate, arrivals
from simulsi.randomness.distributions import Categorical, LogNormal, Triangular

SIZES = Categorical({1: 0.08, 2: 0.42, 3: 0.15, 4: 0.22, 5: 0.07, 6: 0.06})


def _table_for(size: int) -> str:
    return "two_top" if size <= 2 else "four_top" if size <= 4 else "six_top"


def _build(sim: Simulation, p: Params) -> None:
    tables = {
        "two_top": sim.resource("two_top", p.two_tops),
        "four_top": sim.resource("four_top", p.four_tops),
        "six_top": sim.resource("six_top", p.six_tops),
    }
    order = ["two_top", "four_top", "six_top"]
    kitchen = sim.resource("cook", p.cooks)
    rs = sim.stream("guests")
    cook = Triangular(8, 14, 25)
    dine = LogNormal.from_moments(p.dining_minutes, 0.3 * p.dining_minutes)
    base = p.walkins_per_hour / 60 * (1 - p.reservation_share)
    walkins = PiecewiseRate(
        [(0, base), (60, base * p.rush_multiplier), (180, base), (270, base * 0.4)]
    )
    m = sim.metrics
    for name in ("parties", "walked_away", "covers", "upsized_tables"):
        m.counter(name)

    def party(sim: Simulation, i: int, reserved: bool = False) -> Any:
        size = SIZES.sample(rs)
        m.increment("parties")
        t0 = sim.now
        wanted = _table_for(size)
        choice = wanted
        if p.flexible_seating and tables[wanted].available == 0:
            bigger = [t for t in order[order.index(wanted) + 1 :] if tables[t].available > 0]
            if bigger:
                choice = bigger[0]
                m.increment("upsized_tables")
        res = tables[choice]
        patience = None if reserved else rs.exponential(p.walkin_patience)
        req = yield sim.request(res, patience=patience, priority=0 if reserved else 1)
        if not req.granted:
            m.increment("walked_away")
            return
        m.observe("wait_for_table", sim.now - t0)
        yield rs.uniform(5, 10)  # menus and ordering
        k = yield sim.request(kitchen)
        yield cook.sample(rs)
        kitchen.release(k)
        yield dine.sample(rs)
        res.release(req)
        m.increment("covers", size)
        m.observe("revenue", size * p.spend_per_cover)

    def reservations() -> Any:
        slots = list(range(0, 300, 15))
        per_slot = p.reservation_share * p.walkins_per_hour * 6 / len(slots)
        for t in slots:
            if sim.now < t:
                yield t - sim.now
            for _ in range(rs.poisson(per_slot)):
                sim.process(party(sim, 0, reserved=True))

    arrivals(sim, walkins, party, stream="walk-ins", until=300.0)
    sim.process(reservations(), name="reservations")

    def finish(sim: Simulation) -> None:
        rev = sim.metrics.tallies.get("revenue")
        sim.metrics.set("revenue_total", rev.total if rev is not None else 0.0)

    sim.on_finish(finish)


restaurant = Model(
    _build,
    name="restaurant",
    duration=360.0,
    version="1",
    description="Restaurant evening: table mix, flexible seating, reservations, kitchen capacity and walk-aways.",
    parameters=[
        Parameter("two_tops", 8, "int", low=0),
        Parameter("four_tops", 8, "int", low=0),
        Parameter("six_tops", 2, "int", low=0),
        Parameter("cooks", 3, "int", low=1),
        Parameter(
            "walkins_per_hour", 14.0, "float", low=0, description="parties per hour (average)"
        ),
        Parameter("rush_multiplier", 1.6, "float", low=0, description="18:00-20:00 rush factor"),
        Parameter("reservation_share", 0.3, "probability"),
        Parameter("walkin_patience", 20.0, "float", low=0.1, unit="min"),
        Parameter(
            "flexible_seating",
            False,
            "bool",
            description="seat small parties at larger free tables",
        ),
        Parameter("dining_minutes", 55.0, "float", low=5, unit="min"),
        Parameter("spend_per_cover", 38.0, "float", low=0),
    ],
    outputs=[
        "covers",
        "walked_away",
        "wait_for_table.mean",
        "revenue_total",
        "resource.cook.wait.mean",
        "resource.four_top.utilization",
    ],
    presets={
        "more_four_tops": {"two_tops": 4, "four_tops": 12},
        "extra_cook": {"cooks": 4},
        "flexible_seating": {"flexible_seating": True},
        "reservations_heavy": {"reservation_share": 0.7},
        "big_rush": {"rush_multiplier": 2.4},
        "big_rush_flexible": {"rush_multiplier": 2.4, "flexible_seating": True},
    },
)
