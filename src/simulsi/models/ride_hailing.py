"""Ride-hailing: matching, pickup times, cancellations and surge pricing.

Riders request trips at a rate that peaks in the morning and evening rush.
The nearest idle driver is assigned; with few idle drivers the nearest one is
far away, so pickups take longer, which keeps drivers busy for longer and
leaves even fewer idle - a vicious circle (the "wild goose chase"). Riders
cancel if no driver is found within their patience or the pickup estimate
is too long. Optional surge pricing raises the price as the share of busy
drivers climbs: some riders decline the higher price, and more drivers log
on.

Try: ``drivers`` 40 vs 90, ``rush_multiplier`` 1 vs 3, ``surge`` false vs
true, ``price_elasticity`` 0.3 vs 2, ``max_pickup_eta`` 8 vs 20.

Time unit: minutes over one day. Synthetic data only.
"""

from __future__ import annotations

import math
from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.schedules import PiecewiseRate, arrivals
from simulsi.randomness.distributions import LogNormal

CHECK_EVERY = 5.0


def _build(sim: Simulation, p: Params) -> None:
    drivers = sim.resource("driver", p.drivers)
    rs = sim.stream("riders")
    base = p.requests_per_hour / 60
    rush = base * p.rush_multiplier
    profile = PiecewiseRate(
        [
            (0, base * 0.3),
            (6 * 60, base),
            (7 * 60, rush),
            (9 * 60, base),
            (17 * 60, rush),
            (19 * 60, base),
            (23 * 60, base * 0.5),
        ],
        period=24 * 60,
    )
    trip = LogNormal.from_moments(p.trip_minutes, 0.5 * p.trip_minutes)
    surge = {"x": 1.0}
    m = sim.metrics
    for name in ("requests", "completed", "price_declined", "no_driver", "eta_too_long"):
        m.counter(name)
    m.record("surge", 1.0)

    def pickup_minutes() -> float:
        # Distance to the nearest of k idle drivers scales like 1 / sqrt(k).
        idle = drivers.available
        return min(30.0, max(2.0, float(p.pickup_scale) / math.sqrt(idle + 1)))

    def rider(sim: Simulation, i: int) -> Any:
        m.increment("requests")
        if surge["x"] > 1 and rs.random() > surge["x"] ** -p.price_elasticity:
            m.increment("price_declined")
            return
        req = yield sim.request(drivers, patience=rs.exponential(p.rider_patience))
        if not req.granted:
            m.increment("no_driver")
            return
        eta = pickup_minutes()
        if eta > p.max_pickup_eta:
            drivers.release(req)
            m.increment("eta_too_long")
            return
        m.observe("pickup_minutes", eta)
        yield eta
        duration = trip.sample(rs)
        yield duration
        drivers.release(req)
        m.increment("completed")
        m.observe("fare", (p.base_fare + p.per_minute * duration) * surge["x"])

    arrivals(sim, profile, rider, stream="requests")

    def pricing() -> Any:
        while True:
            yield CHECK_EVERY
            if not p.surge:
                continue
            busy = drivers.in_use / max(1, drivers.capacity)
            pressure = min(1.0, max(0.0, (busy - p.surge_from) / (1 - p.surge_from)))
            surge["x"] = 1.0 + (p.max_surge - 1.0) * pressure
            m.record("surge", surge["x"])
            online = round(p.drivers * (1 + p.supply_response * (surge["x"] - 1)))
            if online != drivers.capacity:
                drivers.set_capacity(online)

    sim.process(pricing(), name="pricing")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        req = c["requests"].value
        sim.metrics.set("service_level", c["completed"].value / req if req else float("nan"))
        fares = sim.metrics.tallies.get("fare")
        revenue = fares.total if fares is not None else 0.0
        sim.metrics.set("revenue", revenue)
        hours = drivers.capacity_level.area() / 60
        sim.metrics.set(
            "driver_earnings_per_hour", p.driver_share * revenue / hours if hours else 0.0
        )

    sim.on_finish(finish)


ride_hailing = Model(
    _build,
    name="ride_hailing",
    duration=24 * 60.0,
    version="1",
    description="Ride-hailing day: rush hours, pickup-time feedback, cancellations and surge pricing.",
    parameters=[
        Parameter("drivers", 60, "int", low=1, high=2000),
        Parameter("requests_per_hour", 150.0, "float", low=0),
        Parameter("rush_multiplier", 2.0, "float", low=0, description="rush-hour demand factor"),
        Parameter("trip_minutes", 18.0, "float", low=1, unit="min"),
        Parameter(
            "pickup_scale",
            25.0,
            "float",
            low=0,
            unit="min",
            description="pickup time with one idle driver",
        ),
        Parameter("max_pickup_eta", 12.0, "float", low=0, unit="min"),
        Parameter("rider_patience", 4.0, "float", low=0.1, unit="min"),
        Parameter("surge", False, "bool"),
        Parameter(
            "surge_from",
            0.7,
            "probability",
            description="share of drivers busy at which prices start to rise",
        ),
        Parameter("max_surge", 2.5, "float", low=1),
        Parameter("price_elasticity", 1.0, "float", low=0),
        Parameter(
            "supply_response",
            0.4,
            "float",
            low=0,
            description="extra drivers online per unit of surge above 1",
        ),
        Parameter("base_fare", 2.5, "float", low=0),
        Parameter("per_minute", 0.6, "float", low=0),
        Parameter("driver_share", 0.75, "probability"),
    ],
    outputs=[
        "service_level",
        "pickup_minutes.mean",
        "eta_too_long",
        "price_declined",
        "surge.mean",
        "resource.driver.utilization",
        "revenue",
        "driver_earnings_per_hour",
    ],
    sim_options={"keep_values": False},
    presets={
        "small_fleet": {"drivers": 40},
        "big_fleet": {"drivers": 90},
        "surge_pricing": {"surge": True},
        "concert_night": {"rush_multiplier": 3.5},
        "concert_night_with_surge": {"rush_multiplier": 3.5, "surge": True},
        "impatient_riders": {"rider_patience": 1.5, "max_pickup_eta": 8.0},
    },
)
