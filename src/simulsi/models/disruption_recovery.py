"""Airline disruption recovery: what should operations control do after a storm?

A fleet flies round trips from a hub on a fixed schedule. A storm closes
the hub to departures for ``closure_minutes``; once it reopens, hub
departures are limited by the runway rate, so a backlog forms. Operations
control
follows one ``policy``:

* ``delay`` - fly every leg, however late;
* ``cancel`` - cancel a round trip whose departure would be more than
  ``cancel_threshold`` late, so the aircraft catches up with its schedule;
* ``spares`` - a spare aircraft operates a badly delayed round trip on
  time-ish, and the delayed aircraft takes over its next trip.

Passengers pay the price as delay minutes; cancelled passengers are
rebooked with a long delay (``rebooking_delay``). The model reports
passenger delay hours, cancellations, a cost (delay and cancellation costs)
and how long the schedule took to recover after the storm.

Try: ``policy`` delay vs cancel vs spares, ``closure_minutes`` 60 vs 240,
``cancel_threshold`` 60 vs 180, ``spares`` 1 vs 4, ``runway_rate`` 20 vs 40.

Time unit: minutes over one day. Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.disruption import capacity_reduction
from simulsi.randomness.distributions import Triangular

ON_TIME = 15.0


def _build(sim: Simulation, p: Params) -> None:
    runway = sim.resource("runway", 1)
    spares = sim.resource("spare", max(1, p.spares))
    rs = sim.stream("operations")
    block = Triangular(0.9 * p.block_time, p.block_time, 1.1 * p.block_time)
    trip = 2 * p.block_time + 2 * p.turn_time
    closure_end = p.closure_start + p.closure_minutes
    if p.closure_minutes > 0:
        capacity_reduction(sim, runway, at=p.closure_start, duration=p.closure_minutes, capacity=0)
    m = sim.metrics
    for name in ("legs.flown", "legs.cancelled", "legs.on_time", "spare_trips"):
        m.counter(name)
    state = {"last_late": 0.0, "pax_delay": 0.0}

    def depart(sched: float, hub: bool) -> Any:
        if hub:
            req = yield sim.request(runway)
            yield 60.0 / p.runway_rate
            runway.release(req)
        delay = max(0.0, sim.now - sched)
        m.observe("departure_delay", delay)
        m.increment("legs.flown")
        state["pax_delay"] += delay * p.passengers
        if delay <= ON_TIME:
            m.increment("legs.on_time")
        elif sim.now > closure_end:
            state["last_late"] = max(state["last_late"], sim.now)
        return delay

    def leg(sched: float, hub: bool) -> Any:
        """Fly one leg departing at ``sched`` (or later) and wait out the turn."""
        if sim.now < sched:
            yield sched - sim.now
        yield from depart(sched, hub)
        yield block.sample(rs)
        yield p.min_turn

    def round_trip(sched: float) -> Any:
        yield from leg(sched, hub=True)
        yield from leg(sched + p.block_time + p.turn_time, hub=False)

    def spare_trip(sched: float) -> Any:
        req = yield sim.request(spares)
        m.increment("spare_trips")
        yield from round_trip(max(sched, sim.now + p.spare_setup))
        spares.release(req)

    def aircraft(i: int) -> Any:
        first = p.first_departure + i * p.departure_spacing
        for k in range(p.trips_per_aircraft):
            sched = first + k * trip
            if sim.now < sched:
                yield sched - sim.now
            projected = projected_delay(sched)
            if projected > p.cancel_threshold and p.policy == "cancel":
                m.increment("legs.cancelled", 2)
                state["pax_delay"] += 2 * p.passengers * p.rebooking_delay
                continue
            if (
                projected > p.cancel_threshold
                and p.policy == "spares"
                and p.spares > 0
                and spares.in_use < spares.capacity
            ):
                sim.process(spare_trip(sched), name="spare")
                continue
            yield from round_trip(sched)

    def projected_delay(sched: float) -> float:
        """Departure delay if the flight joined the runway queue now."""
        start = max(sim.now, closure_end if p.closure_start <= sim.now < closure_end else sim.now)
        backlog = (runway.queue_size + runway.in_use) * 60.0 / p.runway_rate
        return float(max(0.0, start + backlog - sched))

    for i in range(p.aircraft):
        sim.process(aircraft(i), name=f"aircraft-{i}")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        flown, cancelled = c["legs.flown"].value, c["legs.cancelled"].value
        total = flown + cancelled
        sim.metrics.set("otp", c["legs.on_time"].value / total if total else float("nan"))
        sim.metrics.set("passenger_delay_hours", state["pax_delay"] / 60)
        sim.metrics.set(
            "recovery_minutes",
            max(0.0, state["last_late"] - closure_end) if p.closure_minutes > 0 else 0.0,
        )
        sim.metrics.set(
            "cost",
            state["pax_delay"] / 60 * p.cost_per_pax_hour
            + cancelled * p.passengers * p.cancel_cost_per_pax,
        )

    sim.on_finish(finish)


disruption_recovery = Model(
    _build,
    name="disruption_recovery",
    duration=24 * 60.0,
    version="1",
    description="Airline operations control after a hub storm: delay vs cancel vs spare-aircraft recovery.",
    parameters=[
        Parameter("aircraft", 16, "int", low=1, high=500),
        Parameter("trips_per_aircraft", 4, "int", low=1, high=12),
        Parameter("block_time", 70.0, "float", low=5, unit="min"),
        Parameter(
            "turn_time", 50.0, "float", low=5, unit="min", description="scheduled ground time"
        ),
        Parameter("min_turn", 30.0, "float", low=1, unit="min", description="minimum ground time"),
        Parameter("first_departure", 360.0, "float", low=0, unit="min"),
        Parameter("departure_spacing", 5.0, "float", low=0, unit="min"),
        Parameter("runway_rate", 30.0, "float", low=1, unit="/h"),
        Parameter("closure_start", 540.0, "float", low=0, unit="min"),
        Parameter("closure_minutes", 150.0, "float", low=0, unit="min"),
        Parameter("policy", "delay", "str", choices=("delay", "cancel", "spares")),
        Parameter("cancel_threshold", 60.0, "float", low=0, unit="min"),
        Parameter("spares", 2, "int", low=0),
        Parameter("spare_setup", 30.0, "float", low=0, unit="min"),
        Parameter("passengers", 150, "int", low=0),
        Parameter(
            "rebooking_delay",
            360.0,
            "float",
            low=0,
            unit="min",
            description="average delay of a cancelled passenger after rebooking",
        ),
        Parameter("cost_per_pax_hour", 40.0, "float", low=0),
        Parameter("cancel_cost_per_pax", 150.0, "float", low=0),
    ],
    outputs=[
        "otp",
        "passenger_delay_hours",
        "legs.cancelled",
        "spare_trips",
        "recovery_minutes",
        "departure_delay.mean",
        "cost",
    ],
    presets={
        "cancel_policy": {"policy": "cancel"},
        "spares_policy": {"policy": "spares"},
        "long_storm_delay": {"closure_minutes": 240.0},
        "long_storm_cancel": {"closure_minutes": 240.0, "policy": "cancel"},
        "long_storm_spares": {"closure_minutes": 240.0, "policy": "spares", "spares": 4},
        "no_storm": {"closure_minutes": 0.0},
    },
)
