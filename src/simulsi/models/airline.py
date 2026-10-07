"""Airline schedule and delay propagation.

A fleet of aircraft flies rotations through a hub. Each aircraft has a
schedule of legs with planned block times and planned ground time (minimum
turn plus a *schedule buffer*). Delays come from random disruptions
(technical, ATC, late passengers), but most delay in real networks is
*reactionary*: a late inbound aircraft, or a crew arriving late from
another aircraft, delays the next departure, which delays the one after
that. Gates at the hub are held from arrival until pushback, so padding the
schedule also ties up gates. Spare aircraft can take over badly delayed
legs, and legs delayed beyond a limit are cancelled. A weather window cuts
the runway departure rate.

Try: ``schedule_buffer`` 0 vs 30, ``crew_swap_prob`` 0 vs 0.8, ``spares``
0 vs 3, ``weather_minutes`` 0 vs 180, ``gates`` 4 vs 10.

Time unit: minutes from the first departure bank. Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.randomness.distributions import LogNormal, Triangular

ON_TIME_MINUTES = 15.0
SWAP_MINUTES = 25.0


def _build(sim: Simulation, p: Params) -> None:
    gates = sim.resource("gate", p.gates)
    runway = sim.resource("runway", 1)
    spares = sim.resource("spare", max(1, p.spares))
    s_delay, s_ops, s_crew = sim.stream("disruptions"), sim.stream("operations"), sim.stream("crew")
    block = LogNormal.from_moments(p.block_time, 0.08 * p.block_time)
    turn = Triangular(0.9 * p.min_turn, p.min_turn, 1.6 * p.min_turn)
    cycle = p.block_time + p.min_turn + p.schedule_buffer
    last_arrival: dict[int, float] = {}
    m = sim.metrics
    for name in (
        "flights.scheduled",
        "flights.operated",
        "flights.cancelled",
        "flights.on_time",
        "spare_swaps",
    ):
        m.counter(name)
    totals = {"primary": 0.0, "reactionary": 0.0}

    def slot_minutes() -> float:
        rate = float(p.runway_rate)
        if p.weather_start <= sim.now < p.weather_start + p.weather_minutes:
            rate *= p.weather_capacity
        return 60.0 / max(rate, 0.1)

    def fly(sched_dep: float) -> Any:
        """Take off and fly one leg; returns the arrival delay."""
        req = yield sim.request(runway)
        yield slot_minutes()
        runway.release(req)
        m.observe("delay.departure", max(0.0, sim.now - sched_dep))
        yield block.sample(s_ops)
        delay = max(0.0, sim.now - (sched_dep + p.block_time))
        m.observe("delay.arrival", delay)
        m.increment("flights.operated")
        if delay <= ON_TIME_MINUTES:
            m.increment("flights.on_time")
        return delay

    def spare_leg(sched_dep: float) -> Any:
        req = yield sim.request(spares)
        yield max(0.0, sched_dep + SWAP_MINUTES - sim.now)
        yield from fly(sched_dep)
        yield p.block_time + p.min_turn  # positioning back to the hub
        spares.release(req)

    def aircraft(i: int) -> Any:
        first = i * p.bank_spacing
        gate = None
        for k in range(p.legs_per_aircraft):
            sched = first + k * cycle
            m.increment("flights.scheduled")
            if sim.now < sched:
                yield sched - sim.now
            if k > 0 and p.crew_swap_prob > 0 and s_crew.random() < p.crew_swap_prob:
                other = s_crew.integers(0, p.aircraft)
                ready = last_arrival.get(other, 0.0) + p.min_crew_connect
                if ready > sim.now:
                    yield ready - sim.now
            reactionary = max(0.0, sim.now - sched)
            primary = (
                s_delay.exponential(p.mean_primary_delay)
                if s_delay.random() < p.disruption_prob
                else 0.0
            )
            projected = reactionary + primary
            spare_free = p.spares > 0 and spares.in_use < spares.capacity
            if projected > p.swap_threshold and spare_free:
                m.increment("spare_swaps")
                sim.process(spare_leg(sched), name="spare")
                yield primary  # this aircraft recovers on the ground and rejoins its rotation
                continue
            if projected > p.cancel_threshold:
                m.increment("flights.cancelled")
                yield primary
                continue
            totals["primary"] += primary
            totals["reactionary"] += reactionary
            m.observe("delay.reactionary", reactionary)
            yield primary
            if gate is not None:
                gates.release(gate)  # pushback
                gate = None
            yield from fly(sched)
            last_arrival[i] = sim.now
            gate = yield sim.request(gates)
            yield turn.sample(s_ops)
        if gate is not None:
            gates.release(gate)

    for i in range(p.aircraft):
        sim.process(aircraft(i), name=f"aircraft-{i}")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        operated = c["flights.operated"].value
        scheduled = c["flights.scheduled"].value
        sim.metrics.set("otp", c["flights.on_time"].value / operated if operated else float("nan"))
        sim.metrics.set(
            "cancellation_rate", c["flights.cancelled"].value / scheduled if scheduled else 0.0
        )
        total = totals["primary"] + totals["reactionary"]
        sim.metrics.set("reactionary_share", totals["reactionary"] / total if total else 0.0)
        arr = sim.metrics.tallies.get("delay.arrival")
        delay_min = arr.total if arr is not None else 0.0
        cancelled_pax = c["flights.cancelled"].value * p.passengers
        sim.metrics.set(
            "passenger_delay_hours", (delay_min * p.passengers + cancelled_pax * 240) / 60
        )

    sim.on_finish(finish)


airline = Model(
    _build,
    name="airline",
    duration=24 * 60.0,
    version="1",
    description="Airline rotations through a hub: delay propagation, crews, gates, spares, weather.",
    parameters=[
        Parameter("aircraft", 12, "int", low=1, high=200, description="aircraft in the rotation"),
        Parameter("legs_per_aircraft", 6, "int", low=1, high=20),
        Parameter(
            "bank_spacing",
            10.0,
            "float",
            low=0,
            unit="min",
            description="minutes between first departures of consecutive aircraft",
        ),
        Parameter(
            "block_time", 90.0, "float", low=10, unit="min", description="planned flight time"
        ),
        Parameter("min_turn", 35.0, "float", low=5, unit="min", description="minimum ground time"),
        Parameter(
            "schedule_buffer",
            15.0,
            "float",
            low=0,
            unit="min",
            description="planned ground time on top of the minimum turn",
        ),
        Parameter(
            "gates", 6, "int", low=1, description="hub gates (held from arrival to pushback)"
        ),
        Parameter(
            "runway_rate", 30.0, "float", low=1, unit="/h", description="departures per hour"
        ),
        Parameter(
            "disruption_prob", 0.2, "probability", description="chance a leg gets a primary delay"
        ),
        Parameter("mean_primary_delay", 30.0, "float", low=0.1, unit="min"),
        Parameter(
            "crew_swap_prob",
            0.3,
            "probability",
            description="chance the crew connects from another aircraft",
        ),
        Parameter("min_crew_connect", 20.0, "float", low=0, unit="min"),
        Parameter("spares", 1, "int", low=0, description="spare aircraft"),
        Parameter(
            "swap_threshold",
            60.0,
            "float",
            low=0,
            unit="min",
            description="use a spare when the projected delay exceeds this",
        ),
        Parameter("cancel_threshold", 180.0, "float", low=0, unit="min"),
        Parameter("weather_start", 300.0, "float", low=0, unit="min"),
        Parameter("weather_minutes", 0.0, "float", low=0, unit="min"),
        Parameter(
            "weather_capacity", 0.4, "probability", description="runway rate factor during weather"
        ),
        Parameter("passengers", 150, "int", low=0, description="passengers per flight"),
    ],
    outputs=[
        "otp",
        "delay.arrival.mean",
        "reactionary_share",
        "cancellation_rate",
        "passenger_delay_hours",
        "resource.gate.wait.mean",
    ],
    presets={
        "tight_schedule": {"schedule_buffer": 0.0},
        "padded_schedule": {"schedule_buffer": 30.0},
        "crew_chaos": {"crew_swap_prob": 0.8},
        "no_spares": {"spares": 0},
        "thunderstorm": {"weather_minutes": 180.0},
        "storm_with_padding": {"weather_minutes": 180.0, "schedule_buffer": 30.0, "spares": 3},
        "gate_crunch": {"gates": 4, "schedule_buffer": 30.0},
    },
)
