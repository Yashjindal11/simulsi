"""Example 5 - Shuttle bus loop.

Buses circulate around a loop of stops. Passengers arrive at each stop and
wait in line; a bus boards as many as fit. Passengers who cannot board are
"left behind" - a direct congestion measure. We compare more buses with
bigger buses for the same total seat count.

    python examples/transportation.py
"""

from __future__ import annotations

from typing import Any

from simulsi import Experiment, Parameter, Scenario, Simulation, compare, model
from simulsi.randomness import Exponential, Normal


@model(
    duration=6 * 60.0,  # minutes
    warmup=30.0,
    version="1",
    parameters=[
        Parameter("stops", 6, "int", low=2),
        Parameter("buses", 3, "int", low=1),
        Parameter("bus_capacity", 40, "int", low=1),
        Parameter("passengers_per_min", 0.8, "float", low=0.0, description="per stop"),
        Parameter(
            "leg_minutes", 4.0, "float", low=0.1, description="mean travel time between stops"
        ),
        Parameter("board_seconds", 3.0, "float", low=0.0),
    ],
)
def shuttle(sim: Simulation, p: Any) -> None:
    lines = [sim.queue(f"stop{i}") for i in range(p.stops)]
    s_pax, s_drive = sim.stream("passengers"), sim.stream("driving")
    leg = Normal(p.leg_minutes, 0.2 * p.leg_minutes)

    def passengers(sim: Simulation, stop: int) -> Any:
        inter = Exponential(rate=p.passengers_per_min)
        while True:
            yield inter.sample(s_pax)
            dest = (stop + 1 + s_pax.integers(0, p.stops - 1)) % p.stops
            pax = sim.entity("passenger", origin=stop, dest=dest)
            pax.set_state("waiting")
            yield lines[stop].put(pax)

    def bus(sim: Simulation, b: int) -> Any:
        onboard: list[Any] = []
        stop = (b * p.stops) // p.buses  # spread buses around the loop
        while True:
            # alight
            staying = []
            for pax in onboard:
                if pax["dest"] == stop:
                    sim.metrics.observe("passenger.ride_time", sim.now - pax["boarded_at"])
                    sim.dispose(pax)
                else:
                    staying.append(pax)
            onboard = staying
            # board
            boarded = 0
            while len(onboard) < p.bus_capacity:
                pax = lines[stop].try_get()
                if pax is None:
                    break
                pax["boarded_at"] = sim.now
                pax.set_state("riding")
                sim.metrics.observe("passenger.wait", sim.now - pax.created_at)
                onboard.append(pax)
                boarded += 1
            sim.metrics.increment("passengers.left_behind", len(lines[stop]))
            sim.metrics.observe("bus.load_factor", len(onboard) / p.bus_capacity)
            yield boarded * p.board_seconds / 60
            yield max(0.5, leg.sample(s_drive))
            stop = (stop + 1) % p.stops

    for i in range(p.stops):
        sim.process(passengers(sim, i), name=f"pax-{i}")
    for b in range(p.buses):
        sim.process(bus(sim, b), name=f"bus-{b}")


model = shuttle

METRICS = [
    "passenger.wait.mean",
    "passenger.wait.p95",
    "passengers.left_behind",
    "bus.load_factor.mean",
]


def main(replications: int = 15) -> None:
    base = Scenario("baseline", {})
    scenarios = [
        base,
        base.derive("four_buses", buses=4),
        base.derive("bigger_buses", bus_capacity=53),  # same seats as four 40-seat buses
    ]
    res = Experiment(shuttle, scenarios, replications=replications, seed=5).run()
    print(res.format_summary(METRICS))
    print()
    print(
        compare(res, "baseline", metrics=["passenger.wait.mean", "passengers.left_behind"]).format()
    )


if __name__ == "__main__":
    main()
