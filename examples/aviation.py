"""Example 6 - Simplified airport gate operations (synthetic data only).

A day of arrivals is generated from a synthetic schedule (no airline data).
Each flight lands with a random delay, needs a contact gate for its
turnaround and then departs. If no gate is free the aircraft holds on the
taxiway (a *gate conflict*). Aircraft on long ground times are towed to a
remote stand by a tug, freeing the gate. A weather disruption reduces the
number of usable gates for part of the afternoon.

This illustrates SimulSI's generality; it is not an airline planning tool.

    python examples/aviation.py
"""

from __future__ import annotations

from typing import Any

from simulsi import Experiment, Parameter, Scenario, Simulation, compare, model
from simulsi.processes import capacity_reduction
from simulsi.randomness import LogNormal, Triangular


@model(
    duration=20 * 60.0,  # 05:00-01:00, minutes after 05:00
    version="1",
    parameters=[
        Parameter("gates", 22, "int", low=1),
        Parameter("tugs", 2, "int", low=0),
        Parameter("flights", 140, "int", low=1),
        Parameter(
            "tow_threshold",
            150.0,
            "float",
            low=0.0,
            description="tow if scheduled ground time exceeds",
        ),
        Parameter("weather_gates_closed", 0, "int", low=0),
        Parameter(
            "weather_start", 10 * 60.0 - 300, "float", low=0.0, description="minutes after 05:00"
        ),
        Parameter("weather_minutes", 150.0, "float", low=0.0),
        Parameter("mean_arrival_delay", 12.0, "float", low=0.1),
    ],
)
def airport(sim: Simulation, p: Any) -> None:
    gates = sim.resource("gate", p.gates)
    tugs = sim.resource("tug", max(1, p.tugs))
    s_sched, s_ops = sim.stream("schedule"), sim.stream("operations")
    delay = LogNormal.from_moments(p.mean_arrival_delay, p.mean_arrival_delay)
    turnaround = Triangular(35, 50, 80)
    tow_time = Triangular(10, 15, 25)
    sim.metrics.counter("tows")  # report 0 rather than "missing" when nothing is towed
    if p.weather_gates_closed > 0:
        capacity_reduction(
            sim,
            gates,
            at=p.weather_start,
            duration=p.weather_minutes,
            capacity=max(0, p.gates - p.weather_gates_closed),
            name="weather",
        )

    def flight(sim: Simulation, f: Any) -> Any:
        yield max(0.0, f["sta"] - sim.now) + delay.sample(s_ops)
        f.set_state("taxi_in")
        t_landed = sim.now
        gate = yield sim.request(gates, entity=f)
        hold = sim.now - t_landed
        sim.metrics.observe("taxiway_hold", hold)
        sim.metrics.observe("gate_conflict", float(hold > 0))
        f.set_state("at_gate")
        ground = f["std"] - f["sta"]
        if p.tugs > 0 and ground > p.tow_threshold:
            yield turnaround.sample(s_ops) / 2  # deplane + clean
            tug = yield sim.request(tugs, entity=f)
            yield tow_time.sample(s_ops)
            sim.release(gate)
            sim.release(tug)
            sim.metrics.increment("tows")
            f.set_state("remote_stand")
            yield max(0.0, f["std"] - 30 - sim.now)  # towed back to a gate before departure
            gate = yield sim.request(gates, entity=f)
            f.set_state("at_gate")
            yield 25.0  # board
        else:
            yield turnaround.sample(s_ops)
        departure = max(sim.now, f["std"])
        yield departure - sim.now
        sim.release(gate)
        sim.metrics.observe("departure_delay", max(0.0, sim.now - f["std"]))
        sim.dispose(f, "departed")

    # synthetic schedule: two banks of arrivals plus a steady flow
    for i in range(p.flights):
        bank = s_sched.random()
        if bank < 0.35:
            sta = s_sched.normal(3 * 60, 60)
        elif bank < 0.7:
            sta = s_sched.normal(11 * 60 - 300, 60)
        else:
            sta = s_sched.uniform(60, 17 * 60)
        sta = min(max(sta, 0.0), 17 * 60.0)
        ground = 75 if s_sched.random() < 0.75 else s_sched.uniform(180, 420)
        f = sim.entity("flight", number=f"SIM{100 + i}", sta=sta, std=sta + ground)
        sim.process(flight(sim, f), entity=f)


model = airport

METRICS = [
    "taxiway_hold.mean",
    "gate_conflict.mean",
    "departure_delay.mean",
    "departure_delay.p95",
    "tows",
    "resource.gate.utilization",
]


def main(replications: int = 20) -> None:
    base = Scenario("baseline", {})
    scenarios = [
        base,
        base.derive("weather_6_gates_closed", weather_gates_closed=6),
        base.derive("no_towing", tugs=0),
        base.derive("weather_no_towing", weather_gates_closed=6, tugs=0),
    ]
    res = Experiment(airport, scenarios, replications=replications, seed=99).run()
    print(res.format_summary(METRICS))
    print()
    print(compare(res, "baseline", metrics=["taxiway_hold.mean", "departure_delay.mean"]).format())


if __name__ == "__main__":
    main()
