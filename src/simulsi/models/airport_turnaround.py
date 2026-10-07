"""Airport turnaround: ground handling as a network of parallel tasks.

Each arriving aircraft needs a set of ground services between on-blocks and
pushback. Some run in parallel, some must wait for others:

* deboarding, then cleaning (cleaning crew) and catering (catering truck);
* baggage unloading then loading (baggage team);
* fuelling (fuel truck) - unless ``fuel_with_pax`` is allowed, boarding can
  only start once fuelling is done;
* boarding starts when the cabin is clean and catered, then pushback (a tug)
  once boarding and loading are complete, not before the scheduled departure.

Shared crews and vehicles serve all aircraft on the ground, so in a *banked*
schedule (flights arriving in waves) they become the bottleneck; a
*depeaked* schedule spreads the same flights out. The model counts which
task was last to finish - the critical path - for every turnaround.

Try: ``banked`` true vs false, ``fuel_trucks`` 1 vs 4, ``baggage_teams`` 2
vs 6, ``fuel_with_pax`` false vs true, ``scheduled_turn`` 40 vs 60.

Time unit: minutes over one day. Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.randomness.distributions import Triangular

ON_TIME = 15.0


def _build(sim: Simulation, p: Params) -> None:
    crews = {
        "cleaning": sim.resource("cleaning_crew", p.cleaning_crews),
        "catering": sim.resource("catering_truck", p.catering_trucks),
        "fuel": sim.resource("fuel_truck", p.fuel_trucks),
        "baggage": sim.resource("baggage_team", p.baggage_teams),
        "tug": sim.resource("tug", p.tugs),
    }
    rs, rd = sim.stream("ground"), sim.stream("arrivals")
    times = {
        "deboard": Triangular(6, 9, 15),
        "clean": Triangular(8, 12, 20),
        "cater": Triangular(6, 10, 18),
        "fuel": Triangular(10, 15, 25),
        "unload": Triangular(8, 12, 20),
        "load": Triangular(10, 14, 24),
        "board": Triangular(12, 18, 28),
        "pushback": Triangular(3, 5, 8),
    }
    m = sim.metrics
    for name in (
        "flights",
        "on_time",
        *(f"critical.{t}" for t in ("fuel", "clean", "cater", "board", "load")),
    ):
        m.counter(name)

    def task(kind: str, crew: str | None) -> Any:
        if crew is None:
            yield times[kind].sample(rs)
        else:
            req = yield sim.request(crews[crew])
            yield times[kind].sample(rs)
            crews[crew].release(req)
        return sim.now

    def bags() -> Any:
        yield from task("unload", "baggage")
        yield from task("load", "baggage")
        return sim.now

    def cabin() -> Any:
        yield from task("deboard", None)
        clean = sim.process(task("clean", "cleaning"))
        cater = sim.process(task("cater", "catering"))
        yield sim.all_of(clean, cater)
        return {"clean": clean.value, "cater": cater.value}

    def turnaround(sched_arr: float) -> Any:
        delay = rd.exponential(p.mean_arrival_delay) if rd.random() < p.late_arrival_prob else 0.0
        yield max(0.0, sched_arr + delay - sim.now)
        on_blocks = sim.now
        sched_dep = sched_arr + p.scheduled_turn
        fuel = sim.process(task("fuel", "fuel"))
        baggage = sim.process(bags())
        cab = sim.process(cabin())
        gate = [cab] if p.fuel_with_pax else [cab, fuel]
        yield sim.all_of(gate)
        ready = dict(cab.value)
        if not p.fuel_with_pax:
            ready["fuel"] = fuel.value
        m.increment("critical." + max(ready, key=lambda k: ready[k]))
        yield from task("board", None)
        board_done = sim.now
        yield sim.all_of(baggage, fuel)
        last = {"board": board_done, "load": baggage.value, "fuel": fuel.value}
        if max(last.values()) > board_done:
            m.increment("critical." + max(last, key=lambda k: last[k]))
        if sim.now < sched_dep:
            yield sched_dep - sim.now
        req = yield sim.request(crews["tug"])
        yield times["pushback"].sample(rs)
        crews["tug"].release(req)
        dep_delay = max(0.0, sim.now - sched_dep)
        m.observe("turn_time", sim.now - on_blocks)
        m.observe("departure_delay", dep_delay)
        m.increment("flights")
        if dep_delay <= ON_TIME:
            m.increment("on_time")

    day = 18 * 60.0
    n = p.flights
    if p.banked:
        per_bank = max(1, round(n / p.banks))
        gap = day / p.banks
        schedule = [
            b * gap + 60 + (i * p.bank_spread / per_bank)
            for b in range(p.banks)
            for i in range(per_bank)
        ][:n]
    else:
        schedule = [60 + i * day / n for i in range(n)]
    for t in schedule:
        sim.process(turnaround(t), name="turnaround")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        flights = c["flights"].value
        sim.metrics.set("otp", c["on_time"].value / flights if flights else float("nan"))

    sim.on_finish(finish)


airport_turnaround = Model(
    _build,
    name="airport_turnaround",
    duration=21 * 60.0,
    version="1",
    description="Aircraft turnarounds: parallel ground-handling tasks, shared crews, banked vs depeaked schedules.",
    parameters=[
        Parameter("flights", 90, "int", low=1, high=1000),
        Parameter("banked", True, "bool", description="flights arrive in waves (banks)"),
        Parameter("banks", 6, "int", low=1),
        Parameter(
            "bank_spread",
            60.0,
            "float",
            low=0,
            unit="min",
            description="arrival spread within a bank",
        ),
        Parameter(
            "scheduled_turn", 55.0, "float", low=10, unit="min", description="planned ground time"
        ),
        Parameter("late_arrival_prob", 0.3, "probability"),
        Parameter("mean_arrival_delay", 15.0, "float", low=0.1, unit="min"),
        Parameter("cleaning_crews", 6, "int", low=1),
        Parameter("catering_trucks", 5, "int", low=1),
        Parameter("fuel_trucks", 5, "int", low=1),
        Parameter("baggage_teams", 6, "int", low=1),
        Parameter("tugs", 4, "int", low=1),
        Parameter("fuel_with_pax", False, "bool", description="allow fuelling during boarding"),
    ],
    outputs=[
        "otp",
        "departure_delay.mean",
        "turn_time.mean",
        "critical.fuel",
        "critical.load",
        "resource.fuel_truck.wait.mean",
        "resource.baggage_team.wait.mean",
    ],
    presets={
        "depeaked": {"banked": False},
        "fuel_with_passengers": {"fuel_with_pax": True},
        "one_fuel_truck_short": {"fuel_trucks": 4},
        "extra_baggage_teams": {"baggage_teams": 8},
        "tight_turns": {"scheduled_turn": 45.0},
        "big_banks": {"banks": 3},
    },
)
