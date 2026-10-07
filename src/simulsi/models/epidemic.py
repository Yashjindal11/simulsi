"""Epidemic with hospital capacity and a lockdown policy (stochastic SEIR).

People are Susceptible, Exposed (infected, not yet infectious), Infectious
or Removed. Transitions happen one at a time at exponential rates
(Gillespie's exact algorithm), so small outbreaks can fizzle out by chance
and the same parameters give different epidemics. A share of new cases need
a hospital bed; when no bed frees up within ``bed_patience`` days the
patient goes untreated, with a much higher fatality rate. A lockdown starts
when bed occupancy reaches ``lockdown_trigger`` and lifts once it falls
below ``lockdown_release`` (after at least ``lockdown_min_days``).

Try: ``r0`` 1.2 vs 3, ``vaccination`` 0 vs 0.6, ``hospital_beds`` 15 vs 60,
``lockdown_trigger`` 0.5 vs 2 (never), ``lockdown_effect`` 0.3 vs 0.8.

Time unit: days. Synthetic, illustrative only - not for public-health use.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation


def _build(sim: Simulation, p: Params) -> None:
    n = p.population
    vaccinated = round(p.vaccination * n)
    state = {
        "S": n - vaccinated - p.initial_infected,
        "E": 0,
        "I": p.initial_infected,
        "R": vaccinated,
        "lockdown": False,
        "peak_day": 0.0,
    }
    if state["S"] < 0:
        raise ValueError("initial_infected + vaccinated exceeds the population")
    beds = sim.resource("bed", p.hospital_beds)
    rs, rh = sim.stream("transmission"), sim.stream("hospital")
    beta = p.r0 / p.infectious_days
    m = sim.metrics
    for name in ("cases", "hospitalised", "untreated", "deaths", "lockdowns"):
        m.counter(name)

    def record() -> None:
        for k in ("S", "E", "I", "R"):
            m.record(f"people.{k}", state[k])
        g = m.gauges.get("people.I")
        if g is not None and state["I"] >= g.max:
            state["peak_day"] = sim.now

    def patient() -> Any:
        m.increment("hospitalised")
        req = yield sim.request(beds, patience=p.bed_patience)
        if req.granted:
            yield rh.exponential(p.length_of_stay)
            beds.release(req)
            if rh.random() < p.treated_fatality:
                m.increment("deaths")
        else:
            m.increment("untreated")
            if rh.random() < p.untreated_fatality:
                m.increment("deaths")

    def epidemic() -> Any:
        record()
        while state["E"] + state["I"] > 0:
            b = beta * (1 - p.lockdown_effect if state["lockdown"] else 1.0)
            infect = b * state["S"] * state["I"] / n
            progress = state["E"] / p.incubation_days
            recover = state["I"] / p.infectious_days
            total = infect + progress + recover
            yield rs.exponential(1 / total)
            u = rs.random() * total
            if u < infect:
                state["S"] -= 1
                state["E"] += 1
                m.increment("cases")
            elif u < infect + progress:
                state["E"] -= 1
                state["I"] += 1
                if rh.random() < p.hospitalisation_rate:
                    sim.process(patient(), name="patient")
            else:
                state["I"] -= 1
                state["R"] += 1
            record()

    def policy() -> Any:
        m.record("lockdown", 0)

        def over() -> bool:
            return bool(state["E"] + state["I"] == 0)

        while True:
            yield sim.wait_until(
                lambda: over() or beds.in_use >= p.lockdown_trigger * p.hospital_beds
            )
            if over():
                return
            state["lockdown"] = True
            m.increment("lockdowns")
            m.record("lockdown", 1)
            yield p.lockdown_min_days
            yield sim.wait_until(
                lambda: over() or beds.in_use < p.lockdown_release * p.hospital_beds
            )
            state["lockdown"] = False
            m.record("lockdown", 0)

    sim.process(epidemic(), name="epidemic")
    if p.lockdown_trigger <= 1 and p.hospital_beds > 0:
        sim.process(policy(), name="lockdown-policy")

    def finish(sim: Simulation) -> None:
        ever = n - vaccinated - state["S"]
        sim.metrics.set("attack_rate", ever / n)
        sim.metrics.set("peak_day", state["peak_day"])
        lock = sim.metrics.gauges.get("lockdown")
        sim.metrics.set("lockdown_days", lock.mean * lock.elapsed() if lock else 0.0)

    sim.on_finish(finish)


epidemic = Model(
    _build,
    name="epidemic",
    duration=365.0,
    version="1",
    description="Stochastic SEIR epidemic with hospital beds, untreated overflow and a lockdown policy.",
    parameters=[
        Parameter("population", 5000, "int", low=10, high=200_000),
        Parameter("initial_infected", 10, "int", low=1),
        Parameter("r0", 2.5, "float", low=0, description="basic reproduction number"),
        Parameter("incubation_days", 4.0, "float", low=0.1, unit="days"),
        Parameter("infectious_days", 6.0, "float", low=0.1, unit="days"),
        Parameter("vaccination", 0.0, "probability", description="share immune at the start"),
        Parameter("hospitalisation_rate", 0.05, "probability"),
        Parameter("hospital_beds", 30, "int", low=0),
        Parameter("length_of_stay", 8.0, "float", low=0.1, unit="days"),
        Parameter(
            "bed_patience",
            1.0,
            "float",
            low=0,
            unit="days",
            description="how long a patient can wait for a bed",
        ),
        Parameter("treated_fatality", 0.05, "probability"),
        Parameter("untreated_fatality", 0.3, "probability"),
        Parameter(
            "lockdown_trigger",
            0.8,
            "float",
            low=0,
            description="bed occupancy share that starts a lockdown (>1: never)",
        ),
        Parameter(
            "lockdown_release", 0.3, "float", low=0, description="occupancy share that ends it"
        ),
        Parameter(
            "lockdown_effect", 0.6, "probability", description="contact reduction in lockdown"
        ),
        Parameter("lockdown_min_days", 14.0, "float", low=0, unit="days"),
    ],
    outputs=[
        "attack_rate",
        "people.I.max",
        "peak_day",
        "deaths",
        "untreated",
        "lockdown_days",
        "resource.bed.utilization",
    ],
    presets={
        "mild_strain": {"r0": 1.3},
        "aggressive_strain": {"r0": 4.0},
        "vaccinated_60pct": {"vaccination": 0.6},
        "no_lockdown": {"lockdown_trigger": 2.0},
        "early_lockdown": {"lockdown_trigger": 0.3},
        "surge_beds": {"hospital_beds": 80},
    },
)
