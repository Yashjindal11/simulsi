"""Example 2 - Emergency department.

Patients arrive with an acuity level (1 = critical ... 3 = routine), are
triaged by a nurse, wait for a treatment room, then for a doctor. Rooms and
doctors serve the most acute patient first. We measure waits by acuity and
the *service level*: the share of patients seen by a doctor within their
target time.

    python examples/hospital.py
"""

from __future__ import annotations

from typing import Any

from simulsi import Experiment, Parameter, Scenario, Simulation, compare, model
from simulsi.randomness import Categorical, Exponential, Triangular

TARGET_MINUTES = {1: 10.0, 2: 30.0, 3: 120.0}


@model(
    duration=7 * 24 * 60.0,  # one week, minutes
    warmup=24 * 60.0,
    version="1",
    parameters=[
        Parameter("arrivals_per_hour", 9.0, "float", low=0.0),
        Parameter("nurses", 2, "int", low=1),
        Parameter("rooms", 8, "int", low=1),
        Parameter("doctors", 3, "int", low=1),
        Parameter("p_critical", 0.1, "probability"),
        Parameter("p_urgent", 0.35, "probability"),
    ],
)
def emergency_department(sim: Simulation, p: Any) -> None:
    nurses = sim.resource("nurse", p.nurses)
    rooms = sim.resource("room", p.rooms, discipline="priority")
    doctors = sim.resource("doctor", p.doctors, discipline="priority")
    inter = Exponential(rate=p.arrivals_per_hour / 60)
    acuity = Categorical([1, 2, 3], [p.p_critical, p.p_urgent, 1 - p.p_critical - p.p_urgent])
    triage_time = Triangular(3, 5, 10)
    consult = {1: Triangular(20, 40, 90), 2: Triangular(15, 25, 50), 3: Triangular(8, 15, 30)}
    s_arr, s_acu, s_svc = sim.stream("arrivals"), sim.stream("acuity"), sim.stream("service")

    def patient(sim: Simulation, pt: Any) -> Any:
        level = pt["acuity"]
        pt.set_state("triage_queue")
        yield from sim.use(nurses, triage_time.sample(s_svc))
        pt.set_state("waiting_room")
        room = yield sim.request(rooms, priority=level)
        pt.set_state("waiting_doctor")
        doc = yield sim.request(doctors, priority=level)
        door_to_doctor = sim.now - pt.created_at
        sim.metrics.observe(f"door_to_doctor.acuity{level}", door_to_doctor)
        sim.metrics.observe("within_target", float(door_to_doctor <= TARGET_MINUTES[level]))
        pt.set_state("treatment")
        yield consult[level].sample(s_svc)
        sim.release(doc)
        yield 10.0  # room turnover / discharge paperwork
        sim.release(room)
        sim.dispose(pt)

    def source(sim: Simulation) -> Any:
        while True:
            yield inter.sample(s_arr)
            pt = sim.entity("patient", acuity=acuity.sample(s_acu))
            sim.process(patient(sim, pt), entity=pt)

    sim.process(source(sim), name="arrivals")


model = emergency_department

METRICS = [
    "door_to_doctor.acuity1.mean",
    "door_to_doctor.acuity2.mean",
    "door_to_doctor.acuity3.mean",
    "within_target.mean",
    "resource.doctor.utilization",
    "resource.room.utilization",
]


def main(replications: int = 10) -> None:
    base = Scenario("baseline", {})
    scenarios = [base, base.derive("extra_doctor", doctors=4), base.derive("more_rooms", rooms=11)]
    res = Experiment(emergency_department, scenarios, replications=replications, seed=7).run()
    print(res.format_summary(METRICS))
    print("\nWhich change helps more? (paired CIs)")
    print(
        compare(
            res, "baseline", metrics=["within_target.mean", "door_to_doctor.acuity3.mean"]
        ).format()
    )


if __name__ == "__main__":
    main()
