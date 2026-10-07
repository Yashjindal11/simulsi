"""Hospital emergency department: triage, acuity, beds, doctors and boarding.

Patients arrive by walk-in (with a daily peak) or ambulance, are triaged by
a nurse and given an acuity level (1 = most urgent). Treatment needs an ED
bed and a doctor; more urgent patients are served first. Low-acuity patients
may leave without being seen if they wait too long, or go to a *fast track*
clinician when one is available. Admitted patients keep their ED bed until a
ward bed frees up - *boarding* - which blocks the department. When the
waiting room is very full, ambulances are diverted.

Try: ``doctors`` 3 vs 5, ``ward_beds`` 75 vs 100, ``fast_track`` false vs
true, ``arrival_multiplier`` 1 vs 1.4, ``diversion_threshold`` 10 vs 1000.

Time unit: minutes; three days with the first day as warm-up. Synthetic.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.schedules import PiecewiseRate, arrivals
from simulsi.randomness.distributions import Categorical, LogNormal, Triangular

DAY = 24 * 60.0
ACUITY = Categorical({1: 0.03, 2: 0.17, 3: 0.40, 4: 0.30, 5: 0.10})


def _build(sim: Simulation, p: Params) -> None:
    nurses = sim.resource("triage_nurse", p.triage_nurses)
    beds = sim.resource("ed_bed", p.ed_beds, discipline="priority")
    doctors = sim.resource("doctor", p.doctors, discipline="priority")
    fast = sim.resource("fast_track", 1 if p.fast_track else 0)
    ward = sim.resource("ward_bed", p.ward_beds)
    rs = sim.stream("patients")
    base = 9.0 / 60 * p.arrival_multiplier  # walk-ins per minute at the daily average
    walk_in = PiecewiseRate(
        [(0, 0.4 * base), (8 * 60, 1.5 * base), (14 * 60, 1.3 * base), (22 * 60, 0.6 * base)],
        period=DAY,
    )
    treat = {
        a: LogNormal.from_moments(m, 0.6 * m)
        for a, m in {1: 120, 2: 90, 3: 60, 4: 25, 5: 15}.items()
    }
    assess = {
        a: LogNormal.from_moments(m, 0.5 * m) for a, m in {1: 40, 2: 30, 3: 20, 4: 12, 5: 8}.items()
    }
    ward_stay = LogNormal.from_moments(p.ward_stay_days * DAY, 0.5 * p.ward_stay_days * DAY)
    admit_prob = {1: 0.8, 2: 0.5, 3: 0.2, 4: 0.05, 5: 0.01}
    m = sim.metrics
    for name in ("patients", "lwbs", "diverted", "admitted"):
        m.counter(name)

    def ward_stay_proc(req: Any) -> Any:
        yield ward_stay.sample(rs)
        ward.release(req)

    def patient(sim: Simulation, i: int, ambulance: bool = False) -> Any:
        m.increment("patients")
        t0 = sim.now
        if not ambulance:
            req = yield sim.request(nurses)
            yield rs.triangular(3, 5, 10)
            nurses.release(req)
        acuity = ACUITY.sample(rs) if not ambulance else min(3, ACUITY.sample(rs))
        if p.fast_track and acuity >= 4 and fast.capacity > 0:
            req = yield sim.request(fast, patience=p.lwbs_patience)
            if not req.granted:
                m.increment("lwbs")
                return
            m.observe("door_to_doctor", sim.now - t0)
            yield assess[acuity].sample(rs) + treat[acuity].sample(rs) / 2
            fast.release(req)
            m.observe("length_of_stay", sim.now - t0)
            return
        patience = p.lwbs_patience if acuity >= 4 else None
        bed = yield sim.request(beds, priority=acuity, patience=patience)
        if not bed.granted:
            m.increment("lwbs")
            return
        doc = yield sim.request(doctors, priority=acuity)
        m.observe("door_to_doctor", sim.now - t0)
        m.observe(f"door_to_doctor.acuity{acuity}", sim.now - t0)
        yield assess[acuity].sample(rs)
        doctors.release(doc)
        yield treat[acuity].sample(rs)  # tests, treatment and observation in the bed
        if rs.random() < admit_prob[acuity]:
            m.increment("admitted")
            t_board = sim.now
            wreq = yield sim.request(ward)
            m.observe("boarding_minutes", sim.now - t_board)
            sim.process(ward_stay_proc(wreq), name="ward-stay")
        beds.release(bed)
        m.observe("length_of_stay", sim.now - t0)

    def ambulances() -> Any:
        while True:
            yield rs.exponential(60 / (p.ambulances_per_hour * p.arrival_multiplier))
            if beds.queue_size >= p.diversion_threshold:
                m.increment("diverted")
                continue
            sim.process(patient(sim, 0, ambulance=True))

    # The ward starts realistically full: occupied beds with staggered discharges.
    def occupied_bed(after: float) -> Any:
        req = yield sim.request(ward)
        yield after
        ward.release(req)

    for _ in range(round(0.85 * p.ward_beds)):
        sim.process(
            occupied_bed(Triangular(0, 0.5, 1).sample(rs) * p.ward_stay_days * DAY),
            name="ward-preload",
        )
    arrivals(sim, walk_in, patient, stream="walk-ins")
    sim.process(ambulances(), name="ambulances")


emergency_department = Model(
    _build,
    name="emergency_department",
    duration=3 * DAY,
    warmup=DAY,
    version="1",
    description="Emergency department: triage, acuity priority, beds, doctors, LWBS, boarding and diversion.",
    parameters=[
        Parameter("triage_nurses", 2, "int", low=1),
        Parameter("ed_beds", 20, "int", low=1),
        Parameter("doctors", 4, "int", low=1),
        Parameter("fast_track", False, "bool", description="separate clinician for acuity 4-5"),
        Parameter(
            "ward_beds", 85, "int", low=1, description="inpatient beds for admitted patients"
        ),
        Parameter("ward_stay_days", 1.5, "float", low=0.1, unit="days"),
        Parameter("arrival_multiplier", 1.0, "float", low=0),
        Parameter("ambulances_per_hour", 1.5, "float", low=0),
        Parameter(
            "lwbs_patience",
            120.0,
            "float",
            low=0,
            unit="min",
            description="how long low-acuity patients wait before leaving",
        ),
        Parameter(
            "diversion_threshold",
            12,
            "int",
            low=0,
            description="patients waiting for a bed at which ambulances are diverted",
        ),
    ],
    outputs=[
        "door_to_doctor.mean",
        "length_of_stay.mean",
        "lwbs",
        "diverted",
        "boarding_minutes.mean",
        "resource.ed_bed.utilization",
        "resource.doctor.utilization",
    ],
    presets={
        "extra_doctor": {"doctors": 5},
        "more_ward_beds": {"ward_beds": 100},
        "ward_crunch": {"ward_beds": 75},
        "fast_track": {"fast_track": True},
        "surge": {"arrival_multiplier": 1.4},
        "surge_with_fast_track": {"arrival_multiplier": 1.4, "fast_track": True},
        "no_diversion": {"diversion_threshold": 1000},
    },
)
