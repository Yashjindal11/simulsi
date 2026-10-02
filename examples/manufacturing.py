"""Example 4 - Manufacturing line with breakdowns.

Jobs flow through two machines in series with a finite buffer between them.
Machines break down at random (exponential time between failures) and need a
technician to repair them; a breakdown interrupts the job, which resumes
when the machine is back. A full buffer blocks machine 1 (starvation and
blocking both appear). A cost model turns the output into money.

    python examples/manufacturing.py
"""

from __future__ import annotations

from typing import Any

from simulsi import Experiment, Interrupt, Parameter, Scenario, Simulation, compare, model
from simulsi.cost import CostModel
from simulsi.processes import FailureProcess, RecoveryProcess
from simulsi.randomness import Exponential, LogNormal, Triangular


@model(
    duration=5 * 24 * 60.0,  # five days, minutes
    warmup=8 * 60.0,
    version="1",
    parameters=[
        Parameter("jobs_per_hour", 11.0, "float", low=0.0),
        Parameter("buffer_size", 5, "int", low=1),
        Parameter("mtbf_hours", 10.0, "float", low=0.1, description="mean time between failures"),
        Parameter("technicians", 1, "int", low=1),
        Parameter("mean_repair_minutes", 45.0, "float", low=1.0),
    ],
)
def line(sim: Simulation, p: Any) -> None:
    m1, m2 = sim.resource("m1", 1), sim.resource("m2", 1)
    techs = sim.resource("technician", p.technicians)
    buffer = sim.queue("buffer", capacity=p.buffer_size)
    s_arr, s_proc = sim.stream("arrivals"), sim.stream("processing")
    t1, t2 = Triangular(3.0, 4.5, 6.0), Triangular(3.5, 4.8, 6.5)
    repair = LogNormal.from_moments(p.mean_repair_minutes, 0.5 * p.mean_repair_minutes)
    for m in (m1, m2):
        FailureProcess(
            sim,
            m,
            Exponential(mean=p.mtbf_hours * 60),
            RecoveryProcess(repair, crew=techs),
            interrupt=True,
        )

    def operate(sim: Simulation, machine: Any, minutes: float) -> Any:
        """Process for `minutes` of machine time, surviving breakdowns.

        Returns the machine request still held, so the caller decides when to
        release it (machine 1 stays occupied while blocked by a full buffer).
        """
        remaining = minutes
        while True:
            req = yield sim.request(machine)
            start = sim.now
            try:
                yield remaining
                return req
            except Interrupt:
                remaining -= sim.now - start
                sim.metrics.increment("jobs.interrupted")
                machine.release(req)

    def job(sim: Simulation, j: Any) -> Any:
        j.set_state("m1")
        req = yield from operate(sim, m1, t1.sample(s_proc))
        j.set_state("blocked")
        put = buffer.put(j)
        while not put.triggered:
            try:
                yield put  # m1 is blocked until the buffer has space
            except Interrupt:
                pass  # a breakdown while blocked does not affect the finished job
        m1.release(req)
        j.set_state("buffer")

    def stage2(sim: Simulation) -> Any:
        while True:
            j = yield buffer.get()
            j.set_state("m2")
            req = yield from operate(sim, m2, t2.sample(s_proc))
            m2.release(req)
            sim.metrics.increment("jobs.completed")
            sim.dispose(j)
            sim.metrics.record("wip", len(sim.entities))

    def source(sim: Simulation) -> Any:
        while True:
            yield s_arr.exponential(60 / p.jobs_per_hour)
            j = sim.entity("job")
            sim.metrics.record("wip", len(sim.entities))
            sim.process(job(sim, j), entity=j)

    sim.process(source(sim), name="arrivals")
    sim.process(stage2(sim), name="stage2")


model = line

COSTS = (
    CostModel()
    .revenue("sales", "jobs.completed", 120.0)
    .variable("wip_holding", "wip.mean", 25.0 * 5)  # per job in WIP per day, over 5 days
    .resource("technician", per_capacity_time=0.75)  # wage per minute
    .variable("repairs", "failure.m1.count", 80.0)
    .variable("repairs_m2", "failure.m2.count", 80.0)
)

METRICS = [
    "jobs.completed",
    "wip.mean",
    "entity.job.time_in_system.mean",
    "resource.m1.availability",
    "resource.m2.availability",
    "queue.buffer.mean_length",
    "cost.profit",
]


def main(replications: int = 10) -> None:
    base = Scenario("baseline", {})
    scenarios = [
        base,
        base.derive("preventive_maintenance", mtbf_hours=16.0),
        base.derive("bigger_buffer", buffer_size=12),
        base.derive("second_technician", technicians=2),
    ]
    res = Experiment(line, scenarios, replications=replications, seed=3).run()
    res.derive(COSTS.metrics)
    print(res.format_summary(METRICS))
    print()
    print(compare(res, "baseline", metrics=["jobs.completed", "cost.profit"]).format())


if __name__ == "__main__":
    main()
