from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simulsi import ConfigError, Experiment, Model, Scenario, Simulation, model, monte_carlo
from simulsi.analysis import correlation_sensitivity, finite_difference, one_at_a_time
from simulsi.cost import CostModel
from simulsi.introspection import flow_graph, model_graph, state_graph
from simulsi.models import mmc
from simulsi.optimization import Objective
from simulsi.processes import (
    FailureProcess,
    RecoveryProcess,
    ScheduledDisruption,
    capacity_reduction,
)
from simulsi.randomness import Constant, Exponential, Uniform
from simulsi.validation import validate_model

SHORT_MMC = Model(
    mmc.build,
    name="mmc_short",
    duration=400.0,
    warmup=40.0,
    parameters=list(mmc.parameters.values()),
    sim_options=mmc.sim_options,
)


# -- disruptions -----------------------------------------------------------------


def test_failure_process_downtime_and_availability() -> None:
    sim = Simulation(seed=1)
    m = sim.resource("machine", 2)
    FailureProcess(sim, m, Constant(10), Constant(5))
    res = sim.run(until=99)
    # up 10, down 5, repeating: failures at 10, 25, 40, 55, 70, 85
    assert res.metrics["failure.machine.count"] == 6
    assert res.metrics["failure.machine.downtime.mean"] == 5
    assert res.metrics["resource.machine.availability"] == pytest.approx(1 - 30 / 99)


def test_recovery_needs_crew() -> None:
    sim = Simulation(seed=1)
    a, b = sim.resource("a", 1), sim.resource("b", 1)
    crew = sim.resource("crew", 1)
    FailureProcess(sim, a, Constant(10), RecoveryProcess(Constant(4), crew=crew), name="a")
    FailureProcess(sim, b, Constant(10), RecoveryProcess(Constant(4), crew=crew), name="b")
    res = sim.run(until=13)
    # both fail at t=10; one crew -> second repair waits
    assert res.metrics["resource.crew.wait.max"] == 0  # second request still waiting at t=13
    sim.run(until=30)
    assert crew.wait_times.max == 4


def test_scheduled_disruptions() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("gate", 3)
    capacity_reduction(sim, r, at=10, duration=5, capacity=1)
    state = {"rate": 1.0}
    ScheduledDisruption(
        sim,
        at=20,
        duration=10,
        apply=lambda: state.update(rate=3.0),
        revert=lambda: state.update(rate=1.0),
        name="spike",
    )
    sim.run(until=12)
    assert r.capacity == 1
    sim.run(until=25)
    assert r.capacity == 3 and state["rate"] == 3.0
    res = sim.run(until=40)
    assert state["rate"] == 1.0 and res.metrics["disruption.spike.count"] == 1


# -- cost --------------------------------------------------------------------------


def test_cost_model_terms() -> None:
    cm = (
        CostModel()
        .fixed("rent", 100)
        .resource("server", per_capacity_time=2.0)
        .waiting("server", per_time=0.5)
        .revenue("sales", "resource.server.releases", 3.0)
        .penalty("sla", "resource.server.wait.mean", above=1.0, amount=50, per_unit=10)
    )
    metrics = {
        "resource.server.capacity_time": 10.0,
        "resource.server.wait.total": 8.0,
        "resource.server.releases": 20.0,
        "resource.server.wait.mean": 1.5,
    }
    b = cm.calculate(metrics)
    assert b.items == {
        "rent": 100,
        "server.capacity": 20,
        "server.waiting": 4,
        "sales": 60,
        "sla": 55,
    }
    assert b.total_cost == 179 and b.total_revenue == 60 and b.profit == -119
    assert cm.metrics(metrics)["cost.profit"] == -119
    again = CostModel.from_dict(cm.to_dict())
    assert again.calculate(metrics) == b
    with pytest.raises(ConfigError):
        CostModel.from_dict({"terms": [{"name": "x", "evil": 1}]})
    with pytest.raises(KeyError):
        cm.calculate({})


def test_cost_on_experiment() -> None:
    cm = CostModel().resource("server", per_capacity_time=1.0).waiting("server", per_time=1.0)
    res = Experiment(
        SHORT_MMC, [Scenario("baseline"), Scenario("two", {"servers": 2})], replications=3
    ).run()
    res.derive(cm.metrics)
    rows = {r["scenario"]: r["mean"] for r in res.summary(["cost.total"])}
    assert rows["baseline"] > 0 and rows["two"] > 0


# -- sensitivity -------------------------------------------------------------------


def test_one_at_a_time_and_finite_difference() -> None:
    out = ["resource.server.utilization"]
    oat = one_at_a_time(
        SHORT_MMC, {"arrival_rate": [0.45, 0.9]}, out, base={"arrival_rate": 0.6}, replications=3
    )
    vals = {r.setting: r.value for r in oat.rows}
    assert vals["arrival_rate=0.45"] < 0 < vals["arrival_rate=0.9"]
    fd = finite_difference(
        SHORT_MMC,
        ["arrival_rate", "service_rate"],
        out,
        base={"arrival_rate": 0.6},
        replications=3,
        relative_step=0.1,
    )
    d_lambda = next(r for r in fd.rows if r.parameter == "arrival_rate").value
    d_mu = next(r for r in fd.rows if r.parameter == "service_rate").value
    # utilization = lambda/mu: d/dlambda = 1, d/dmu = -lambda/mu^2 = -0.6
    assert d_lambda == pytest.approx(1.0, abs=0.15)
    assert d_mu == pytest.approx(-0.6, abs=0.15)
    assert fd.ranking()[0].parameter == "arrival_rate"
    assert "arrival_rate" in fd.format()


def test_correlation_sensitivity_ranks_inputs() -> None:
    mc = monte_carlo(
        lambda a, b, c: 5 * a + 0.5 * b + 0 * c,
        {"a": Uniform(0, 1), "b": Uniform(0, 1), "c": Uniform(0, 1)},
        3000,
        seed=1,
    )
    s = correlation_sensitivity(mc)
    top = s.ranking(method="spearman")
    assert [r.parameter for r in top][:2] == ["a", "b"]
    src = {r.parameter: r.value for r in s.rows if r.method == "src"}
    assert src["a"] > 0.9 and abs(src["c"]) < 0.05
    assert s.info["r2"]["value"] > 0.99


# -- optimization interface -----------------------------------------------------------


def test_objective_with_scipy() -> None:
    from scipy.optimize import minimize_scalar

    cm = CostModel().resource("server", per_capacity_time=1.0).waiting("server", per_time=4.0)
    obj = Objective(
        SHORT_MMC,
        None,
        ["servers"],
        fixed={"arrival_rate": 1.8},
        transform=lambda m: cm.calculate(m).total_cost,
        replications=2,
    )
    assert obj.bounds == [(1.0, None)]
    costs = {c: obj([c]) for c in (2, 3, 4, 5)}
    best = min(costs, key=lambda c: costs[c])
    assert best in (3, 4)
    assert obj([0]) == float("inf")  # invalid -> penalty, no exception
    n_runs = len(obj.history)
    obj([3])
    assert len(obj.history) == n_runs  # cached
    res = minimize_scalar(
        lambda x: obj([x]), bounds=(2, 6), method="bounded", options={"maxiter": 8}
    )
    assert 2 <= res.x <= 6
    b = obj.best()
    assert b is not None and b.parameters["servers"] in (3, 4)


# -- validation ------------------------------------------------------------------------


def test_validate_model_detects_problems() -> None:
    assert validate_model(SHORT_MMC).ok
    r = validate_model(SHORT_MMC, {"servers": 0})
    assert not r.ok and r.errors[0].code == "parameter"

    @model(duration=10)
    def leaky(sim: Simulation, p: Any) -> None:
        res = sim.resource("r", 1)
        dead = sim.resource("dead", 0)

        def job(sim: Simulation) -> Any:
            yield sim.request(res)

        def stuck(sim: Simulation) -> Any:
            yield sim.request(dead)

        sim.process(job(sim), name="job")
        sim.process(stuck(sim), name="stuck")

    rep = validate_model(leaky, smoke_duration=5)
    codes = {i.code for i in rep.issues}
    assert {"zero-capacity", "resource-not-released", "possible-deadlock"} <= codes
    assert rep.ok  # all warnings
    assert "warning" in rep.format()

    @model(duration=10)
    def broken(sim: Simulation, p: Any) -> None:
        sim.schedule(time=-5, event_type="bad")

    rep = validate_model(broken)
    assert not rep.ok and rep.errors[0].code == "build-error"
    with pytest.raises(Exception, match="validation failed"):
        rep.raise_for_errors()


def test_validate_expected_states() -> None:
    @model(duration=20)
    def states(sim: Simulation, p: Any) -> None:
        def go(sim: Simulation) -> Any:
            e = sim.entity("job")
            e.set_state("queued")
            yield 1
            e.set_state("weird")
            sim.dispose(e)

        sim.process(go(sim))

    rep = validate_model(states, expected_states={"job": ["queued", "running"]}, smoke_duration=10)
    codes = [(i.code, i.message) for i in rep.issues]
    assert any(c == "unreached-state" and "running" in m for c, m in codes)
    assert any(c == "undeclared-state" and "weird" in m for c, m in codes)


# -- graph ----------------------------------------------------------------------------------


@model(duration=50)
def flow_model(sim: Simulation, p: Any) -> None:
    desk = sim.resource("desk", 1)
    arrivals = sim.stream("a")

    def customer(sim: Simulation, c: Any) -> Any:
        c.set_state("waiting")
        req = yield sim.request(desk)
        c.set_state("served")
        yield 2
        sim.release(req)
        sim.dispose(c)

    def src(sim: Simulation) -> Any:
        while True:
            yield Exponential(rate=0.3).sample(arrivals)
            c = sim.entity("customer")
            sim.process(customer(sim, c), entity=c)

    sim.process(src(sim))


def test_flow_and_state_graphs() -> None:
    g = model_graph(flow_model, seed=1)
    flow = g["flow"]
    assert flow.successors("arrival:customer") == ["wait:desk"]
    assert flow.successors("wait:desk") == ["resource:desk"]
    assert flow.successors("resource:desk") == ["exit:customer"]
    assert "flowchart LR" in flow.to_mermaid() and "->" in flow.to_dot()
    states = g["states"]
    assert "customer:served" in states.successors("customer:waiting")
    assert g["inventory"]["resources"]["desk"]["capacity"] == 1
    json.dumps(flow.to_dict())
    sim = flow_model.create(seed=1)
    sim.run(until=20)
    assert state_graph(sim).kind == "state"
    assert flow_graph([]).edges == []


# -- event log export ----------------------------------------------------------------------


def test_event_log_export(tmp_path: Path) -> None:
    sim = flow_model.create(seed=2, trace=True)
    res = sim.run(until=30)
    assert res.log is not None and len(res.log) > 0
    for suffix in ("json", "csv", "parquet"):
        p = res.log.export(tmp_path / f"log.{suffix}")
        assert p.stat().st_size > 0
    import pandas as pd

    df = pd.read_parquet(tmp_path / "log.parquet")
    assert {"timestamp", "event_type", "entity", "resource"} <= set(df.columns)
    with pytest.raises(ValueError):
        res.log.export(tmp_path / "log.xml")


def test_series_recorded() -> None:
    res = flow_model.simulate(seed=3)
    assert "resource.desk.busy" in res.series and "resource.desk.queue_length" in res.series
    ts = [t for t, _ in res.series["resource.desk.busy"]]
    assert ts == sorted(ts)
    assert np.all(np.asarray([v for _, v in res.series["resource.desk.busy"]]) <= 1)
