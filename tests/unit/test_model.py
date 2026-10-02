from __future__ import annotations

import pickle
from typing import Any

import pytest

from simulsi import ConfigError, Model, Parameter, Scenario, Simulation, grid, model
from simulsi.models import erlang_c, mmc
from simulsi.randomness import Exponential
from simulsi.scenarios import one_at_a_time


def test_scenario_derive_and_scale() -> None:
    base = Scenario("baseline", {"rate": 1.0, "capacity": 3})
    hi = base.derive("high", rate=2.0)
    assert hi.parameters == {"rate": 2.0, "capacity": 3}
    assert base.parameters["rate"] == 1.0
    assert base.scale("x", capacity=1.5).parameters["capacity"] == 4  # stays int (rounded)
    with pytest.raises(ConfigError):
        base.scale("bad", missing=2)
    assert Scenario.from_dict(hi.to_dict()) == hi


def test_grid_and_one_at_a_time() -> None:
    g = grid({"capacity": [1, 2], "rate": [0.5, 1.0]})
    assert [s.name for s in g] == [
        "capacity=1,rate=0.5",
        "capacity=1,rate=1",
        "capacity=2,rate=0.5",
        "capacity=2,rate=1",
    ]
    with pytest.raises(ConfigError):
        grid({"capacity": []})
    oat = one_at_a_time(Scenario("b", {"a": 1, "b": 2}), {"a": [0, 2]})
    assert [s.parameters for s in oat] == [{"a": 0, "b": 2}, {"a": 2, "b": 2}]


def test_parameter_coercion_and_bounds() -> None:
    p = Parameter("n", 1, "int", low=1)
    assert p.coerce(3.0) == 3
    for bad in (0, 2.5, "3", True):
        with pytest.raises(ConfigError):
            p.coerce(bad)
    prob = Parameter("p", 0.1, "probability")
    with pytest.raises(ConfigError):
        prob.coerce(1.2)
    d = Parameter("svc", Exponential(rate=1), "distribution")
    assert d.coerce({"distribution": "exponential", "mean": 2}) == Exponential(rate=0.5)
    with pytest.raises(ConfigError):
        d.coerce({"distribution": "nope"})
    assert Parameter.infer("x", 2.0).kind == "float"
    assert Parameter.infer("x", True).kind == "bool"


def test_model_resolve_errors() -> None:
    m = Model(
        lambda sim, p: None, name="m", duration=1, parameters=[Parameter("a"), Parameter("b", 2)]
    )
    issues = m.check_parameters({"zzz": 1})
    assert any("unknown parameter 'zzz'" in i for i in issues)
    assert any("missing required parameter 'a'" in i for i in issues)
    assert m.resolve({"a": 1}) == {"a": 1, "b": 2}
    with pytest.raises(ConfigError):
        m.resolve({})
    with pytest.raises(ConfigError):
        Model(lambda s, p: None, duration=-1)
    with pytest.raises(ConfigError):
        Model(lambda s, p: None, duration=10, warmup=10)


def test_params_are_read_only() -> None:
    p = mmc.resolve({"servers": 2})
    assert p.servers == 2 and p["servers"] == 2
    with pytest.raises(AttributeError):
        p.servers = 3
    with pytest.raises(AttributeError):
        _ = p.nothing
    assert pickle.loads(pickle.dumps(p)) == p


@model(duration=50, parameters={"servers": 1, "rate": 1.0})
def tiny(sim: Simulation, p: Any) -> None:
    r = sim.resource("r", p.servers)
    st = sim.stream("a")

    def src(sim: Simulation) -> Any:
        while True:
            yield st.exponential(1 / p.rate)
            sim.process(sim.use(r, 0.5))

    sim.process(src(sim))


def test_model_decorator_and_simulate() -> None:
    assert isinstance(tiny, Model) and tiny.name == "tiny"
    r1 = tiny.simulate(seed=3)
    r2 = tiny.simulate(seed=3)
    assert r1.metrics == r2.metrics
    assert r1.end_time == 50
    assert r1.details["parameters"] == {"servers": 1, "rate": 1.0}
    assert pickle.loads(pickle.dumps(tiny)) is tiny


def test_model_evaluate_uses_common_random_numbers() -> None:
    a = tiny.evaluate({"servers": 1}, metric="resource.r.wait.mean", replications=3)
    b = tiny.evaluate({"servers": 2}, metric="resource.r.wait.mean", replications=3)
    assert b < a
    with pytest.raises(KeyError):
        tiny.evaluate(metric="nope", replications=1)


def test_builtin_mmc_and_erlang_c() -> None:
    theory = erlang_c(0.9, 1.0, 1)
    assert theory["mean_wait"] == pytest.approx(9.0)
    assert theory["utilization"] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        erlang_c(2, 1, 1)
    m = mmc.simulate({"arrival_rate": 0.5}, seed=1)
    assert m.metrics["resource.server.utilization"] == pytest.approx(0.5, abs=0.05)
    m2 = pickle.loads(pickle.dumps(mmc))
    assert m2.simulate({"arrival_rate": 0.5}, seed=1).metrics == m.metrics
