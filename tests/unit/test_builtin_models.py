from __future__ import annotations

import json
import math
import pickle

import pytest

from simulsi import Experiment, Model, Parameter
from simulsi.cli import main
from simulsi.errors import ConfigError
from simulsi.models import BUILTIN_MODELS, traffic_signal, webster_delay

SHORT = {
    "mmc": 2_000.0,
    "airline": None,
    "epidemic": None,
    "supply_chain": None,
    "ride_hailing": None,
    "cloud_autoscaling": 600.0,
    "traffic_signal": 1_800.0,
}


def _mean(model: Model, params: dict[str, object], metric: str, reps: int = 4) -> float:
    return model.evaluate(params, metric=metric, replications=reps, seed=3)


@pytest.mark.parametrize("name", sorted(BUILTIN_MODELS))
def test_builtin_model_runs_and_reports_outputs(name: str) -> None:
    model = BUILTIN_MODELS[name]
    if SHORT[name] is not None:
        model = model.with_options(duration=SHORT[name], warmup=SHORT[name] / 10)
    res = model.simulate(seed=1)
    assert model.outputs and model.presets
    for out in model.outputs:
        assert out in res.metrics, out
    assert not [w for w in res.warnings if "blocked" in w]
    assert pickle.loads(pickle.dumps(model)).name == model.name
    desc = model.describe()
    assert desc["presets"] == model.presets
    scenarios = model.preset_scenarios()
    assert scenarios[0].name == "baseline" and len(scenarios) == len(model.presets) + 1


def test_presets_are_validated_and_preserved() -> None:
    def build(sim: object, p: object) -> None:
        return None

    with pytest.raises(ConfigError, match="preset 'bad'"):
        Model(
            build,
            duration=1,
            parameters=[Parameter("x", 1, "int", low=0)],
            presets={"bad": {"x": -1}},
        )
    m = Model(build, duration=1, parameters={"x": 1}, presets={"two": {"x": 2}})
    assert m.with_options(duration=5).presets == {"two": {"x": 2}}


def test_airline_schedule_buffer_and_gates() -> None:
    m = BUILTIN_MODELS["airline"]
    tight = _mean(m, {"schedule_buffer": 0.0}, "otp")
    padded = _mean(m, {"schedule_buffer": 30.0}, "otp")
    assert padded > tight + 0.2
    assert _mean(m, {"schedule_buffer": 0.0}, "reactionary_share") > _mean(
        m, {"schedule_buffer": 30.0}, "reactionary_share"
    )
    assert _mean(m, {"gates": 3, "schedule_buffer": 30.0}, "resource.gate.wait.mean") > 5


def test_epidemic_vaccination_and_r0() -> None:
    m = BUILTIN_MODELS["epidemic"]
    assert _mean(m, {"vaccination": 0.6}, "attack_rate") < 0.1
    assert _mean(m, {"r0": 1.3}, "attack_rate") < _mean(m, {"r0": 4.0}, "attack_rate")
    assert _mean(m, {"lockdown_trigger": 2.0}, "people.I.max") > _mean(
        m, {"lockdown_trigger": 0.3}, "people.I.max"
    )


def test_supply_chain_bullwhip() -> None:
    m = BUILTIN_MODELS["supply_chain"]
    base = _mean(m, {}, "bullwhip.factory")
    assert base > _mean(m, {}, "bullwhip.retailer") > 1  # amplification upstream
    assert _mean(m, {"information_sharing": True}, "bullwhip.factory") < 0.7 * base
    assert _mean(m, {"lead_time": 5}, "bullwhip.retailer") > _mean(m, {}, "bullwhip.retailer")


def test_ride_hailing_fleet_and_surge() -> None:
    m = BUILTIN_MODELS["ride_hailing"]
    assert _mean(m, {"drivers": 90}, "service_level", 2) > _mean(
        m, {"drivers": 40}, "service_level", 2
    )
    surge = m.simulate({"surge": True}, seed=2).metrics
    plain = m.simulate({}, seed=2).metrics
    assert surge["surge.mean"] > 1 and surge["price_declined"] > 0
    assert surge["driver_earnings_per_hour"] > plain["driver_earnings_per_hour"]


def test_cloud_autoscaling_reacts_to_load() -> None:
    m = BUILTIN_MODELS["cloud_autoscaling"].with_options(duration=900.0, warmup=60.0)
    r = m.simulate({"burst_multiplier": 4.0, "mean_time_between_bursts": 100.0}, seed=1).metrics
    assert r["scale_ups"] > 0 and r["servers.max"] > 4
    starved = m.simulate(
        {"max_servers": 4, "burst_multiplier": 4.0, "mean_time_between_bursts": 100.0}, seed=1
    ).metrics
    assert starved["error_rate"] > r["error_rate"]


def test_traffic_signal_matches_webster_and_actuated_helps() -> None:
    m = traffic_signal
    sim_delay = _mean(m, {}, "delay.ns.mean")
    webster = m.simulate(seed=1).metrics["webster.delay_ns"]
    assert sim_delay == pytest.approx(webster, rel=0.25)
    assert _mean(m, {"cycle": 150.0}, "delay.mean") > _mean(m, {}, "delay.mean")
    assert _mean(m, {"control": "actuated"}, "delay.mean") < _mean(m, {}, "delay.mean")
    assert math.isnan(webster_delay(2000, 20, 60, 2.0))
    assert webster_delay(0, 20, 60, 2.0) == 0.0


def test_presets_experiment_with_workers() -> None:
    m = BUILTIN_MODELS["supply_chain"]
    res = Experiment(m, m.preset_scenarios(), replications=2, workers=2).run()
    assert len(res.scenarios) == len(m.presets) + 1 and not res.errors


def test_models_and_whatif_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models"]) == 0
    assert "builtin:airline" in capsys.readouterr().out
    assert main(["models", "epidemic"]) == 0
    out = capsys.readouterr().out
    assert "Presets" in out and "vaccinated_60pct" in out and "Gillespie" in out
    assert main(["models", "nope"]) != 0
    capsys.readouterr()
    assert (
        main(
            [
                "whatif",
                "builtin:traffic_signal",
                "--vary",
                "cycle=40,90",
                "-r",
                "2",
                "--duration",
                "1200",
                "-m",
                "delay.mean",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "cycle=40" in out and "cycle=90" in out and "#" in out
    assert main(["whatif", "builtin:supply_chain", "--presets", "-r", "2", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["scenario"] for r in rows} >= {"baseline", "shared_pos_data"}
    assert main(["whatif", "builtin:mmc"]) != 0
    assert main(["whatif", "builtin:mmc", "--vary", "servers"]) != 0
