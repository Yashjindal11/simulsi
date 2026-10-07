from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from simulsi import Experiment
from simulsi.cli import main
from simulsi.config.schema import resolve_model
from simulsi.errors import ConfigError
from simulsi.flowchart import is_flowchart, load_flowchart

ROOT = Path(__file__).resolve().parents[2]
CLINIC = ROOT / "examples" / "flowcharts" / "clinic.yaml"


def _flow(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": "shop",
        "duration": 500,
        "parameters": {"clerks": 1, "gap": 2.0},
        "resources": {"clerk": "$clerks"},
        "sources": [
            {
                "name": "customer",
                "interarrival": {"distribution": "exponential", "mean": "$gap"},
                "next": "till",
            }
        ],
        "stations": {"till": {"resource": "clerk", "service": 1.5, "next": "exit"}},
    }
    spec.update(overrides)
    return spec


def test_example_clinic_runs_and_has_presets() -> None:
    m = load_flowchart(CLINIC)
    assert m.name == "clinic" and set(m.presets) == {
        "extra_doctor",
        "flu_season",
        "flu_season_staffed",
    }
    r = m.simulate(seed=1).metrics
    assert r["entity.patient.time_in_system.mean"] > 0
    assert r["station.triage.visits"] > 50 and r["sink.exit"] > 0
    lab_share = r["station.lab.visits"] / r["station.consult.visits"]
    assert 0.15 < lab_share < 0.45
    busier = m.simulate({"arrival_mean": 3.0}, seed=1).metrics
    assert busier["resource.doctor.utilization"] > r["resource.doctor.utilization"]


def test_queue_matches_deterministic_service_and_parameters() -> None:
    m = load_flowchart({"flow": _flow()})
    one = m.evaluate({"clerks": 1}, metric="resource.clerk.wait.mean", replications=3)
    two = m.evaluate({"clerks": 2}, metric="resource.clerk.wait.mean", replications=3)
    assert one > two >= 0
    with pytest.raises(ConfigError):
        m.simulate({"tellers": 2})


def test_routing_rest_patience_units_delay_and_rate_table() -> None:
    spec = _flow(
        resources={"clerk": 1, "pair": {"capacity": 4, "schedule": [[0, 4], [250, 2]]}},
        sources=[
            {"name": "a", "rate_table": [[0, 0.3], [100, 0.6]], "period": 200, "next": "split"}
        ],
        stations={
            "split": {"delay": 0.5, "next": {"till": 0.25, "duo": "rest"}},
            "till": {"resource": "clerk", "service": 6.0, "patience": 1.0, "next": "exit"},
            "duo": {"resource": "pair", "service": 1.0, "units": 2, "next": "done"},
        },
        sinks=["done"],
    )
    r = load_flowchart(spec).simulate(seed=2).metrics
    till, duo = r["station.till.visits"], r["station.duo.visits"]
    assert 0.15 < till / (till + duo) < 0.35
    assert r["station.till.abandoned"] > 0 and r["sink.abandoned"] == r["station.till.abandoned"]
    assert r["sink.done"] > 0 and r["route.split.duo"] == duo


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sources": []}, "at least one source"),
        (
            {"stations": {"till": {"resource": "clerk", "service": 1, "next": "nowhere"}}},
            "unknown destination",
        ),
        (
            {"stations": {"till": {"resource": "ghost", "service": 1, "next": "exit"}}},
            "unknown resource",
        ),
        ({"stations": {"till": {"resource": "clerk", "next": "exit"}}}, "needs 'service'"),
        ({"stations": {"till": {"next": "exit"}}}, "'delay'"),
        ({"stations": {"till": {"delay": 1, "next": "exit", "colour": "red"}}}, "unknown key"),
        ({"sinks": ["till"]}, "both station and sink"),
        ({"colour": 1}, "unknown key"),
        ({"sources": [{"name": "x", "rate": 1, "interarrival": 2, "next": "till"}]}, "exactly one"),
        ({"sources": [{"name": "x", "rate": 1}]}, "'next'"),
    ],
)
def test_validation_errors(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_flowchart(_flow(**change))


def test_runtime_errors() -> None:
    bad_ref = load_flowchart(_flow(resources={"clerk": "$missing"}))
    with pytest.raises(ConfigError, match="unknown parameter reference"):
        bad_ref.simulate(seed=1)
    loop = load_flowchart(
        _flow(
            stations={"a": {"delay": 0, "next": "b"}, "b": {"delay": 0, "next": "a"}},
            sources=[{"name": "s", "rate": 1, "limit": 1, "next": "a"}],
        )
    )
    with pytest.raises(ConfigError, match="routing loop"):
        loop.simulate(seed=1)
    with pytest.raises(ConfigError):
        load_flowchart("[1, 2]")
    with pytest.raises(ConfigError):
        load_flowchart({"flow": [1]})


def test_cli_and_configs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert is_flowchart(CLINIC) and not is_flowchart(tmp_path / "none.yaml")
    assert main(["run", str(CLINIC), "--seed", "3"]) == 0
    assert "entity.patient" in capsys.readouterr().out
    assert (
        main(
            [
                "whatif",
                str(CLINIC),
                "--presets",
                "-r",
                "2",
                "-m",
                "entity.patient.time_in_system.mean",
            ]
        )
        == 0
    )
    assert "flu_season" in capsys.readouterr().out
    assert main(["validate", str(CLINIC)]) == 0
    (tmp_path / "shop.yaml").write_text(
        textwrap.dedent("""
        flow:
          name: shop
          duration: 200
          resources: {clerk: 1}
          sources: [{name: c, rate: 0.3, next: till}]
          stations: {till: {resource: clerk, service: 2, next: exit}}
        """)
    )
    (tmp_path / "experiment.yaml").write_text("model: shop.yaml\nexperiment: {replications: 2}\n")
    assert (
        main(["experiment", str(tmp_path / "experiment.yaml"), "-q", "-o", str(tmp_path / "out")])
        == 0
    )
    m = resolve_model("shop.yaml", base_dir=tmp_path)
    assert Experiment(m, replications=2).run().scenarios == ["baseline"]
    (tmp_path / "plain.yaml").write_text("x: 1\n")
    with pytest.raises(ConfigError, match="not a flowchart"):
        resolve_model("plain.yaml", base_dir=tmp_path)
    with pytest.raises(ConfigError, match="outside"):
        resolve_model("../x.yaml", base_dir=tmp_path, allow_outside=False)
