from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from simulsi import ConfigError
from simulsi.cli import main
from simulsi.config import load_experiment, parse_config, resolve_model


def run_cli(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(args)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture()
def project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    code, out, _ = run_cli(["init", str(tmp_path / "proj")], capsys)
    assert code == 0 and "created" in out
    cfg = tmp_path / "proj" / "experiment.yaml"
    cfg.write_text(cfg.read_text().replace("replications: 30", "replications: 4"))
    return tmp_path / "proj"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "simulsi" in capsys.readouterr().out


def test_init_refuses_overwrite(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run_cli(["init", str(project)], capsys)
    assert code == 1 and "--force" in err
    assert run_cli(["init", str(project), "--force"], capsys)[0] == 0


def test_validate_config_and_model(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(["validate", str(project / "experiment.yaml")], capsys)
    assert code == 0 and "configuration OK" in out and "4 scenario(s)" in out
    code, out, _ = run_cli(["validate", str(project / "model.py"), "-p", "servers=0"], capsys)
    assert code == 2 and "parameter" in out


def test_run_model_file_with_params(
    project: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = run_cli(
        [
            "run",
            str(project / "model.py"),
            "-p",
            "servers=3",
            "--seed",
            "1",
            "-m",
            "resource.server.utilization",
        ],
        capsys,
    )
    assert code == 0 and "resource.server.utilization" in out and "service_desk" in out
    code, out, _ = run_cli(["run", "builtin:mmc", "--json", "--duration", "500"], capsys)
    data = json.loads(out)
    assert data["end_time"] == 500 and "resource.server.utilization" in data["metrics"]
    trace = tmp_path / "trace.csv"
    code, _, err = run_cli(
        ["run", str(project / "model.py"), "--trace", str(trace), "--duration", "60"], capsys
    )
    assert code == 0 and trace.exists() and "log records" in err


def test_run_is_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    a = run_cli(["run", "builtin:mmc", "--json", "--seed", "5", "--duration", "300"], capsys)[1]
    b = run_cli(["run", "builtin:mmc", "--json", "--seed", "5", "--duration", "300"], capsys)[1]
    assert json.loads(a)["metrics"] == json.loads(b)["metrics"]


def test_experiment_analyze_visualize(
    project: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir = tmp_path / "results"
    code, out, _ = run_cli(
        ["experiment", str(project / "experiment.yaml"), "-o", str(out_dir), "-q"], capsys
    )
    assert code == 0, out
    assert "Comparison with 'baseline'" in out and "cost.total" in out
    assert (out_dir / "experiment.json").exists() and (out_dir / "replications.csv").exists()

    code, out, _ = run_cli(
        ["analyze", str(out_dir), "-m", "resource.server.wait.mean", "--precision", "0.1"], capsys
    )
    assert code == 0 and "Replication analysis" in out and "extra_server" in out
    code, out, _ = run_cli(
        ["analyze", str(out_dir), "--json", "-m", "resource.server.utilization"], capsys
    )
    payload = json.loads(out)
    assert payload["comparison"] and payload["metadata"]["replications"] == 4

    plots = tmp_path / "plots"
    code, out, _ = run_cli(
        ["visualize", str(out_dir), "-o", str(plots), "-m", "resource.server.wait.mean"], capsys
    )
    assert code == 0 and (plots / "comparison.png").exists()
    code, out, _ = run_cli(
        ["visualize", "builtin:mmc", "-o", str(plots), "--backend", "plotly"], capsys
    )
    assert code == 0 and (plots / "utilization.html").exists()


def test_experiment_checkpoint_and_overrides(
    project: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    ck = tmp_path / "ck.jsonl"
    args = [
        "experiment",
        str(project / "experiment.yaml"),
        "-r",
        "2",
        "-o",
        str(tmp_path / "o"),
        "-q",
        "--checkpoint",
        str(ck),
    ]
    assert run_cli(args, capsys)[0] == 0
    assert len(ck.read_text().splitlines()) == 1 + 4 * 2
    assert run_cli(args, capsys)[0] == 0


def test_benchmark_small(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run_cli(
        [
            "benchmark",
            "--sizes",
            "2000",
            "--workers",
            "1",
            "--replications",
            "2",
            "--no-memory",
            "--json",
        ],
        capsys,
    )
    data = json.loads(out)
    assert code == 0 and {r["name"] for r in data["results"]} >= {"events", "processes"}
    assert all(r["events_per_second"] > 0 for r in data["results"])


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: builtin:mmc\nexperiment: {replications: 0}\n")
    code, _, err = run_cli(["experiment", str(bad)], capsys)
    assert code == 2 and "replications" in err
    code, _, err = run_cli(["run", "builtin:nope"], capsys)
    assert code == 2 and "unknown builtin" in err
    code, _, err = run_cli(["run", "builtin:mmc", "-p", "novalue"], capsys)
    assert code == 2 and "key=value" in err
    code, _, err = run_cli(["analyze", str(tmp_path / "missing")], capsys)
    assert code == 2


# -- configuration security and schema --------------------------------------------


def test_config_rejects_unknown_keys_and_bad_values() -> None:
    with pytest.raises(ConfigError, match="simulaton: Extra inputs"):
        parse_config({"model": "builtin:mmc", "simulaton": {}})
    with pytest.raises(ConfigError, match="greater than 0"):
        parse_config({"model": "builtin:mmc", "simulation": {"duration": -1}})
    with pytest.raises(ConfigError):
        parse_config({"model": ""})


def test_yaml_is_loaded_safely(tmp_path: Path) -> None:
    evil = tmp_path / "evil.yaml"
    evil.write_text("model: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_experiment(evil)


def test_output_directory_must_stay_inside_config_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg = cfg_dir / "exp.yaml"
    cfg.write_text(
        "model: builtin:mmc\nsimulation: {duration: 100, warmup: 0}\nexperiment: {replications: 1}\n"
        "output: {directory: ../escaped}\n"
    )
    code, _, err = run_cli(["experiment", str(cfg), "-q"], capsys)
    assert code == 2 and "outside the configuration directory" in err
    assert not (tmp_path / "escaped").exists()


def test_model_path_traversal_blocked(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("raise RuntimeError('must not be imported')\n")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg = cfg_dir / "exp.yaml"
    cfg.write_text("model: ../outside.py\n")
    with pytest.raises(ConfigError, match="outside the configuration directory"):
        load_experiment(cfg)


def test_resolve_model_references(tmp_path: Path) -> None:
    assert resolve_model("builtin:mmc").name == "mmc"
    assert resolve_model("simulsi.models.queueing:mmc").name == "mmc"
    with pytest.raises(ConfigError):
        resolve_model("os; rm -rf /")
    with pytest.raises(ConfigError, match="not a simulsi Model"):
        resolve_model("simulsi.models.queueing:erlang_c")
    two = tmp_path / "two.py"
    two.write_text(
        "from simulsi.models import mmc\na = mmc\nb = mmc.with_options(duration=5, warmup=0)\n"
    )
    with pytest.raises(ConfigError, match="found 2 Model"):
        resolve_model(str(two))
    assert resolve_model(f"{two}:b").duration == 5


def test_file_model_runs_in_worker_processes(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "from simulsi import model\n"
        "@model(duration=50, parameters={'k': 1})\n"
        "def m(sim, p):\n"
        "    sim.on_finish(lambda s: s.metrics.set('k2', p.k * 2 + s.stream('x').random()))\n"
    )
    cfg = tmp_path / "e.yaml"
    cfg.write_text(
        "model: m.py\nexperiment: {replications: 4, workers: 2}\nscenarios: [{name: big, parameters: {k: 5}}]\n"
    )
    exp, _ = load_experiment(cfg)
    par = exp.run()
    exp.workers = 1
    ser = exp.run()
    assert par.values("k2", "big").tolist() == ser.values("k2", "big").tolist()


def test_experiment_until_precision(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "e.yaml"
    cfg.write_text(
        "model: builtin:mmc\nsimulation: {duration: 600, warmup: 60}\n"
        "experiment: {replications: 3, max_replications: 40}\n"
        "parameters: {arrival_rate: 0.5}\nmetrics: [resource.server.utilization]\n"
    )
    code, out, _ = run_cli(["experiment", str(cfg), "-q", "--until-precision", "0.05"], capsys)
    assert code == 0 and "stopped after" in out and "precision reached" in out
