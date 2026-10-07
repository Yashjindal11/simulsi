from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from simulsi import Model, Parameter, Simulation
from simulsi.cli import main
from simulsi.optimization import dominates, pareto_front, pareto_search


def _tradeoff(sim: Simulation, p: Any) -> None:
    noise = sim.stream("n").normal(0, 0.01)
    sim.metrics.set("cost", p.servers * 10 + noise)
    sim.metrics.set("wait", 100 / p.servers + noise)
    sim.metrics.set("quality", -p.servers + noise)


TRADE = Model(
    _tradeoff,
    name="trade",
    duration=1.0,
    parameters=[Parameter("servers", 1, "int", low=1), Parameter("waste", 0.0, "float")],
)


def test_dominance_and_front() -> None:
    assert dominates([1, 5], [2, 5], ["min", "min"])
    assert not dominates([1, 5], [1, 5], ["min", "min"])
    assert dominates([1, 6], [1, 5], ["min", "max"])
    pts = [[1, 10], [2, 5], [3, 6], [math.nan, 0], [2, 5]]
    assert pareto_front(pts, ["min", "min"]) == [0, 1, 4]


def test_grid_ranges_and_scenarios() -> None:
    res = pareto_search(
        TRADE, {"cost": "min", "wait": "min"}, grid={"servers": [1, 2, 4, 8]}, replications=2
    )
    assert len(res.front) == 4  # a pure trade-off: everything is optimal
    res2 = pareto_search(
        TRADE, {"cost": "min", "quality": "max"}, grid={"servers": [1, 2, 4]}, replications=2
    )
    assert [d.parameters["servers"] for d in res2.front] == [1]  # cost and quality agree
    lhs = pareto_search(
        TRADE, {"cost": "min", "wait": "min"}, ranges={"servers": (1, 10)}, budget=8, replications=1
    )
    assert len(lhs.designs) == 8 and all(
        isinstance(d.parameters["servers"], int) for d in lhs.designs
    )
    sc = pareto_search(
        TRADE,
        {"cost": "min"},
        scenarios={"one": {"servers": 1}, "two": {"servers": 2}},
        fixed={"waste": 1.0},
    )
    assert [d.name for d in sc.front] == ["one"] and "waste" not in sc.designs[0].parameters
    text = res.format(all_designs=True)
    assert "Pareto front: 4 of 4" in text and "*" in text
    json.dumps(res.to_dict())
    assert "<table" in res._repr_html_()


def test_overlap_note_and_errors() -> None:
    noisy = pareto_search(
        TRADE,
        {"cost": "min", "wait": "min"},
        scenarios={"a": {"servers": 2}, "b": {"servers": 2}},
        replications=3,
    )
    assert len(noisy.front) >= 1
    for kwargs in ({}, {"grid": {"servers": [1]}, "ranges": {"servers": (1, 2)}}):
        with pytest.raises(ValueError, match="exactly one"):
            pareto_search(TRADE, {"cost": "min"}, **kwargs)
    with pytest.raises(ValueError, match="sense"):
        pareto_search(TRADE, {"cost": "lowest"}, grid={"servers": [1]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="at least one"):
        pareto_search(TRADE, {}, grid={"servers": [1]})
    with pytest.raises(ValueError):
        pareto_search(TRADE, {"cost": "min"}, grid={"servers": [0]})


def test_plot_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    res = pareto_search(
        TRADE, {"cost": "min", "wait": "min"}, grid={"servers": [1, 2, 4]}, replications=1
    )
    assert res.plot().get_title() == "Pareto front"
    with pytest.raises(ValueError):
        pareto_search(TRADE, {"cost": "min"}, grid={"servers": [1]}).plot()
    out = tmp_path / "front.png"
    assert (
        main(
            [
                "pareto",
                "builtin:mmc",
                "-o",
                "resource.server.wait.mean:min",
                "-o",
                "resource.server.utilization:max",
                "--vary",
                "servers=1,2,3",
                "-p",
                "arrival_rate=0.8",
                "-r",
                "2",
                "--duration",
                "2000",
                "--plot",
                str(out),
            ]
        )
        == 0
    )
    assert out.is_file() and "Pareto front" in capsys.readouterr().out
    assert (
        main(
            [
                "pareto",
                "builtin:supply_chain",
                "-o",
                "cost_per_day:min",
                "-o",
                "fill_rate:max",
                "--presets",
                "-r",
                "1",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["front"]
    assert main(["pareto", "builtin:mmc", "-o", "x:best", "--presets"]) != 0
    assert main(["pareto", "builtin:mmc", "-o", "x:min"]) != 0
