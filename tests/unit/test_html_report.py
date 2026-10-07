from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulsi import Experiment, Scenario, compare
from simulsi.analysis import morris_screening, select_best
from simulsi.analysis.report import html_table
from simulsi.cli import main
from simulsi.experiments import monte_carlo
from simulsi.models import mmc
from simulsi.optimization import optimize
from simulsi.randomness import Uniform, fit_distribution

SHORT = mmc.with_options(duration=300, warmup=30)


@pytest.fixture(scope="module")
def result() -> object:
    return Experiment(
        SHORT,
        [Scenario("baseline", {"servers": 1}), Scenario("<two>", {"servers": 2})],
        replications=4,
        seed=1,
    ).run()


def test_html_table_escapes_and_truncates() -> None:
    out = html_table([{"a": "<script>x</script>", "b": 1.5}] * 3, caption="c&d", max_rows=2)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "c&amp;d" in out and "1 more rows not shown" in out
    assert "(no rows)" in html_table([])


def test_experiment_report(result: object, tmp_path: Path) -> None:
    path = tmp_path / "r" / "report.html"
    text = result.report(path, metrics=["resource.server.wait.mean"])  # type: ignore[attr-defined]
    assert path.read_text(encoding="utf-8") == text
    assert text.startswith("<!DOCTYPE html>") and "<script" not in text
    assert "&lt;two&gt;" in text and "<two>" not in text  # scenario names escaped
    assert "Differences from baseline" in text and "p_adjusted" in text
    assert text.count("<svg") == 1
    default = result.report()  # type: ignore[attr-defined]
    assert "resource.server.utilization" in default and "sim.events" not in default


def test_notebook_reprs(result: object) -> None:
    assert "<table" in result._repr_html_()  # type: ignore[attr-defined]
    run = SHORT.simulate(seed=1)
    assert "resource.server.utilization" in run._repr_html_()
    assert "<table" in compare(result, "baseline")._repr_html_()  # type: ignore[arg-type]
    sens = morris_screening(lambda a, b: {"y": a + 2 * b}, {"a": (0, 1), "b": (0, 1)}, r=3)
    assert "morris-mu_star" in sens._repr_html_()
    mc = monte_carlo(lambda x: x * 2, {"x": Uniform(0, 1)}, 50)
    assert "Monte Carlo" in mc._repr_html_()
    fit = fit_distribution(np.random.default_rng(1).exponential(2.0, 200))
    assert "Distribution fits" in fit._repr_html_()
    sel = select_best(
        SHORT,
        {"a": {"servers": 1}, "b": {"servers": 3}},
        "resource.server.wait.mean",
        indifference=0.5,
    )
    assert "Selected" in sel._repr_html_()
    opt = optimize(
        SHORT,
        "resource.server.wait.mean",
        {"servers": (1, 3)},
        method="grid",
        budget=3,
        replications=1,
    )
    assert "grid search" in opt._repr_html_()


def test_report_cli(result: object, tmp_path: Path) -> None:
    result.save(tmp_path / "res")  # type: ignore[attr-defined]
    out = tmp_path / "report.html"
    assert (
        main(
            [
                "report",
                str(tmp_path / "res"),
                "-o",
                str(out),
                "--adjust",
                "bh",
                "-m",
                "resource.server.wait.mean",
                "--title",
                "My study",
            ]
        )
        == 0
    )
    text = out.read_text(encoding="utf-8")
    assert "<title>My study</title>" in text and "bh" in text
    assert main(["report", str(tmp_path / "missing"), "-o", str(out)]) != 0
