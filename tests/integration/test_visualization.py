from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

from simulsi import Experiment, Model, Scenario, compare
from simulsi.analysis import finite_difference
from simulsi.models import mmc
from simulsi.visualization import (
    plot_comparison,
    plot_convergence,
    plot_distribution,
    plot_entity_trajectories,
    plot_queue_length,
    plot_sensitivity,
    plot_series,
    plot_timeline,
    plot_utilization,
    save_figure,
)

matplotlib.use("Agg")

MODEL = Model(mmc.build, name="mmc_viz", duration=200.0, parameters=list(mmc.parameters.values()))


@pytest.fixture(scope="module")
def traced():  # type: ignore[no-untyped-def]
    from test_analysis_features import flow_model

    sim = flow_model.create(seed=1, trace=True)
    return sim.run(until=50)


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_all_plots_render(backend: str, traced, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    res = MODEL.simulate(seed=1)
    exp = Experiment(
        MODEL, [Scenario("baseline"), Scenario("two", {"servers": 2})], replications=3
    ).run()
    cmp = compare(exp, metrics=["resource.server.wait.mean"])
    sens = finite_difference(
        MODEL, ["arrival_rate"], ["resource.server.utilization"], replications=2
    )
    figs = [
        plot_series(res, backend=backend),  # type: ignore[arg-type]
        plot_queue_length(res, backend=backend),  # type: ignore[arg-type]
        plot_utilization(res, backend=backend),  # type: ignore[arg-type]
        plot_timeline(traced.log, backend=backend),  # type: ignore[arg-type]
        plot_entity_trajectories(traced.log, backend=backend, end_time=50),  # type: ignore[arg-type]
        plot_distribution(exp.values("resource.server.wait.mean"), backend=backend),  # type: ignore[arg-type]
        plot_convergence(exp.values("resource.server.wait.mean"), backend=backend),  # type: ignore[arg-type]
        plot_comparison(cmp, backend=backend),  # type: ignore[arg-type]
        plot_sensitivity(sens, backend=backend),  # type: ignore[arg-type]
    ]
    assert all(f is not None for f in figs)
    out = tmp_path / ("fig.png" if backend == "matplotlib" else "fig.html")
    save_figure(figs[2], str(out))
    assert out.stat().st_size > 0
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close("all")


def test_plot_series_errors() -> None:
    res = MODEL.simulate(seed=1, record_series=False)
    with pytest.raises(ValueError):
        plot_series(res)
