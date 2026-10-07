from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from simulsi import Model, Parameter, Simulation
from simulsi.analysis import morris_screening, select_best
from simulsi.models import mmc
from simulsi.optimization import (
    GaussianProcess,
    Objective,
    fit_surrogate,
    latin_hypercube,
    optimize,
)


def _bowl(sim: Simulation, p: Any) -> None:
    noise = sim.stream("noise").normal(0, 0.05)
    sim.metrics.set("cost", (p.x - 2.0) ** 2 + (p.n - 3) ** 2 + noise)


BOWL = Model(
    _bowl,
    name="bowl",
    duration=1.0,
    parameters=[
        Parameter("x", 0.0, "float", low=-5, high=5),
        Parameter("n", 1, "int", low=0, high=8),
    ],
)


def _arms(sim: Simulation, p: Any) -> None:
    sim.metrics.set("y", p.mu + sim.stream("noise").normal(0, 1.0))


ARMS = Model(_arms, name="arms", duration=1.0, parameters=[Parameter("mu", 0.0, "float")])

# -- ranking and selection ---------------------------------------------------------------


def test_select_best_on_queue_model() -> None:
    m = mmc.with_options(duration=500, warmup=50)
    cands = {f"s{c}": {"servers": c, "arrival_rate": 1.5} for c in (2, 3, 4)}
    sel = select_best(m, cands, "resource.server.wait.mean", indifference=0.05)
    assert sel.best == "s4" and sel.converged
    assert set(sel.eliminated) == {"s2", "s3"}
    assert sel.total_replications == sum(sel.replications.values())
    assert "selected" in sel.format()
    json.dumps(sel.to_dict())
    util = select_best(m, cands, "resource.server.utilization", indifference=0.01, minimize=False)
    assert util.best == "s2"


def test_select_best_achieves_nominal_probability_of_correct_selection() -> None:
    # best arm mu=0 (minimise); others are exactly one indifference zone worse
    cands = {"best": {"mu": 0.0}, "b": {"mu": 0.5}, "c": {"mu": 0.5}, "d": {"mu": 0.8}}
    correct = sum(
        select_best(ARMS, cands, "y", indifference=0.5, seed=s, n0=10).best == "best"
        for s in range(40)
    )
    assert correct >= 35  # nominal >= 0.95; allow sampling slack


def test_select_best_cap_and_validation() -> None:
    same = {"a": {"mu": 0.0}, "b": {"mu": 0.0}}
    sel = select_best(ARMS, same, "y", indifference=0.01, max_replications=30)
    assert not sel.converged and "max_replications" in sel.note
    assert sel.replications["a"] == 30
    with pytest.raises(ValueError, match="two scenarios"):
        select_best(ARMS, {"a": {}}, "y", indifference=1)
    with pytest.raises(ValueError, match="indifference"):
        select_best(ARMS, same, "y", indifference=0)
    with pytest.raises(ValueError, match="n0"):
        select_best(ARMS, same, "y", indifference=1, n0=1)
    with pytest.raises(KeyError):
        select_best(ARMS, same, "nope", indifference=1)


# -- Morris ----------------------------------------------------------------------------------


def test_morris_screening_on_known_function() -> None:
    def f(a: float, b: float, c: float) -> dict[str, float]:
        return {"y": 4 * a + 2 * b**2 + 0 * c}

    res = morris_screening(f, {"a": (0, 1), "b": (0, 1), "c": (0, 1)}, r=30, seed=3)
    rows = {(r.parameter, r.method): r for r in res.rows}
    assert rows[("a", "morris-mu_star")].value == pytest.approx(4.0)
    assert rows[("a", "morris-sigma")].value == pytest.approx(0.0, abs=1e-9)
    assert rows[("c", "morris-mu_star")].value == 0.0
    assert rows[("b", "morris-sigma")].value > 0.1  # non-linear in b
    ranking = [r.parameter for r in res.ranking(method="morris-mu_star")]
    assert ranking == ["a", "b", "c"]
    assert res.info["evaluations"] == 30 * 4
    lo, hi = rows[("b", "morris-mu_star")].ci_low, rows[("b", "morris-mu_star")].ci_high
    assert lo <= rows[("b", "morris-mu_star")].value <= hi


def test_morris_on_simulation_model_and_validation() -> None:
    m = mmc.with_options(duration=300, warmup=30)
    res = morris_screening(
        m,
        {"arrival_rate": (0.4, 0.8), "servers": (1, 3)},
        r=4,
        outputs=["resource.server.utilization"],
    )
    star = {r.parameter: r.value for r in res.rows if r.method == "morris-mu_star"}
    assert set(star) == {"arrival_rate", "servers"} and all(v > 0 for v in star.values())
    with pytest.raises(ValueError, match="levels"):
        morris_screening(m, {"servers": (1, 3)}, levels=3)
    with pytest.raises(ValueError, match="r must"):
        morris_screening(m, {"servers": (1, 3)}, r=1)
    with pytest.raises(ValueError, match="low < high"):
        morris_screening(m, {"servers": (3, 1)})
    with pytest.raises(ValueError, match="at least one"):
        morris_screening(m, {})
    with pytest.raises(KeyError):
        morris_screening(lambda a: {"y": a}, {"a": (0, 1)}, r=2, outputs=["z"])


# -- surrogates ------------------------------------------------------------------------------


def test_gaussian_process_fit_predict_and_uncertainty() -> None:
    X = np.linspace(0, 1, 15)[:, None]
    gp = GaussianProcess(seed=0).fit(X, np.sin(6 * X[:, 0]))
    assert gp.loo_r2() > 0.99
    mean, sd = gp.predict(np.array([[0.31], [3.0]]), return_std=True)
    assert mean[0] == pytest.approx(math.sin(6 * 0.31), abs=0.02)
    assert sd[1] > 10 * sd[0]  # far outside the data: uncertain
    noisy = GaussianProcess(seed=1).fit(
        X, np.sin(6 * X[:, 0]) + np.random.default_rng(0).normal(0, 0.2, 15)
    )
    assert noisy.noise_std > 0.05
    with pytest.raises(RuntimeError):
        GaussianProcess().predict([[0.0]])
    with pytest.raises(RuntimeError):
        GaussianProcess().loo_residuals()
    with pytest.raises(ValueError):
        GaussianProcess().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError):
        GaussianProcess().fit([[0.0], [1.0], [2.0]], [0.0, 1.0])
    with pytest.raises(ValueError):
        GaussianProcess().fit([[0.0], [1.0], [math.nan]], [0.0, 1.0, 2.0])


def test_latin_hypercube_is_stratified() -> None:
    u = latin_hypercube(10, 3, seed=1)
    for j in range(3):
        assert sorted(np.floor(u[:, j] * 10).astype(int)) == list(range(10))


def test_fit_surrogate_for_model_and_function() -> None:
    s = fit_surrogate(BOWL, {"x": (-5, 5), "n": (0, 8)}, ["cost"], n=30)
    assert s.accuracy()["cost"] > 0.95
    assert all(isinstance(p["n"], int) for p in s.design)
    mean, sd = s.predict({"x": 2.0, "n": 3}, return_std=True)
    assert abs(mean[0]) < 1.0 and sd[0] >= 0
    with pytest.raises(ValueError, match="extrapolation"):
        s.predict({"x": 6.0, "n": 3})
    with pytest.raises(KeyError):
        s.predict({"x": 1.0, "n": 3}, "other")
    f = fit_surrogate(lambda a: a**2, {"a": (-1, 1)}, ["value"], n=12)
    assert f.predict([{"a": 0.5}])[0] == pytest.approx(0.25, abs=0.05)
    with pytest.raises(KeyError):
        fit_surrogate(lambda a: a, {"a": (0, 1)}, ["nope"], n=5)
    with pytest.raises(ValueError):
        fit_surrogate(BOWL, {"x": (1, 1)}, ["cost"])


# -- optimisation ----------------------------------------------------------------------------


def test_optimize_bayes_finds_interior_optimum_and_confirms() -> None:
    res = optimize(
        BOWL,
        "cost",
        {"x": (-5, 5), "n": (0, 8)},
        method="bayes",
        budget=40,
        replications=2,
        indifference=0.05,
    )
    assert res.best_parameters["n"] == 3
    assert res.best_parameters["x"] == pytest.approx(2.0, abs=0.3)
    assert res.evaluations <= 40
    assert res.selection is not None and res.selection.converged
    assert "bayes search" in res.format()
    json.dumps(res.to_dict())


def test_optimize_grid_random_and_errors() -> None:
    grid = optimize(BOWL, "cost", {"x": (0, 4), "n": (2, 4)}, method="grid", budget=15)
    assert grid.evaluations == 15 and grid.best_parameters == {"x": 2.0, "n": 3}
    rnd = optimize(BOWL, "cost", {"x": (-5, 5), "n": (0, 8)}, method="random", budget=25)
    assert rnd.evaluations == 25 and rnd.best_value < 5
    small = optimize(BOWL, "cost", {"n": (2, 4)}, method="bayes", budget=20)
    assert small.evaluations == 3 and small.best_parameters == {"n": 3}

    cost = optimize(
        BOWL,
        None,
        {"x": (0, 4)},
        method="grid",
        budget=5,
        transform=lambda m: m["cost"] + 1,
        indifference=0.1,
    )
    assert cost.best_value == pytest.approx(1 + (2 - 2) ** 2 + (1 - 3) ** 2, abs=0.2)
    assert any("transform" in n for n in cost.notes)
    with pytest.raises(ValueError, match="budget"):
        optimize(BOWL, "cost", {"x": (0, 4), "n": (0, 8)}, method="grid", budget=10)
    with pytest.raises(ValueError, match="unknown method"):
        optimize(BOWL, "cost", {"x": (0, 4)}, method="anneal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="low < high"):
        optimize(BOWL, "cost", {"x": (4, 0)})
    with pytest.raises(ValueError, match="budget must"):
        optimize(BOWL, "cost", {"x": (0, 4)}, budget=0)
    assert isinstance(Objective(BOWL, "cost", ["x"]), Objective)
