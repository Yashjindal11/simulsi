from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from simulsi import Model, Parameter, Simulation
from simulsi.analysis import (
    InputData,
    clopper_pearson,
    gpd_tail_probability,
    input_uncertainty,
    rare_event_probability,
)
from simulsi.models import mmc
from simulsi.optimization import optimize

SHORT = mmc.with_options(duration=3_000, warmup=300)


def _gaps(n: int) -> list[float]:
    return list(np.random.default_rng(5).exponential(1 / 0.7, size=n))


def test_input_uncertainty_shrinks_with_more_data() -> None:
    def rate(d: Any) -> float:
        return 1 / d.mean

    small = input_uncertainty(
        SHORT,
        {"arrival_rate": InputData(_gaps(25), "exponential", rate)},
        "resource.server.wait.mean",
        n_bootstrap=8,
        replications=3,
        seed=1,
    )
    big = input_uncertainty(
        SHORT,
        {"arrival_rate": InputData(_gaps(3000), "exponential", rate)},
        "resource.server.wait.mean",
        n_bootstrap=8,
        replications=3,
        seed=1,
    )
    assert small.input_variance > 5 * big.input_variance
    assert small.input_share > 0.5 and small.ci_low < small.point_estimate < small.ci_high * 1.5
    assert "most of the uncertainty" in small.format()
    json.dumps(small.to_dict())
    assert "<table" in small._repr_html_()
    with pytest.raises(ValueError):
        input_uncertainty(SHORT, {}, "x")
    with pytest.raises(ValueError):
        input_uncertainty(SHORT, {"arrival_rate": InputData(_gaps(10))}, "x", n_bootstrap=2)
    with pytest.raises(ValueError):
        input_uncertainty(SHORT, {"arrival_rate": InputData(_gaps(10))}, "x", replications=1)


def test_clopper_pearson_reference_values() -> None:
    assert clopper_pearson(0, 10) == (0.0, pytest.approx(0.3085, abs=1e-4))
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)
    assert clopper_pearson(10, 10)[1] == 1.0


def test_gpd_tail_extrapolates_beyond_the_data() -> None:
    x = np.random.default_rng(3).exponential(1.0, size=3000)
    q = -math.log(1e-5)  # true P(X > q) = 1e-5, far beyond 3000 samples
    est = gpd_tail_probability(x, q, seed=1, n_bootstrap=100)
    assert est.ci_low <= 1e-5 <= est.ci_high
    assert 1e-6 < est.probability < 1e-4 and abs(est.shape) < 0.2
    inside = gpd_tail_probability(x, 0.5)
    assert inside.probability == pytest.approx(np.mean(x > 0.5))
    with pytest.raises(ValueError):
        gpd_tail_probability(x[:10], 1.0)


def test_rare_event_probability_on_a_queue() -> None:
    res = rare_event_probability(
        SHORT, "resource.server.wait.max", 30.0, replications=40, params={"arrival_rate": 0.8}
    )
    assert res.n == 40 and 0 <= res.hits <= 40
    assert res.ci_low <= res.probability <= res.ci_high
    assert res.tail is not None and "Clopper-Pearson" in res.format()
    json.dumps(res.to_dict(), default=float)
    none = rare_event_probability(
        SHORT, "resource.server.wait.max", 1e9, replications=5, tail=False
    )
    assert none.hits == 0 and none.ci_low == 0 and math.isinf(none.replications_for_10pct)
    with pytest.raises(KeyError):
        rare_event_probability(SHORT, "nope", 1, replications=2)


def _bowl(sim: Simulation, p: Any) -> None:
    sim.metrics.set("cost", (p.x - 2.0) ** 2 + (p.n - 3) ** 2 + sim.stream("n").normal(0, 0.05))


BOWL = Model(
    _bowl,
    name="bowl",
    duration=1.0,
    parameters=[Parameter("x", 0.0, "float"), Parameter("n", 1, "int")],
)


def test_cross_entropy_method_finds_the_optimum() -> None:
    res = optimize(
        BOWL, "cost", {"x": (-5, 5), "n": (0, 8)}, method="cem", budget=80, replications=2
    )
    assert res.best_parameters["n"] == 3 and res.best_parameters["x"] == pytest.approx(2.0, abs=0.5)
    assert res.evaluations <= 80
