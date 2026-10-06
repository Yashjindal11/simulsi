from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from simulsi import ConfigError, Experiment, ExperimentResult, Model, Scenario, compare, monte_carlo
from simulsi.analysis import compare_samples
from simulsi.models import mmc
from simulsi.randomness import Normal, Uniform

pytestmark = pytest.mark.integration

FAST = {"arrival_rate": 0.8}


def short_mmc() -> Model:
    m = Model(
        mmc.build,
        name="mmc_short",
        duration=500.0,
        warmup=50.0,
        parameters=list(mmc.parameters.values()),
        sim_options=mmc.sim_options,
    )
    return m


def test_experiment_records_and_provenance() -> None:
    exp = Experiment(
        short_mmc(),
        [Scenario("baseline", FAST), Scenario("two", {**FAST, "servers": 2})],
        replications=4,
        seed=7,
    )
    res = exp.run()
    assert len(res.records) == 8
    assert res.scenarios == ["baseline", "two"]
    md = res.metadata
    assert md.seed == 7 and md.model_name == "mmc_short" and md.environment["simulsi"]
    assert md.experiment_id.startswith("EXP-")
    assert res.records[0].parameters == {"arrival_rate": 0.8, "service_rate": 1.0, "servers": 1}
    # common random numbers: same seeds across scenarios
    assert res.seeds("baseline") == res.seeds("two")
    assert len(set(res.seeds("baseline"))) == 4


def test_experiment_is_reproducible() -> None:
    a = Experiment(short_mmc(), Scenario("baseline", FAST), replications=3, seed=1).run()
    b = Experiment(short_mmc(), Scenario("baseline", FAST), replications=3, seed=1).run()
    c = Experiment(short_mmc(), Scenario("baseline", FAST), replications=3, seed=2).run()
    m = "resource.server.wait.mean"
    assert a.values(m).tolist() == b.values(m).tolist()
    assert a.values(m).tolist() != c.values(m).tolist()


def test_parallel_matches_serial() -> None:
    sc = [Scenario("baseline", FAST), Scenario("fast", {**FAST, "service_rate": 1.5})]
    serial = Experiment(mmc, sc, replications=4, seed=3).run()
    parallel = Experiment(mmc, sc, replications=4, seed=3, workers=2).run()
    m = "resource.server.wait.mean"
    for s in ("baseline", "fast"):
        assert serial.values(m, s).tolist() == parallel.values(m, s).tolist()


def test_parallel_rejects_unpicklable_model() -> None:
    m = Model(lambda sim, p: None, duration=1)
    with pytest.raises(ConfigError, match="picklable"):
        Experiment(m, replications=2, workers=2).run()


def test_validation_errors() -> None:
    with pytest.raises(ConfigError, match="unknown parameter"):
        Experiment(short_mmc(), Scenario("x", {"bogus": 1})).run()
    with pytest.raises(ConfigError, match="duplicate scenario"):
        Experiment(short_mmc(), [Scenario("x"), Scenario("x")]).run()
    with pytest.raises(ConfigError, match="no duration"):
        Experiment(Model(lambda s, p: None)).run()


def test_grid_runs_matrix() -> None:
    exp = Experiment(short_mmc(), replications=2, seed=0)
    scenarios = exp.grid({"servers": [1, 2], "arrival_rate": [0.5, 0.8]})
    assert len(scenarios) == 4
    res = exp.run()
    assert len(res.records) == 8
    rows = res.summary(["resource.server.utilization"])
    assert {r["scenario"] for r in rows} == {s.name for s in scenarios}


def test_checkpoint_resume(tmp_path: Path) -> None:
    ck = tmp_path / "run.checkpoint.jsonl"
    exp = Experiment(short_mmc(), Scenario("baseline", FAST), replications=3, seed=5)
    first = exp.run(checkpoint=ck)
    lines = ck.read_text().splitlines()
    assert len(lines) == 1 + 3
    # drop the last record, as if the run had been interrupted
    ck.write_text("\n".join(lines[:-1]) + "\n")
    calls: list[int] = []
    resumed = exp.run(checkpoint=ck, progress=lambda done, total: calls.append(done))
    assert calls == [3]  # only one replication actually ran
    m = "resource.server.wait.mean"
    assert resumed.values(m).tolist() == first.values(m).tolist()
    with pytest.raises(ConfigError, match="different experiment"):
        Experiment(short_mmc(), Scenario("baseline", FAST), replications=3, seed=6).run(
            checkpoint=ck
        )


def test_error_recording() -> None:
    def build(sim, p):  # type: ignore[no-untyped-def]
        def bad(sim):  # type: ignore[no-untyped-def]
            yield 1
            raise RuntimeError("kaput")

        sim.process(bad(sim))

    m = Model(build, duration=5)
    with pytest.raises(RuntimeError):
        Experiment(m, replications=1).run()
    res = Experiment(m, replications=2, on_error="record").run()
    assert len(res.errors) == 2 and "kaput" in (res.errors[0].error or "")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    res = Experiment(short_mmc(), [Scenario("baseline", FAST)], replications=3, seed=1).run()
    d = res.save(tmp_path / "out", formats=("json", "csv", "parquet"))
    for f in ("experiment.json", "replications.csv", "summary.csv", "replications.parquet"):
        assert (d / f).exists()
    json.loads((d / "experiment.json").read_text())  # strict JSON (no NaN)
    again = ExperimentResult.load(d)
    m = "resource.server.wait.mean"
    assert again.values(m).tolist() == res.values(m).tolist()
    assert again.metadata == res.metadata
    df = res.to_dataframe()
    assert len(df) == 3 and "param.servers" in df.columns
    with pytest.raises(ConfigError):
        ExperimentResult.from_dict({"nope": 1})


def test_compare_paired_detects_capacity_effect() -> None:
    exp = Experiment(
        short_mmc(),
        [Scenario("baseline", FAST), Scenario("two", {**FAST, "servers": 2})],
        replications=5,
        seed=2,
    )
    res = exp.run()
    cmp = compare(res, "baseline", metrics=["resource.server.wait.mean"])
    row = cmp.get("resource.server.wait.mean", "two")
    assert row.method == "paired-t"
    assert row.absolute_difference < 0 and row.significant
    assert row.percentage_difference < -50
    assert "resource.server.wait.mean" in cmp.format()
    with pytest.raises(KeyError):
        compare(res, "missing")


def test_compare_samples_welch() -> None:
    c = compare_samples({"x": [1, 2, 3, 4]}, {"x": [11, 12, 13, 14]})
    assert c.rows[0].method == "welch-t" and c.rows[0].absolute_difference == 10


def test_replication_advice_from_experiment() -> None:
    res = Experiment(short_mmc(), Scenario("baseline", FAST), replications=5, seed=1).run()
    adv = res.replication_advice("resource.server.utilization", relative_precision=0.5)
    assert adv.sufficient


def test_monte_carlo_scalar_function() -> None:
    def profit(price: float, demand: float) -> float:
        return price * demand - 1000

    r = monte_carlo(profit, {"price": Uniform(9, 11), "demand": Normal(200, 20)}, 20_000, seed=1)
    s = r.summary()
    assert s.mean == pytest.approx(1000, rel=0.03)
    p, (lo, hi) = r.probability("value", "<", 0)
    assert 0 <= lo <= p <= hi <= 1
    assert r.percentile(50) == pytest.approx(r.quantile(0.5))
    lo_q, hi_q = r.quantile_ci(0.95, n_resamples=300)
    assert lo_q <= r.quantile(0.95) <= hi_q
    again = monte_carlo(
        profit, {"price": Uniform(9, 11), "demand": Normal(200, 20)}, 20_000, seed=1
    )
    assert np.array_equal(again.outputs["value"], r.outputs["value"])


def test_monte_carlo_vectorized_matches_loop() -> None:
    params = {"a": Uniform(0, 1), "b": Uniform(0, 2)}
    loop = monte_carlo(lambda a, b: {"s": a + b}, params, 1000, seed=4)
    vec = monte_carlo(
        lambda a, b: {"s": a + b},
        params,
        1000,
        seed=4,
        vectorized=True,
        objective=lambda o: o["s"] * 2,
    )
    assert np.allclose(loop.outputs["s"], vec.outputs["s"])
    assert np.allclose(vec.outputs["objective"], 2 * vec.outputs["s"])


def test_monte_carlo_rng_injection_and_lhs() -> None:
    def noisy(x: float, rng) -> float:  # type: ignore[no-untyped-def]
        return x + rng.normal(0, 1)

    r = monte_carlo(noisy, {"x": 5.0}, 2000, seed=2)
    assert r.summary().mean == pytest.approx(5, abs=0.1)
    lhs = monte_carlo(lambda x: x, {"x": Uniform(0, 1)}, 100, seed=1, sampling="lhs")
    # exactly one sample per stratum
    assert sorted(np.floor(lhs.inputs["x"] * 100).astype(int).tolist()) == list(range(100))


def test_monte_carlo_over_simulation_model() -> None:
    r = monte_carlo(short_mmc(), {"arrival_rate": Uniform(0.3, 0.6)}, 5, seed=1)
    u = r.outputs["resource.server.utilization"]
    assert len(u) == 5 and np.all((u > 0.2) & (u < 0.7))
    assert not math.isnan(r.summary("resource.server.utilization").mean)


def test_worker_pool_is_reused_and_can_be_shut_down() -> None:
    from simulsi.experiments import experiment as exp_mod
    from simulsi.experiments import shutdown_workers

    shutdown_workers()
    for seed in (1, 2):
        Experiment(mmc, Scenario("baseline", FAST), replications=2, seed=seed, workers=2).run()
    assert len(exp_mod._POOLS) == 1
    shutdown_workers()
    assert not exp_mod._POOLS
