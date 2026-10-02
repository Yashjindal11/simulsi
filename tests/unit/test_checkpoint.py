from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from simulsi import Model, Simulation, model
from simulsi.core.checkpoint import CheckpointError
from simulsi.models import mmc

MODEL = mmc.with_options(duration=2_000, warmup=100)


def test_resume_from_checkpoint_matches_uninterrupted_run(tmp_path: Path) -> None:
    full = MODEL.create({"servers": 2, "arrival_rate": 1.6}, seed=4)
    expected = full.run(until=1_500).metrics

    first = MODEL.create({"servers": 2, "arrival_rate": 1.6}, seed=4)
    first.run(until=700)
    ck = first.save_checkpoint(tmp_path / "run.ckpt.json")
    restored = Simulation.load_checkpoint(ck, model=MODEL)
    assert restored.now == 700 and restored.events_processed == first.events_processed
    assert restored.run(until=1_500).metrics == expected


def test_builtin_model_resolves_automatically(tmp_path: Path) -> None:
    sim = mmc.create(seed=1)
    sim.run(until=300)
    ck = sim.save_checkpoint(tmp_path / "c.json")
    again = Simulation.load_checkpoint(ck)
    assert again.now == 300 and again.flat_metrics() == sim.flat_metrics()


@model(duration=100, version="1")
def counter_model(sim: Simulation, p: Any) -> None:
    def tick(sim: Simulation) -> Any:
        while True:
            yield sim.stream("t").exponential(1.0)
            sim.metrics.increment("ticks")
            if sim.now > 30 and not sim.metrics.values.get("stopped"):
                sim.metrics.set("stopped", 1)
                sim.stop()

    sim.process(tick(sim))


def test_module_level_model_and_stop_inside_event(tmp_path: Path) -> None:
    sim = counter_model.create(seed=3)
    sim.run(until=100)  # stops early, mid-time, via sim.stop()
    assert sim.now < 100
    ck = sim.save_checkpoint(tmp_path / "c.json")
    again = Simulation.load_checkpoint(ck)  # found via module + attribute
    assert again.now == sim.now and again.events_processed == sim.events_processed
    assert again.run(until=60).metrics["ticks"] == sim.run(until=60).metrics["ticks"]


def test_checkpoint_errors(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match=r"Model\.create"):
        Simulation(seed=1).save_checkpoint(tmp_path / "x.json")

    sim = MODEL.create(seed=1)
    sim.run(until=50)
    ck = sim.save_checkpoint(tmp_path / "c.json")
    with pytest.raises(CheckpointError, match="cannot locate"):
        Simulation.load_checkpoint(ck)  # with_options copy is not importable
    other = MODEL.with_options(name="renamed")
    with pytest.raises(CheckpointError, match="written by model"):
        Simulation.load_checkpoint(ck, model=other)
    (tmp_path / "bad.json").write_text('{"format": "nope"}')
    with pytest.raises(CheckpointError, match="not a SimulSI checkpoint"):
        Simulation.load_checkpoint(tmp_path / "bad.json")


def test_nondeterministic_model_is_detected(tmp_path: Path) -> None:
    def build(sim: Simulation, p: Any) -> None:
        def proc(sim: Simulation) -> Any:
            while True:
                yield random.random()  # global randomness: not reproducible
                sim.metrics.observe("x", 1)

        sim.process(proc(sim))

    m = Model(build, name="bad", duration=100)
    sim = m.create(seed=1)
    sim.run(until=40)
    ck = sim.save_checkpoint(tmp_path / "c.json")
    with pytest.raises(CheckpointError):
        Simulation.load_checkpoint(ck, model=m)
