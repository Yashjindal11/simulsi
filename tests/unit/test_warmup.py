from __future__ import annotations

import numpy as np
import pytest

from simulsi.analysis import mser, suggest_warmup
from simulsi.cli import main
from simulsi.models import mmc


def test_mser_finds_transient() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(2000)
    x = 20 * np.exp(-t / 100) + rng.normal(0, 1, len(t))
    r = mser(x)
    assert 150 <= r.truncation <= 700 and r.reliable
    flat = mser(rng.normal(0, 1, 2000))
    assert flat.truncation < 300


def test_mser_flags_too_short_and_tiny_inputs() -> None:
    trend = np.linspace(0, 10, 500)  # never settles
    assert not mser(trend).reliable
    assert not mser([1, 2, 3]).reliable
    with pytest.raises(ValueError):
        mser([1.0] * 50, batch_size=0)


def _backlogged(sim, p):  # type: ignore[no-untyped-def]
    mmc.build(sim, p)
    server = sim.resources["server"]

    def initial(sim):  # type: ignore[no-untyped-def]
        yield from sim.use(server, sim.stream("backlog").exponential(1 / p.service_rate))

    for _ in range(80):  # a large initial backlog: a clear start-up transient
        sim.process(initial(sim))


def test_suggest_warmup_detects_initial_backlog() -> None:
    from simulsi import Model

    model = Model(
        _backlogged, name="backlog", duration=4_000.0, parameters=list(mmc.parameters.values())
    )
    adv = suggest_warmup(model, {"arrival_rate": 0.7}, replications=4, seed=1)
    assert adv.series == "resource.server.queue_length"
    # 80 customers drain at rate 1 - 0.7 = 0.3 per time unit: roughly 270 time units
    assert 100 < adv.warmup < 1_500, adv.warmup
    assert len(adv.averaged) == 200
    with pytest.raises(KeyError):
        suggest_warmup(model, series="nope", replications=1)


def test_warmup_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["warmup", "builtin:mmc", "--duration", "2000", "-r", "2"]) == 0
    assert "suggested warm-up" in capsys.readouterr().out
