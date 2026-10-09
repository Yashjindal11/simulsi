from __future__ import annotations

from typing import Any

import pytest

from simulsi import Simulation
from simulsi.processes import Level


def test_level_crossings_are_exact_and_follow_rate_changes() -> None:
    sim = Simulation()
    tank = Level(sim, "tank", init=10, rate=-1.0, low=0, high=50)
    seen: list[tuple[str, float]] = []

    def pump(sim: Simulation) -> Any:
        t = yield tank.when(4, "down")
        seen.append(("low", t))
        tank.set_rate(3.0)
        t = yield tank.when(13, "up")
        seen.append(("high", t))
        delivered = tank.add(100)
        seen.append(("added", delivered))

    sim.process(pump(sim))
    result = sim.run(until=20)
    assert seen == [("low", 6.0), ("high", 9.0), ("added", 37.0)]
    assert tank.value == 50.0
    assert result.metrics["level.tank.min"] == 4.0
    assert result.metrics["level.tank.max"] == 50.0


def test_level_pins_at_bounds_and_time_average_is_exact() -> None:
    sim = Simulation()
    lvl = Level(sim, "x", init=0, rate=1.0, high=5)
    sim.run(until=10)
    assert lvl.value == 5.0 and lvl.rate == 0.0
    # 0..5 ramps (area 12.5) then 5 for 5 units (25) -> 37.5 / 10
    assert lvl.time_average() == pytest.approx(3.75)


def test_level_when_already_reached_and_unreachable() -> None:
    sim = Simulation()
    lvl = Level(sim, "x", init=5, rate=1.0, low=0, high=8)
    now = lvl.when(6, "down")
    never = lvl.when(10, "up")  # beyond the bound
    sim.run(until=20)
    assert now.triggered and now.value == 0.0
    assert not never.triggered
    with pytest.raises(ValueError):
        lvl.when(1, "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Level(sim, "bad", init=9, high=8)


def test_level_rate_change_cancels_stale_crossing() -> None:
    sim = Simulation()
    lvl = Level(sim, "x", init=10, rate=-1.0)
    hits: list[float] = []

    def watch(sim: Simulation) -> Any:
        hits.append((yield lvl.when(0, "down")))

    def stop(sim: Simulation) -> Any:
        yield 4
        lvl.set_rate(0.0)
        yield 10
        lvl.set_rate(-2.0)

    sim.process(watch(sim))
    sim.process(stop(sim))
    sim.run()
    assert hits == [17.0]
