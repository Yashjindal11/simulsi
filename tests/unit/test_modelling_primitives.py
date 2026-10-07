from __future__ import annotations

import math
from typing import Any

import pytest

from simulsi import Container, Simulation
from simulsi.errors import CapacityError
from simulsi.processes import (
    PiecewiseRate,
    Router,
    arrivals,
    batch,
    capacity_schedule,
    join,
    shortest_queue,
    split,
)

# -- Container -------------------------------------------------------------------


def test_container_get_waits_for_stock() -> None:
    sim = Simulation(seed=1)
    tank = sim.container("tank", 100, init=50)
    log: list[tuple[float, float]] = []

    def consumer(sim: Simulation) -> Any:
        got = yield tank.get(80)
        log.append((sim.now, got))

    def producer(sim: Simulation) -> Any:
        yield 5.0
        yield tank.put(40)

    sim.process(consumer(sim))
    sim.process(producer(sim))
    res = sim.run(until=10)
    assert log == [(5.0, 80.0)]
    assert tank.level == pytest.approx(10.0)
    m = res.metrics
    assert m["container.tank.stockouts"] == 1
    assert m["container.tank.get_wait.mean"] == pytest.approx(5.0)
    assert m["container.tank.level.mean"] == pytest.approx((50 * 5 + 10 * 5) / 10)
    assert m["container.tank.level_final"] == pytest.approx(10.0)
    assert m["container.tank.put_amount"] == 40 and m["container.tank.get_amount"] == 80
    assert "container.tank.level" in res.series
    assert sim.snapshot()["containers"]["tank"]["level"] == pytest.approx(10.0)
    assert res.details["containers"]["tank"]["level_now"] == pytest.approx(10.0)


def test_container_put_blocks_when_full_and_gets_are_fifo() -> None:
    sim = Simulation(seed=1)
    bin_ = Container("bin", 10, init=8, sim=sim)
    events: list[tuple[str, float]] = []

    def put5(sim: Simulation) -> Any:
        yield bin_.put(5)
        events.append(("put", sim.now))

    def take(sim: Simulation, amount: float, at: float, tag: str) -> Any:
        yield at
        yield bin_.get(amount)
        events.append((tag, sim.now))

    sim.process(put5(sim))
    sim.process(take(sim, 4, 3.0, "get4"))
    sim.run(until=4)
    assert events == [("get4", 3.0), ("put", 3.0)]
    assert bin_.level == pytest.approx(9.0)

    # strict FIFO: a large waiting get holds back a later small one
    sim.process(take(sim, 10, 0.0, "big"))
    sim.process(take(sim, 1, 0.5, "small"))
    sim.run(until=6)
    assert [e[0] for e in events[2:]] == []

    def top_up(sim: Simulation) -> Any:
        yield bin_.put(1)

    sim.process(top_up(sim))
    sim.run(until=7)
    assert [e[0] for e in events[2:]] == ["big"]
    assert bin_.level == 0


def test_container_cancel_and_validation() -> None:
    sim = Simulation(seed=1)
    c = sim.container("c", 5)
    g = c.get(3)
    assert not g.triggered and c.cancel(g) and not c.cancel(g)
    with pytest.raises(ValueError):
        c.put(0)
    with pytest.raises(ValueError):
        c.get(math.inf)
    with pytest.raises(CapacityError):
        c.put(6)
    with pytest.raises(CapacityError):
        Container("x", 5, init=6)
    with pytest.raises(CapacityError):
        Container("x", 0)
    from simulsi.errors import ResourceUsageError

    with pytest.raises(ResourceUsageError):
        Container("loose").get(1)
    with pytest.raises(ResourceUsageError):
        sim.add_container(Container("c"))
    assert "Container('c'" in repr(c)


def test_container_warmup_resets_statistics() -> None:
    sim = Simulation(seed=1)
    c = sim.container("c", init=10)
    sim.warmup(5)

    def drain(sim: Simulation) -> Any:
        yield 2.0
        yield c.get(10)

    sim.process(drain(sim))
    res = sim.run(until=10)
    assert res.metrics["container.c.get_amount"] == 0
    assert res.metrics["container.c.level.mean"] == 0


# -- Queue filters -------------------------------------------------------------------


def test_filtered_get_takes_first_match_without_blocking_others() -> None:
    sim = Simulation(seed=1)
    q = sim.queue("parts")
    got: list[tuple[str, float, str]] = []

    def taker(sim: Simulation, tag: str, kind: str | None, at: float) -> Any:
        yield at
        item = yield q.get(None if kind is None else (lambda p: p == kind))
        got.append((tag, sim.now, item))

    def feeder(sim: Simulation) -> Any:
        for item in ["A1", "B1"]:
            yield q.put(item)
        yield 5.0
        yield q.put("C1")

    sim.process(feeder(sim))
    sim.process(taker(sim, "wantsC", "C1", 1.0))
    sim.process(taker(sim, "wantsB", "B1", 2.0))
    sim.process(taker(sim, "any", None, 3.0))
    sim.run()
    assert got == [("wantsB", 2.0, "B1"), ("any", 3.0, "A1"), ("wantsC", 5.0, "C1")]
    assert q.puts.value == q.gets.value == 3

    q.put("x1")
    q.put("y1")
    assert q.try_get(lambda s: s.startswith("y")) == "y1"
    assert q.try_get(lambda s: s.startswith("z")) is None
    assert q.try_get() == "x1"


def test_filtered_get_respects_discipline() -> None:
    sim = Simulation(seed=1)
    q = sim.queue("jobs", discipline="priority")
    for p, item in [(3, "a3"), (1, "a1"), (2, "b2")]:
        q.put(item, priority=p)
    assert q.try_get(lambda s: s.startswith("a")) == "a1"
    assert q.items == ["b2", "a3"]


# -- wait_until --------------------------------------------------------------------


def test_wait_until_triggers_on_state_change() -> None:
    sim = Simulation(seed=1)
    stock = sim.container("stock", init=50)
    seen: list[float] = []

    def demand(sim: Simulation) -> Any:
        while True:
            yield 1.0
            yield stock.get(7)

    def reorder(sim: Simulation) -> Any:
        t = yield sim.wait_until(lambda: stock.level < 20)
        seen.append(t)
        t2 = yield sim.wait_until(lambda: True)
        seen.append(t2)

    sim.process(demand(sim))
    sim.process(reorder(sim))
    sim.run(until=10)
    assert seen == [5.0, 5.0]  # 50 - 5 * 7 = 15 < 20 at t = 5
    assert not sim._watchers


# -- schedules -----------------------------------------------------------------------


def test_capacity_schedule_one_shot_and_periodic() -> None:
    sim = Simulation(seed=1)
    desk = sim.resource("desk", 1)
    capacity_schedule(sim, desk, [(0, 1), (10, 3), (20, 0)])
    caps = {}
    for t in (5, 15, 25):
        sim.run(until=t)
        caps[t] = desk.capacity
    assert caps == {5: 1, 15: 3, 25: 0}

    sim2 = Simulation(seed=1)
    line = sim2.resource("line", 2)
    capacity_schedule(sim2, line, [(0, 2), (8, 4)], period=24)
    seen = []
    for t in (4, 9, 26, 33, 50):
        sim2.run(until=t)
        seen.append(line.capacity)
    assert seen == [2, 4, 2, 4, 2]
    assert sim2.run(until=60).metrics["resource.line.utilization"] == 0

    with pytest.raises(ValueError, match="start at time 0"):
        capacity_schedule(sim2, line, [(1, 2)])
    with pytest.raises(ValueError, match="< period"):
        capacity_schedule(sim2, line, [(0, 2), (30, 1)], period=24)
    with pytest.raises(ValueError, match="integers"):
        capacity_schedule(sim2, line, [(0, 1.5)])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="duplicate"):
        capacity_schedule(sim2, line, [(0, 1), (0, 2)])


def test_piecewise_rate() -> None:
    r = PiecewiseRate([(0, 2.0), (8, 6.0), (17, 1.0)], period=24)
    assert (r(0), r(9.5), r(30.0), r(47.9), r.max) == (2.0, 6.0, 2.0, 1.0, 6.0)
    assert r.mean() == pytest.approx((2 * 8 + 6 * 9 + 1 * 7) / 24)
    gap = PiecewiseRate([(0, 0.0), (10, 2.0)])
    assert gap.next_arrival(0.0, 1.0) == pytest.approx(10.5)
    assert gap.next_arrival(12.0, 4.0) == pytest.approx(14.0)
    off = PiecewiseRate([(0, 1.0), (5, 0.0)])
    assert off.next_arrival(0.0, 10.0) == math.inf
    periodic = PiecewiseRate([(0, 1.0), (5, 0.0)], period=10)
    assert periodic.next_arrival(4.0, 3.0) == pytest.approx(12.0)  # 1 unit in [4,5), 2 in [10,12)
    assert "PiecewiseRate" in repr(r)
    with pytest.raises(ValueError):
        PiecewiseRate([(0, -1.0)])
    with pytest.raises(ValueError):
        PiecewiseRate([])
    with pytest.raises(ValueError):
        PiecewiseRate([(0, 1.0)], period=0)


def _count_arrivals(rate: Any, horizon: float, **kw: Any) -> list[float]:
    sim = Simulation(seed=11)
    times: list[float] = []
    arrivals(sim, rate, lambda sim, i: times.append(sim.now), **kw)
    sim.run(until=horizon)
    return times


def test_arrivals_constant_piecewise_and_thinning() -> None:
    times = _count_arrivals(2.0, 5_000)
    assert abs(len(times) - 10_000) < 4 * 100

    table = PiecewiseRate([(0, 1.0), (12, 5.0)], period=24)
    times = _count_arrivals(table, 24 * 200)
    day = sum(1 for t in times if t % 24 < 12)
    night = len(times) - day
    assert abs(day - 2400) < 4 * 49 and abs(night - 12_000) < 4 * 110

    def wave(t: float) -> float:
        return 2.0 + math.sin(2 * math.pi * t / 50)

    times = _count_arrivals(wave, 5_000, max_rate=3.0)
    assert abs(len(times) - 10_000) < 4 * 100
    first_half = sum(1 for t in times if (t % 50) < 25)
    assert first_half > 0.6 * len(times)  # the sine is positive in the first half

    assert len(_count_arrivals(5.0, 1_000, limit=7)) == 7
    assert max(_count_arrivals(5.0, 1_000, until=10.0)) <= 10.0
    with pytest.raises(ValueError, match="max_rate"):
        _count_arrivals(wave, 10)
    with pytest.raises(ValueError, match="exceeds"):
        _count_arrivals(wave, 100, max_rate=1.0)
    with pytest.raises(ValueError):
        _count_arrivals(0.0, 10)


def test_arrivals_start_generator_processes_and_are_reproducible() -> None:
    def run() -> list[float]:
        sim = Simulation(seed=3)
        done: list[float] = []

        def customer(sim: Simulation, i: int) -> Any:
            yield 1.0
            done.append(sim.now)

        arrivals(sim, 1.0, customer, limit=5)
        sim.run()
        return done

    a = run()
    assert len(a) == 5 and a == run()


# -- flow helpers ----------------------------------------------------------------------


def test_batch_with_and_without_timeout() -> None:
    sim = Simulation(seed=1)
    q = sim.queue("in")
    out: list[tuple[float, list[int]]] = []

    def feed(sim: Simulation) -> Any:
        for i in range(1, 8):
            yield 1.0
            yield q.put(i)

    def packer(sim: Simulation) -> Any:
        first = yield from batch(q, 3)
        out.append((sim.now, first))
        second = yield from batch(q, 5, timeout=2.5)
        out.append((sim.now, second))

    sim.process(feed(sim))
    sim.process(packer(sim))
    sim.run()
    assert out == [(3.0, [1, 2, 3]), (5.5, [4, 5])]
    assert q.items == [6, 7]  # the withdrawn get did not swallow item 6

    with pytest.raises(ValueError):
        next(batch(q, 0))


def test_split_join_and_failure() -> None:
    sim = Simulation(seed=1)
    out: list[Any] = []

    def task(sim: Simulation, d: float, v: str) -> Any:
        yield d
        return v

    def parent(sim: Simulation) -> Any:
        results = yield from join(split(sim, [task(sim, 3, "a"), task(sim, 5, "b")]))
        out.append((sim.now, results))
        out.append((yield from join([])))

    sim.process(parent(sim))
    sim.run()
    assert out == [(5.0, ["a", "b"]), []]

    def bad(sim: Simulation) -> Any:
        yield 1.0
        raise RuntimeError("boom")

    def parent2(sim: Simulation) -> Any:
        try:
            yield from join(split(sim, [bad(sim)]))
        except RuntimeError as e:
            out.append(str(e))

    sim.process(parent2(sim))
    sim.run()
    assert out[-1] == "boom"


def test_router_and_shortest_queue() -> None:
    sim = Simulation(seed=4)
    r = Router(sim, "qc", {"rework": 0.2, "ship": 0.8})
    picks = [r.choose() for _ in range(5_000)]
    assert r.routes == ("rework", "ship")
    assert abs(picks.count("rework") / 5_000 - 0.2) < 0.02
    m = sim.run(until=1).metrics
    assert m["route.qc.rework"] + m["route.qc.ship"] == 5_000
    sim2 = Simulation(seed=4)
    assert [Router(sim2, "qc", {"rework": 0.2, "ship": 0.8}).choose() for _ in range(50)] == picks[
        :50
    ]

    a, b = sim.resource("a", 1), sim.resource("b", 2)
    assert shortest_queue([a, b]) is b  # same (empty) queue, b has more free units
    a.request()
    a.request()
    assert shortest_queue([a, b]) is b
    with pytest.raises(ValueError):
        shortest_queue([])
