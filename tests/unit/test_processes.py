from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from simulsi import EventStateError, Interrupt, Simulation


def test_process_holds_with_numbers_and_timeouts() -> None:
    sim = Simulation(seed=1)
    trace: list[tuple[float, Any]] = []

    def proc(sim: Simulation) -> Any:
        trace.append((sim.now, "start"))
        yield 2
        v = yield sim.timeout(3, value="hello")
        trace.append((sim.now, v))
        yield timedelta(minutes=1)
        return "done"

    p = sim.process(proc(sim))
    sim.run()
    assert trace == [(0.0, "start"), (5.0, "hello")]
    assert sim.now == 6.0
    assert p.value == "done" and not p.is_alive


def test_waiting_on_another_process() -> None:
    sim = Simulation(seed=1)

    def child(sim: Simulation) -> Any:
        yield 4
        return 42

    def parent(sim: Simulation) -> Any:
        result = yield sim.process(child(sim))
        return (sim.now, result)

    p = sim.process(parent(sim))
    sim.run()
    assert p.value == (4.0, 42)


def test_failed_child_raises_in_parent() -> None:
    sim = Simulation(seed=1)

    def child(sim: Simulation) -> Any:
        yield 1
        raise ValueError("boom")

    def parent(sim: Simulation) -> Any:
        try:
            yield sim.process(child(sim))
        except ValueError as exc:
            return str(exc)

    p = sim.process(parent(sim))
    sim.run()
    assert p.value == "boom"


def test_unhandled_process_error_propagates() -> None:
    sim = Simulation(seed=1)

    def bad(sim: Simulation) -> Any:
        yield 1
        raise RuntimeError("model bug")

    sim.process(bad(sim))
    with pytest.raises(RuntimeError, match="model bug"):
        sim.run()


def test_yielding_garbage_is_a_type_error() -> None:
    sim = Simulation(seed=1)

    def bad(sim: Simulation) -> Any:
        yield "soon"

    sim.process(bad(sim))
    with pytest.raises(TypeError, match="yielded 'soon'"):
        sim.run()


def test_process_requires_generator() -> None:
    sim = Simulation(seed=1)

    def not_a_generator(sim: Simulation) -> None:
        return None

    with pytest.raises(TypeError, match="needs a generator"):
        sim.process(not_a_generator(sim))  # type: ignore[arg-type]


def test_interrupt() -> None:
    sim = Simulation(seed=1)
    log: list[Any] = []

    def worker(sim: Simulation) -> Any:
        try:
            yield 10
            log.append("finished")
        except Interrupt as i:
            log.append((sim.now, i.cause))
            yield 1
            log.append(("resumed", sim.now))

    w = sim.process(worker(sim))
    sim.call_at(lambda: w.interrupt("breakdown"), time=3)
    sim.run()
    assert log == [(3.0, "breakdown"), ("resumed", 4.0)]
    with pytest.raises(EventStateError):
        w.interrupt()


def test_signal_and_conditions() -> None:
    sim = Simulation(seed=1)
    go = sim.signal("go")
    out: dict[str, Any] = {}

    def waiter(sim: Simulation) -> Any:
        v = yield go
        out["signal"] = (sim.now, v)
        t1, t2 = sim.timeout(2, "a"), sim.timeout(5, "b")
        res = yield sim.all_of(t1, t2)
        out["all"] = (sim.now, sorted(res.values()))
        t3, t4 = sim.timeout(1, "fast"), sim.timeout(9, "slow")
        res = yield sim.any_of(t3, t4)
        out["any"] = (sim.now, list(res.values()))

    sim.process(waiter(sim))
    sim.call_at(lambda: go.succeed("green"), time=7)
    sim.run(until=100)
    assert out == {"signal": (7.0, "green"), "all": (12.0, ["a", "b"]), "any": (13.0, ["fast"])}
    with pytest.raises(EventStateError):
        go.succeed()


def test_signal_failure_raises_in_waiter() -> None:
    sim = Simulation(seed=1)
    sig = sim.signal()
    got: list[str] = []

    def waiter(sim: Simulation) -> Any:
        try:
            yield sig
        except KeyError as e:
            got.append(str(e))

    sim.process(waiter(sim))
    sim.call_at(lambda: sig.fail(KeyError("x")), time=1)
    sim.run()
    assert got == ["'x'"]


def test_processes_start_in_creation_order() -> None:
    sim = Simulation(seed=1)
    order: list[int] = []

    def p(sim: Simulation, i: int) -> Any:
        order.append(i)
        yield 0

    for i in range(5):
        sim.process(p(sim, i))
    sim.run()
    assert order == [0, 1, 2, 3, 4]


def test_blocked_processes_produce_warning() -> None:
    sim = Simulation(seed=1)
    never = sim.signal()

    def stuck(sim: Simulation) -> Any:
        yield never

    sim.process(stuck(sim), name="stuck")
    result = sim.run(until=10)
    assert any("blocked" in w and "stuck" in w for w in result.warnings)
