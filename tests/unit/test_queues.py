from __future__ import annotations

from typing import Any

import pytest

from simulsi import CapacityError, Queue, Simulation
from simulsi.queues import OrderedBuffer


def test_ordered_buffer_disciplines() -> None:
    for disc, expected in [("fifo", "abc"), ("lifo", "cba"), ("priority", "bca")]:
        buf: OrderedBuffer[str] = OrderedBuffer(disc)  # type: ignore[arg-type]
        for item, prio in [("a", 3), ("b", 1), ("c", 2)]:
            buf.push(item, prio)
        assert "".join(buf) == expected
        assert "".join(buf.pop() for _ in range(3)) == expected


def test_ordered_buffer_remove_and_errors() -> None:
    buf: OrderedBuffer[str] = OrderedBuffer()
    buf.push("a")
    buf.push("b")
    assert buf.remove("a") and not buf.remove("a")
    assert len(buf) == 1 and buf.peek() == "b"
    buf.pop()
    with pytest.raises(IndexError):
        buf.pop()
    with pytest.raises(ValueError):
        OrderedBuffer("random")  # type: ignore[arg-type]


def test_blocking_get_and_put() -> None:
    sim = Simulation(seed=1)
    q = sim.queue("orders", capacity=2)
    got: list[tuple[float, Any]] = []
    put_times: list[float] = []

    def producer(sim: Simulation) -> Any:
        for i in range(4):
            yield q.put(i)
            put_times.append(sim.now)

    def consumer(sim: Simulation) -> Any:
        yield 5
        for _ in range(4):
            item = yield q.get()
            got.append((sim.now, item))
            yield 1

    sim.process(producer(sim))
    sim.process(consumer(sim))
    m = sim.run().metrics
    assert put_times == [0.0, 0.0, 5.0, 6.0]
    assert got == [(5.0, 0), (6.0, 1), (7.0, 2), (8.0, 3)]
    assert m["queue.orders.puts"] == 4 and m["queue.orders.gets"] == 4
    assert m["queue.orders.max_length"] == 2


def test_getter_waits_for_item_and_handover_has_zero_wait() -> None:
    sim = Simulation(seed=1)
    q = sim.queue("q")
    got: list[float] = []

    def consumer(sim: Simulation) -> Any:
        yield q.get()
        got.append(sim.now)

    sim.process(consumer(sim))
    sim.call_at(lambda: q.put("x"), time=3)
    m = sim.run().metrics
    assert got == [3.0]
    assert m["queue.q.wait.mean"] == 0.0


def test_priority_queue_wait_times_and_accounting() -> None:
    sim = Simulation(seed=1)
    q: Queue[str] = Queue("triage", discipline="priority", sim=sim)
    q.put("minor", priority=3)
    q.put("major", priority=1)
    q.put("drop", priority=2)
    sim.run(until=2)
    assert q.items == ["major", "drop", "minor"]
    assert q.remove("drop")
    assert q.try_get() == "major"
    s = q.summary()
    assert s["puts"] == s["gets"] + s["removed"] + s["length_now"]
    assert s["wait"]["mean"] == 2.0
    assert q.try_get() == "minor" and q.try_get() is None


def test_cancel_pending_get() -> None:
    sim = Simulation(seed=1)
    q = sim.queue("q")
    g = q.get()
    assert q.cancel(g)
    q.put(1)
    assert len(q) == 1


def test_queue_validation() -> None:
    with pytest.raises(CapacityError):
        Queue("bad", capacity=0)
