from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from simulsi import Event, EventStateError, EventStatus, Priority, SchedulingError, Simulation
from simulsi.core.clock import Clock


def test_events_execute_in_time_order() -> None:
    sim = Simulation(seed=1)
    seen: list[tuple[float, str]] = []
    for t, name in [(5, "c"), (1, "a"), (3, "b")]:
        sim.schedule(
            time=t, event_type=name, callback=lambda s, e: seen.append((s.now, e.event_type))
        )
    sim.run()
    assert seen == [(1.0, "a"), (3.0, "b"), (5.0, "c")]


def test_same_time_events_follow_priority_then_scheduling_order() -> None:
    sim = Simulation(seed=1)
    seen: list[str] = []

    def record(s: Simulation, e: Event) -> None:
        seen.append(e.event_type)

    sim.schedule(Event("normal-1", callback=record), time=2)
    sim.schedule(Event("low", callback=record, priority=Priority.LOW), time=2)
    sim.schedule(Event("normal-2", callback=record), time=2)
    sim.schedule(Event("urgent", callback=record, priority=Priority.URGENT), time=2)
    sim.schedule(Event("high", callback=record, priority=Priority.HIGH), time=2)
    sim.run()
    assert seen == ["urgent", "high", "normal-1", "normal-2", "low"]


def test_cancel_and_reschedule() -> None:
    sim = Simulation(seed=1)
    seen: list[tuple[str, float]] = []
    a = sim.schedule(time=1, event_type="a", callback=lambda s, e: seen.append(("a", s.now)))
    b = sim.schedule(time=2, event_type="b", callback=lambda s, e: seen.append(("b", s.now)))
    assert sim.cancel(a)
    assert not sim.cancel(a)
    assert a.status is EventStatus.CANCELLED
    sim.reschedule(b, time=7)
    assert len(sim.queue) == 1
    sim.run()
    assert seen == [("b", 7.0)]
    assert b.status is EventStatus.EXECUTED
    with pytest.raises(EventStateError):
        sim.reschedule(b, time=9)


def test_reschedule_cancelled_event_revives_it() -> None:
    sim = Simulation(seed=1)
    ev = sim.schedule(time=1, event_type="x")
    sim.cancel(ev)
    sim.reschedule(ev, delay=4)
    assert len(sim.queue) == 1
    sim.run()
    assert ev.status is EventStatus.EXECUTED and ev.timestamp == 4


def test_cannot_schedule_in_the_past_or_twice() -> None:
    sim = Simulation(seed=1)
    sim.run(until=10)
    with pytest.raises(SchedulingError):
        sim.schedule(time=5, event_type="late")
    with pytest.raises(SchedulingError):
        sim.schedule(delay=-1, event_type="neg")
    with pytest.raises(SchedulingError):
        sim.schedule(time=float("nan"), event_type="nan")
    ev = sim.schedule(delay=1, event_type="ok")
    with pytest.raises(EventStateError):
        sim.schedule(ev, delay=2)
    with pytest.raises(SchedulingError):
        sim.schedule(time=1, delay=1)


def test_run_stops_exactly_at_until_and_includes_boundary_events() -> None:
    sim = Simulation(seed=1)
    seen: list[float] = []
    for t in (2, 10, 11):
        sim.schedule(time=t, callback=lambda s, e: seen.append(s.now))
    result = sim.run(until=10)
    assert result.end_time == 10.0 and sim.now == 10.0
    assert seen == [2.0, 10.0]
    sim.run(until=20)
    assert seen == [2.0, 10.0, 11.0]
    assert sim.now == 20.0


def test_run_until_advances_clock_even_when_idle() -> None:
    sim = Simulation(seed=1)
    assert sim.run(until=42.5).end_time == 42.5


def test_stop_and_max_events() -> None:
    sim = Simulation(seed=1)
    for t in range(1, 6):
        sim.schedule(time=t, callback=(lambda s, e: s.stop()) if t == 3 else None)
    sim.run(until=100)
    assert sim.now == 3.0
    sim.run(max_events=1)
    assert sim.now == 4.0


def test_subclassed_event() -> None:
    class Arrival(Event):
        def __init__(self, n: int) -> None:
            super().__init__("arrival", {"n": n})

        def execute(self, sim: Simulation) -> None:
            if self.payload["n"] < 3:
                sim.schedule(Arrival(self.payload["n"] + 1), delay=1)

    sim = Simulation(seed=1, trace=True)
    sim.schedule(Arrival(0), time=0)
    sim.run()
    assert sim.now == 3.0
    assert sim.log is not None
    assert [r.metadata["n"] for r in sim.log] == [0, 1, 2, 3]


def test_event_ids_are_unique_and_sequential() -> None:
    sim = Simulation(seed=1)
    ids = [sim.schedule(delay=i).event_id for i in range(5)]
    assert ids == [0, 1, 2, 3, 4]


def test_timedelta_and_datetime_scheduling() -> None:
    epoch = datetime(2026, 1, 1, 6, 0)
    sim = Simulation(seed=1, epoch=epoch, time_unit=timedelta(minutes=1))
    a = sim.schedule(delay=timedelta(hours=1))
    b = sim.schedule(time=datetime(2026, 1, 1, 6, 30))
    assert a.timestamp == 60.0 and b.timestamp == 30.0
    sim.run(until=timedelta(hours=2))
    assert sim.now == 120.0
    assert sim.now_datetime == datetime(2026, 1, 1, 8, 0)


def test_datetime_without_epoch_is_rejected() -> None:
    with pytest.raises(SchedulingError):
        Simulation(seed=1).schedule(time=datetime(2026, 1, 1))


def test_clock_validation() -> None:
    with pytest.raises(ValueError):
        Clock(unit=timedelta(0))
    with pytest.raises(TypeError):
        Clock().duration("5")  # type: ignore[arg-type]
    clock = Clock(5)
    with pytest.raises(SchedulingError):
        clock.advance_to(4)


def test_snapshot_lists_pending_events() -> None:
    sim = Simulation(seed=3)
    sim.schedule(time=4, event_type="later")
    sim.schedule(time=1, event_type="sooner")
    snap = sim.snapshot()
    assert snap["pending_event_count"] == 2
    assert [e["event_type"] for e in snap["pending_events"]] == ["sooner", "later"]


def test_call_at_is_not_logged() -> None:
    sim = Simulation(seed=1, trace=True)
    calls: list[float] = []
    sim.call_at(lambda: calls.append(sim.now), delay=3)
    sim.run()
    assert calls == [3.0]
    assert sim.log is not None and len(sim.log) == 0
