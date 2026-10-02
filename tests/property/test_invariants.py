"""Property-based tests of engine invariants."""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from simulsi import Event, Simulation
from simulsi.queues import OrderedBuffer
from simulsi.randomness import Binomial, Categorical, Exponential, RandomStream, Uniform
from simulsi.statistics import proportion_ci

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

times = st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)


@SETTINGS
@given(st.lists(st.tuples(times, st.integers(-5, 5)), min_size=1, max_size=200))
def test_events_execute_in_time_priority_fifo_order(specs: list[tuple[float, int]]) -> None:
    sim = Simulation(seed=0)
    executed: list[tuple[float, int, int]] = []
    for i, (t, prio) in enumerate(specs):
        sim.schedule(
            Event(
                "e",
                {"i": i},
                lambda s, e: executed.append((e.timestamp, e.priority, e.payload["i"])),
                priority=prio,
            ),
            time=t,
        )
    sim.run()
    assert executed == sorted(executed)  # time, then priority, then scheduling order
    assert len(executed) == len(specs)


@SETTINGS
@given(st.lists(times, min_size=1, max_size=100), st.data())
def test_cancelled_events_never_run(ts: list[float], data: st.DataObject) -> None:
    sim = Simulation(seed=0)
    ran: set[int] = set()
    events = [sim.schedule(time=t, callback=lambda s, e: ran.add(e.event_id)) for t in ts]
    cancel = data.draw(st.sets(st.integers(0, len(events) - 1)))
    for i in cancel:
        sim.cancel(events[i])
    sim.run()
    assert ran == {e.event_id for i, e in enumerate(events) if i not in cancel}
    assert len(sim.event_queue) == 0


@SETTINGS
@given(
    capacity=st.integers(1, 5),
    jobs=st.lists(st.tuples(st.floats(0, 50), st.floats(0.01, 20)), min_size=1, max_size=40),
    until=st.floats(1, 200),
)
def test_resource_invariants(capacity: int, jobs: list[tuple[float, float]], until: float) -> None:
    sim = Simulation(seed=0)
    r = sim.resource("r", capacity)
    violations: list[str] = []

    def check(sim: Simulation) -> None:
        if not 0 <= r.in_use <= r.capacity:
            violations.append(f"in_use={r.in_use}")
        if r.queue_size < 0:
            violations.append("negative queue")

    def job(sim: Simulation, arrive: float, service: float) -> Any:
        yield arrive
        req = yield sim.request(r)
        check(sim)
        yield service
        sim.release(req)
        check(sim)

    for a, s in jobs:
        sim.process(job(sim, a, s))
    m = sim.run(until=until).metrics
    assert not violations
    assert 0.0 <= m["resource.r.utilization"] <= 1.0 + 1e-12
    assert m["resource.r.mean_queue_length"] >= 0
    # accounting: every request is granted, waiting or (here never) reneged
    assert m["resource.r.requests"] == m["resource.r.grants"] + r.queue_size
    assert m["resource.r.grants"] == m["resource.r.releases"] + r.in_use


@SETTINGS
@given(
    ops=st.lists(
        st.one_of(st.tuples(st.just("put"), st.integers(0, 9)), st.just(("get", 0))), max_size=80
    ),
    discipline=st.sampled_from(["fifo", "lifo", "priority"]),
)
def test_queue_conservation(ops: list[tuple[str, int]], discipline: str) -> None:
    sim = Simulation(seed=0)
    q = sim.queue("q", discipline=discipline)  # type: ignore[arg-type]
    for op, prio in ops:
        if op == "put":
            q.put(object(), priority=prio)
        else:
            q.try_get()
    s = q.summary()
    assert s["puts"] == s["gets"] + s["removed"] + s["length_now"]
    assert s["length_now"] >= 0 and s["mean_length"] >= 0


@SETTINGS
@given(st.lists(st.tuples(st.integers(0, 3), st.integers()), max_size=60))
def test_priority_buffer_pops_sorted_and_stable(items: list[tuple[int, int]]) -> None:
    buf: OrderedBuffer[tuple[int, int, int]] = OrderedBuffer("priority")
    for i, (prio, payload) in enumerate(items):
        buf.push((prio, i, payload), prio)
    popped = [buf.pop() for _ in range(len(items))]
    assert popped == sorted(popped)


@SETTINGS
@given(st.integers(0, 2**63 - 1), st.integers(1, 50))
def test_streams_are_reproducible_for_any_seed(seed: int, n: int) -> None:
    a, b = RandomStream(seed), RandomStream(seed)
    d = Exponential(rate=2.0)
    assert [d.sample(a) for _ in range(n)] == [d.sample(b) for _ in range(n)]


@SETTINGS
@given(st.floats(-1e6, 1e6), st.floats(0, 1e6), st.integers(0, 2**32))
def test_uniform_samples_within_bounds(low: float, width: float, seed: int) -> None:
    d = Uniform(low, low + width)
    xs = d.sample_n(RandomStream(seed), 50)
    assert ((xs >= low) & (xs <= low + width)).all()


@SETTINGS
@given(st.integers(0, 50), st.floats(0, 1), st.integers(0, 2**32))
def test_binomial_within_support(n: int, p: float, seed: int) -> None:
    xs = Binomial(n, p).sample_n(RandomStream(seed), 30)
    assert ((xs >= 0) & (xs <= n)).all()


@SETTINGS
@given(st.lists(st.floats(0.01, 10), min_size=1, max_size=8))
def test_categorical_probabilities_valid(weights: list[float]) -> None:
    total = sum(weights)
    probs = [w / total for w in weights]
    probs[-1] = 1 - sum(probs[:-1])
    if probs[-1] < 0:
        return
    d = Categorical(list(range(len(probs))), probs)
    assert all(0 <= p <= 1 for p in d.probabilities)


@SETTINGS
@given(st.integers(0, 1000), st.integers(1, 1000))
def test_proportion_interval_within_unit(k: int, n: int) -> None:
    k = min(k, n)
    lo, hi = proportion_ci(k, n)
    assert 0 <= lo <= k / n <= hi <= 1


@SETTINGS
@given(st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=100))
def test_entity_ids_unique(types: list[str]) -> None:
    sim = Simulation(seed=0)
    ids = [sim.entity(t).id for t in types]
    assert len(set(ids)) == len(ids)


@SETTINGS
@given(st.floats(0.1, 1e4), st.lists(times, max_size=30))
def test_run_stops_exactly_at_until(until: float, ts: list[float]) -> None:
    sim = Simulation(seed=0)
    seen: list[float] = []
    for t in ts:
        sim.schedule(time=t, callback=lambda s, e: seen.append(s.now))
    res = sim.run(until=until)
    assert res.end_time == until
    assert all(t <= until for t in seen)
    assert sorted(seen) == sorted(t for t in ts if t <= until)
