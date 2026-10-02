from __future__ import annotations

import contextlib
from typing import Any

import pytest

from simulsi import CapacityError, Interrupt, Request, Resource, ResourceUsageError, Simulation


def customer(
    sim: Simulation, res: Resource, arrive: float, service: float, log: list[Any], name: str
) -> Any:
    yield arrive
    req = yield sim.request(res)
    log.append((name, "start", sim.now, req.waiting_time))
    yield service
    sim.release(res)
    log.append((name, "end", sim.now))


def test_fifo_resource_serves_in_arrival_order() -> None:
    sim = Simulation(seed=1)
    server = sim.resource("server", capacity=1)
    log: list[Any] = []
    for i, arrive in enumerate([0, 1, 2]):
        sim.process(customer(sim, server, arrive, 5, log, f"c{i}"))
    sim.run()
    starts = [(e[0], e[2], e[3]) for e in log if e[1] == "start"]
    assert starts == [("c0", 0.0, 0.0), ("c1", 5.0, 4.0), ("c2", 10.0, 8.0)]
    m = sim.run().metrics
    assert m["resource.server.utilization"] == pytest.approx(1.0)
    assert m["resource.server.wait.mean"] == pytest.approx(4.0)
    assert m["resource.server.grants"] == 3 and m["resource.server.releases"] == 3


def test_utilization_and_queue_length_time_weighted() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("r", capacity=2)
    log: list[Any] = []
    sim.process(customer(sim, r, 0, 4, log, "a"))
    sim.process(customer(sim, r, 0, 4, log, "b"))
    sim.process(customer(sim, r, 0, 4, log, "c"))
    m = sim.run(until=10).metrics
    # busy: 2 units for [0,4], 1 unit for [4,8], 0 for [8,10] -> area 12 over capacity area 20
    assert m["resource.r.utilization"] == pytest.approx(12 / 20)
    # queue: 1 waiting during [0,4] -> mean 0.4
    assert m["resource.r.mean_queue_length"] == pytest.approx(0.4)
    assert m["resource.r.max_queue_length"] == 1


def test_priority_discipline() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("doctor", capacity=1, discipline="priority")
    order: list[str] = []

    def patient(sim: Simulation, name: str, prio: int, arrive: float) -> Any:
        yield arrive
        yield sim.request(r, priority=prio)
        order.append(name)
        yield 10
        sim.release(r)

    sim.process(patient(sim, "first", 5, 0))
    sim.process(patient(sim, "routine", 3, 1))
    sim.process(patient(sim, "critical", 1, 2))
    sim.process(patient(sim, "routine2", 3, 3))
    sim.run()
    assert order == ["first", "critical", "routine", "routine2"]


def test_lifo_and_custom_disciplines() -> None:
    for discipline, expected in [
        ("lifo", ["a", "d", "c", "b"]),
        (lambda r: -r.priority, ["a", "d", "c", "b"]),
    ]:
        sim = Simulation(seed=1)
        r = sim.resource("r", capacity=1, discipline=discipline)  # type: ignore[arg-type]
        order: list[str] = []

        def job(
            sim: Simulation,
            name: str,
            arrive: float,
            prio: int,
            r: Resource = r,
            order: list[str] = order,
        ) -> Any:
            yield arrive
            yield sim.request(r, priority=prio)
            order.append(name)
            yield 10
            sim.release(r)

        for i, name in enumerate("abcd"):
            sim.process(job(sim, name, i, i))
        sim.run()
        assert order == expected


def test_reneging_with_patience() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("r", capacity=1)
    out: list[Any] = []

    def holder(sim: Simulation) -> Any:
        yield sim.request(r)
        yield 10
        sim.release(r)

    def impatient(sim: Simulation) -> Any:
        req = yield sim.request(r, patience=3)
        out.append((sim.now, req.granted, req.reneged))

    sim.process(holder(sim))
    sim.process(impatient(sim))
    m = sim.run().metrics
    assert out == [(3.0, False, True)]
    assert m["resource.r.reneges"] == 1
    assert r.queue_size == 0


def test_release_errors() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("r", capacity=1)
    other = sim.resource("o", capacity=1)
    req = r.request()
    with pytest.raises(ResourceUsageError):
        other.release(req)
    r.release(req)
    with pytest.raises(ResourceUsageError):
        r.release(req)
    with pytest.raises(ResourceUsageError):
        sim.release(r)
    req2 = r.request()
    req3 = r.request()
    with pytest.raises(ResourceUsageError, match="never granted"):
        r.release(req3)
    req3.cancel()
    assert r.queue_size == 0
    r.release(req2)


def test_capacity_validation() -> None:
    for bad in (-1, 1.5, True):
        with pytest.raises(CapacityError):
            Resource("x", bad)  # type: ignore[arg-type]
    r = Simulation(seed=1).resource("x", 0)
    with pytest.raises(CapacityError):
        r.set_capacity(-2)


def test_resource_cannot_be_shared_between_simulations() -> None:
    r = Resource("shared", 1)
    Simulation(seed=1).add_resource(r)
    with pytest.raises(ResourceUsageError):
        Simulation(seed=2).add_resource(r)


def test_duplicate_resource_names_rejected() -> None:
    sim = Simulation(seed=1)
    sim.resource("x")
    with pytest.raises(ResourceUsageError):
        sim.resource("x")


def test_capacity_change_and_failure_affect_service() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("machine", capacity=1)
    starts: list[float] = []

    def job(sim: Simulation) -> Any:
        yield sim.request(r)
        starts.append(sim.now)
        yield 1
        sim.release(r)

    sim.call_at(lambda: r.fail(), time=0)
    sim.call_at(lambda: r.repair(), time=5)
    for _ in range(2):
        sim.process(job(sim))
    m = sim.run(until=10).metrics
    assert starts == [5.0, 6.0]
    assert m["resource.machine.availability"] == pytest.approx(0.5)
    assert m["resource.machine.failures"] == 1


def test_fail_with_interrupt_hits_current_user() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("machine", capacity=1)
    log: list[Any] = []

    def job(sim: Simulation) -> Any:
        req = yield sim.request(r)
        try:
            yield 10
        except Interrupt as i:
            log.append((sim.now, i.cause))
            r.release(req)
            return
        r.release(req)

    sim.process(job(sim))
    sim.call_at(lambda: r.fail(interrupt=True, cause="jam"), time=4)
    sim.run()
    assert log == [(4.0, "jam")]
    assert r.in_use == 0


def test_use_helper_releases_on_interrupt() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("r", capacity=1)

    def job(sim: Simulation) -> Any:
        with contextlib.suppress(Interrupt):
            yield from sim.use(r, 10)

    p = sim.process(job(sim))
    sim.call_at(lambda: p.interrupt(), time=2)
    sim.run()
    assert r.in_use == 0 and r.releases.value == 1


def test_interrupting_a_waiting_request_withdraws_it() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("r", capacity=1)

    def holder(sim: Simulation) -> Any:
        yield from sim.use(r, 10)

    def waiter(sim: Simulation) -> Any:
        try:
            yield sim.request(r)
        except Interrupt:
            return "gave up"

    sim.process(holder(sim))
    w = sim.process(waiter(sim))
    sim.call_at(lambda: w.interrupt(), time=1)
    sim.run()
    assert w.value == "gave up"
    assert r.queue_size == 0 and r.grants.value == 1


def test_unreleased_resource_warning() -> None:
    sim = Simulation(seed=1)
    r = sim.resource("r", capacity=1)

    def leaky(sim: Simulation) -> Any:
        yield sim.request(r)
        yield 1

    sim.process(leaky(sim), name="leaky")
    result = sim.run()
    assert any("leaky" in w and "'r'" in w for w in result.warnings)
    assert result.metrics["sim.unreleased_resources"] == 1


def test_request_object_release() -> None:
    sim = Simulation(seed=1)
    r = Resource("r", 1)

    def job(sim: Simulation) -> Any:
        req: Request = yield sim.request(r)
        yield 1
        req.release()

    sim.process(job(sim))
    sim.run()
    assert r.sim is sim and r.in_use == 0
