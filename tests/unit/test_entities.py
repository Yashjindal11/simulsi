from __future__ import annotations

import pytest

from simulsi import Entity, EventStateError, Simulation


def test_entities_get_reproducible_per_simulation_ids() -> None:
    ids = []
    for _ in range(2):
        sim = Simulation(seed=1)
        ids.append([sim.entity("customer").id for _ in range(3)] + [sim.entity("job").id])
    assert ids[0] == ids[1] == ["customer-1", "customer-2", "customer-3", "job-1"]


def test_entity_attributes_and_state_history() -> None:
    sim = Simulation(seed=1, trace=True)
    c = sim.entity("customer", priority=2, service_type="premium")
    assert c["priority"] == 2 and c.get("missing", 0) == 0
    sim.schedule(time=3, callback=lambda s, e: c.set_state("waiting"))
    sim.schedule(time=5, callback=lambda s, e: c.set_state("served", by="teller-1"))
    sim.schedule(time=9, callback=lambda s, e: s.dispose(c))
    sim.run()
    assert [(h.time, h.old_state, h.new_state) for h in c.history] == [
        (3.0, "created", "waiting"),
        (5.0, "waiting", "served"),
        (9.0, "served", "disposed"),
    ]
    assert c.history[1].metadata == {"by": "teller-1"}
    assert c.time_in_system == 9.0
    assert c.time_in_states() == {"created": 3.0, "waiting": 2.0, "served": 4.0, "disposed": 0.0}
    assert not c.alive and c.id not in sim.entities
    assert sim.log is not None
    assert [r.event_type for r in sim.log if r.entity == c.id][:2] == [
        "entity.created",
        "entity.state",
    ]


def test_standalone_entity_then_added() -> None:
    e = Entity(entity_type="aircraft", attributes={"tail": "N1"})
    assert e.id.startswith("aircraft-x")
    sim = Simulation(seed=1)
    sim.add_entity(e)
    assert e.id == "aircraft-1"
    with pytest.raises(EventStateError):
        sim.add_entity(e)


def test_explicit_ids_must_be_unique() -> None:
    sim = Simulation(seed=1)
    sim.add_entity(Entity("truck", id="T1"))
    with pytest.raises(EventStateError):
        sim.add_entity(Entity("truck", id="T1"))


def test_double_dispose_raises() -> None:
    sim = Simulation(seed=1)
    e = sim.entity("x")
    sim.dispose(e)
    with pytest.raises(EventStateError):
        sim.dispose(e)


def test_entity_counts_in_metrics() -> None:
    sim = Simulation(seed=1)
    for _ in range(3):
        sim.entity("pkg")
    sim.dispose(next(iter(sim.entities.values())))
    m = sim.run().metrics
    assert m["entity.pkg.created"] == 3 and m["entity.pkg.disposed"] == 1
