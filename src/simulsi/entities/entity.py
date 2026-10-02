"""Entities: the things that flow through a simulated system."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation

_standalone_ids = itertools.count(1)


@dataclass(frozen=True, slots=True)
class StateChange:
    time: float
    old_state: str
    new_state: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Entity:
    """A customer, job, patient, aircraft, packet ... anything with identity and state.

    Create entities with :meth:`Simulation.entity` so that ids are assigned
    per simulation (``"customer-1"``, ``"customer-2"``, ...) and therefore
    reproducible. Entities built directly get a process-wide id unless one
    is given, and are attached when passed to :meth:`Simulation.add_entity`.
    """

    __slots__ = (
        "_auto_id",
        "_sim",
        "attributes",
        "created_at",
        "disposed_at",
        "entity_type",
        "history",
        "id",
        "state",
    )

    def __init__(
        self,
        entity_type: str = "entity",
        attributes: dict[str, Any] | None = None,
        *,
        id: str | None = None,
        created_at: float = 0.0,
        state: str = "created",
    ) -> None:
        self.entity_type = entity_type
        self.attributes: dict[str, Any] = {} if attributes is None else dict(attributes)
        self._auto_id = id is None
        self.id = id if id is not None else f"{entity_type}-x{next(_standalone_ids)}"
        self.created_at = created_at
        self.state = state
        self.history: list[StateChange] = []
        self.disposed_at: float | None = None
        self._sim: Simulation | None = None

    def __getitem__(self, key: str) -> Any:
        return self.attributes[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def set_state(self, new_state: str, *, time: float | None = None, **metadata: Any) -> None:
        """Transition to ``new_state``, recording history and the event log."""
        sim = self._sim
        if time is None:
            time = sim.now if sim is not None else self.created_at
        old = self.state
        self.state = new_state
        if sim is None or sim.keep_entity_history:
            self.history.append(StateChange(time, old, new_state, metadata))
        if sim is not None:
            sim._on_entity_state(self, old, new_state, metadata)

    @property
    def alive(self) -> bool:
        return self.disposed_at is None

    @property
    def time_in_system(self) -> float | None:
        if self.disposed_at is None:
            return None
        return self.disposed_at - self.created_at

    def time_in_states(self, until: float | None = None) -> dict[str, float]:
        """Total time spent in each state according to the recorded history."""
        totals: dict[str, float] = {}
        if not self.history:
            return totals
        end = until if until is not None else self.disposed_at
        if end is None and self._sim is not None:
            end = self._sim.now
        state, start = self.history[0].old_state, self.created_at
        for change in self.history:
            totals[state] = totals.get(state, 0.0) + (change.time - start)
            state, start = change.new_state, change.time
        if end is not None:
            totals[state] = totals.get(state, 0.0) + max(0.0, end - start)
        return totals

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "state": self.state,
            "created_at": self.created_at,
            "attributes": dict(self.attributes),
        }

    def __repr__(self) -> str:
        return f"Entity(id={self.id!r}, state={self.state!r})"
