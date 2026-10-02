"""The simulation engine."""

from __future__ import annotations

import math
import time as _wall
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from simulsi.core.clock import Clock, DurationLike, TimeLike
from simulsi.core.trace import EventLog, LogRecord
from simulsi.entities.entity import Entity
from simulsi.errors import EventStateError, SchedulingError
from simulsi.events.event import Event, EventCallback, EventQueue, EventStatus, Priority, _CallEvent
from simulsi.randomness.stream import RandomStream


@dataclass
class SimulationResult:
    """Outcome of :meth:`Simulation.run`.

    ``metrics`` is a flat ``name -> value`` mapping (e.g.
    ``"resource.teller.utilization"``) suitable for experiments and export;
    ``details`` holds the nested per-component summaries.
    """

    seed: int
    start_time: float
    end_time: float
    events_processed: int
    wall_time: float
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    log: EventLog | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events_processed": self.events_processed,
            "wall_time": self.wall_time,
            "metrics": dict(self.metrics),
            "details": self.details,
            "warnings": list(self.warnings),
        }


class Simulation:
    """A discrete-event simulation.

    >>> sim = Simulation(seed=42)
    >>> sim.schedule(Event("ping"), time=5)  # doctest: +ELLIPSIS
    Event(...)
    >>> sim.run(until=10).end_time
    10.0
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        name: str = "simulation",
        start: float = 0.0,
        epoch: datetime | None = None,
        time_unit: timedelta = timedelta(minutes=1),
        trace: bool = False,
        max_log_records: int | None = None,
        keep_entity_history: bool = True,
    ) -> None:
        self.name = name
        self.clock = Clock(start, epoch=epoch, unit=time_unit)
        self.rng = RandomStream(seed)
        self.seed = self.rng.seed
        self.queue = EventQueue()
        self.log: EventLog | None = EventLog(max_log_records) if trace else None
        self.keep_entity_history = keep_entity_history
        self.events_processed = 0
        self.warnings: list[str] = []
        self.entities: dict[str, Entity] = {}
        self._entity_counts: dict[str, int] = {}
        self._disposed_counts: dict[str, int] = {}
        self._next_event_id = 0
        self._stop_requested = False
        self._finish_hooks: list[Callable[[Simulation], None]] = []
        self._wall_time = 0.0

    # -- time --------------------------------------------------------------

    @property
    def now(self) -> float:
        return self.clock.now

    @property
    def now_datetime(self) -> datetime:
        return self.clock.to_datetime()

    def stream(self, name: str) -> RandomStream:
        """A named, independent random stream derived from the simulation seed."""
        return self.rng.stream(name)

    # -- scheduling --------------------------------------------------------

    def _resolve_time(self, time: TimeLike | None, delay: DurationLike | None) -> float:
        if (time is None) == (delay is None):
            raise SchedulingError("give exactly one of time= or delay=")
        if time is not None:
            t = self.clock.to_time(time)
        else:
            assert delay is not None
            d = self.clock.duration(delay)
            if d < 0:
                raise SchedulingError(f"delay must be >= 0, got {d}")
            t = self.clock.now + d
        if math.isnan(t) or math.isinf(t):
            raise SchedulingError(f"event time must be finite, got {t}")
        if t < self.clock.now:
            raise SchedulingError(f"cannot schedule in the past: {t} < now={self.clock.now}")
        return t

    def schedule(
        self,
        event: Event | None = None,
        *,
        time: TimeLike | None = None,
        delay: DurationLike | None = None,
        priority: int | None = None,
        callback: EventCallback | None = None,
        event_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """Schedule ``event`` at an absolute ``time`` or after ``delay``.

        Without an event, one is created from ``callback``/``event_type``/``payload``.
        """
        if event is None:
            event = Event(event_type or "event", payload, callback)
        elif callback is not None or event_type is not None or payload is not None:
            raise SchedulingError("callback/event_type/payload are only used when event is None")
        if event.status is EventStatus.SCHEDULED:
            raise EventStateError(f"{event!r} is already scheduled; use reschedule()")
        if priority is not None:
            event.priority = int(priority)
        t = self._resolve_time(time, delay)
        if event.event_id < 0:
            event.event_id = self._next_event_id
            self._next_event_id += 1
        self.queue.push(event, t)
        return event

    def call_at(
        self,
        fn: Callable[[], Any],
        *,
        time: TimeLike | None = None,
        delay: DurationLike | None = None,
        priority: int = Priority.NORMAL,
        label: str = "call",
    ) -> Event:
        """Schedule a plain function call. Lighter than a full :class:`Event`."""
        return self.schedule(_CallEvent(fn, label, priority), time=time, delay=delay)

    def reschedule(
        self,
        event: Event,
        *,
        time: TimeLike | None = None,
        delay: DurationLike | None = None,
        priority: int | None = None,
    ) -> Event:
        """Move a pending (or cancelled) event to a new time."""
        if event.status is EventStatus.EXECUTED:
            raise EventStateError(f"{event!r} already executed; schedule a new event instead")
        if event.event_id < 0:
            raise EventStateError(f"{event!r} was never scheduled")
        if priority is not None:
            event.priority = int(priority)
        self.queue.push(event, self._resolve_time(time, delay))
        return event

    def cancel(self, event: Event) -> bool:
        """Cancel a pending event. Returns ``False`` if it was not pending."""
        if event.status is not EventStatus.SCHEDULED:
            return False
        self.queue.discard(event)
        return True

    def pending_events(self, limit: int | None = None) -> list[Event]:
        events = list(self.queue)
        return events if limit is None else events[:limit]

    def peek(self) -> float:
        """Time of the next pending event (``inf`` if none)."""
        return self.queue.peek_time()

    # -- execution ---------------------------------------------------------

    def step(self) -> Event | None:
        """Execute the next event. Returns it, or ``None`` if the queue is empty."""
        event = self.queue.pop()
        if event is None:
            return None
        self.clock.advance_to(event.timestamp)
        event.status = EventStatus.EXECUTED
        self.events_processed += 1
        if self.log is not None and not event.internal:
            self.log.append(
                LogRecord(event.timestamp, event.event_type, metadata=dict(event.payload))
            )
        event.execute(self)
        return event

    def stop(self) -> None:
        """Ask :meth:`run` to return after the current event."""
        self._stop_requested = True

    def run(
        self,
        until: TimeLike | None = None,
        *,
        max_events: int | None = None,
    ) -> SimulationResult:
        """Process events in order.

        With ``until``, every event whose timestamp is ``<= until`` runs and the
        clock then stands exactly at ``until`` (even if the queue emptied
        earlier). Without it, the run lasts until no events remain. The run
        can be resumed by calling :meth:`run` again with a later ``until``.
        """
        horizon = math.inf
        if until is not None:
            horizon = self.clock.to_time(until)
            if horizon < self.clock.now:
                raise SchedulingError(f"until={horizon} is before now={self.clock.now}")
        start = self.clock.now
        wall0 = _wall.perf_counter()
        self._stop_requested = False
        executed = 0
        queue = self.queue
        while not self._stop_requested:
            if max_events is not None and executed >= max_events:
                break
            if queue.peek_time() > horizon:
                break
            if self.step() is None:
                break
            executed += 1
        stopped_early = self._stop_requested or (max_events is not None and executed >= max_events)
        if not stopped_early and math.isfinite(horizon):
            self.clock.advance_to(horizon)
        self._wall_time += _wall.perf_counter() - wall0
        return self._finish(start)

    def on_finish(self, hook: Callable[[Simulation], None]) -> None:
        """Register a function called whenever :meth:`run` returns (e.g. to set final metrics)."""
        self._finish_hooks.append(hook)

    def _finish(self, start: float) -> SimulationResult:
        for hook in self._finish_hooks:
            hook(self)
        return SimulationResult(
            seed=self.seed,
            start_time=start,
            end_time=self.clock.now,
            events_processed=self.events_processed,
            wall_time=self._wall_time,
            metrics=self._flat_metrics(),
            details=self._details(),
            warnings=list(self.warnings),
            log=self.log,
        )

    def _flat_metrics(self) -> dict[str, float]:
        flat: dict[str, float] = {"sim.events": float(self.events_processed)}
        for etype, n in sorted(self._entity_counts.items()):
            flat[f"entity.{etype}.created"] = float(n)
            flat[f"entity.{etype}.disposed"] = float(self._disposed_counts.get(etype, 0))
        return flat

    def _details(self) -> dict[str, Any]:
        return {
            "entities": {
                etype: {"created": n, "disposed": self._disposed_counts.get(etype, 0)}
                for etype, n in sorted(self._entity_counts.items())
            }
        }

    # -- entities ----------------------------------------------------------

    def entity(self, entity_type: str = "entity", /, **attributes: Any) -> Entity:
        """Create and register an entity with a reproducible id like ``"customer-7"``."""
        return self.add_entity(Entity(entity_type, attributes))

    def add_entity(self, entity: Entity) -> Entity:
        """Register an entity. Auto-generated ids are replaced with per-simulation ones."""
        if entity._sim is not None:
            raise EventStateError(f"{entity!r} is already registered with a simulation")
        if not entity._auto_id and entity.id in self.entities:
            raise EventStateError(f"duplicate entity id {entity.id!r}")
        n = self._entity_counts.get(entity.entity_type, 0) + 1
        self._entity_counts[entity.entity_type] = n
        if entity._auto_id:
            entity.id = f"{entity.entity_type}-{n}"
            entity._auto_id = False
        entity.created_at = self.clock.now
        entity._sim = self
        self.entities[entity.id] = entity
        if self.log is not None:
            self.log.append(
                LogRecord(
                    self.clock.now, "entity.created", entity=entity.id, new_state=entity.state
                )
            )
        return entity

    def dispose(self, entity: Entity, state: str = "disposed") -> None:
        """Mark an entity as having left the system."""
        if entity.disposed_at is not None:
            raise EventStateError(f"{entity!r} was already disposed")
        if entity.state != state:
            entity.set_state(state)
        entity.disposed_at = self.clock.now
        self.entities.pop(entity.id, None)
        self._disposed_counts[entity.entity_type] = (
            self._disposed_counts.get(entity.entity_type, 0) + 1
        )
        self._on_entity_disposed(entity)

    def _on_entity_disposed(self, entity: Entity) -> None:
        """Hook for metrics (time in system)."""

    def _on_entity_state(
        self, entity: Entity, old: str, new: str, metadata: dict[str, Any]
    ) -> None:
        if self.log is not None:
            self.log.append(
                LogRecord(
                    self.clock.now,
                    "entity.state",
                    entity=entity.id,
                    old_state=old,
                    new_state=new,
                    metadata=dict(metadata),
                )
            )

    def active_entities(self, entity_type: str | None = None) -> list[Entity]:
        return [
            e for e in self.entities.values() if entity_type is None or e.entity_type == entity_type
        ]

    # -- inspection --------------------------------------------------------

    def snapshot(self, *, max_events: int = 50) -> dict[str, Any]:
        """A JSON-serialisable view of the current state (for debugging and UIs)."""
        return {
            "name": self.name,
            "now": self.clock.now,
            "seed": self.seed,
            "events_processed": self.events_processed,
            "pending_event_count": len(self.queue),
            "pending_events": [e.describe() for e in self.pending_events(max_events)],
            "active_entities": [e.describe() for e in self.entities.values()],
        }

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def __repr__(self) -> str:
        return f"Simulation(name={self.name!r}, now={self.now}, pending={len(self.queue)})"
