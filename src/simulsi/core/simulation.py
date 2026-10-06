"""The simulation engine."""

from __future__ import annotations

import math
import time as _wall
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from heapq import heappush
from pathlib import Path
from typing import Any

from simulsi.core.clock import Clock, DurationLike, TimeLike
from simulsi.core.trace import EventLog, LogRecord
from simulsi.entities.entity import Entity
from simulsi.errors import EventStateError, ResourceUsageError, SchedulingError
from simulsi.events.event import Event, EventCallback, EventQueue, EventStatus, Priority, _CallEvent
from simulsi.metrics.collectors import Metrics
from simulsi.processes.process import (
    AllOf,
    AnyOf,
    Process,
    ProcessGenerator,
    Signal,
    Timeout,
    Waitable,
)
from simulsi.queues.discipline import Discipline
from simulsi.queues.queue import Queue
from simulsi.randomness.stream import RandomStream, SamplingMode
from simulsi.resources.resource import Request, Resource

_SCHEDULED = EventStatus.SCHEDULED


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
    series: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

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
    """A discrete-event simulation: clock, event queue, randomness, components and metrics.

    >>> sim = Simulation(seed=42)
    >>> server = sim.resource("server", capacity=1)
    >>> def customer(sim):
    ...     yield sim.request(server)
    ...     yield 3.0
    ...     sim.release(server)
    >>> _ = sim.process(customer(sim))
    >>> sim.run(until=10).metrics["resource.server.utilization"]
    0.3
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
        keep_values: bool = True,
        record_series: bool = True,
        sampling: SamplingMode = "native",
    ) -> None:
        self.name = name
        self.clock = Clock(start, epoch=epoch, unit=time_unit)
        self.rng = RandomStream(seed, mode=sampling)
        self.seed = self.rng.seed
        self.event_queue = EventQueue()
        self.log: EventLog | None = EventLog(max_log_records) if trace else None
        self.keep_entity_history = keep_entity_history
        self.metrics = Metrics(self.clock, keep_values=keep_values, record_series=record_series)
        self.events_processed = 0
        self.warnings: list[str] = []
        self.entities: dict[str, Entity] = {}
        self.resources: dict[str, Resource] = {}
        self.queues: dict[str, Queue[Any]] = {}
        self._id_counters: dict[str, int] = {}
        self._created: dict[str, int] = {}
        self._disposed: dict[str, int] = {}
        self._processes: dict[Process, None] = {}
        self._processes_started = 0
        self._active_process: Process | None = None
        self._next_event_id = 0
        self._stop_requested = False
        self._finish_hooks: list[Callable[[Simulation], None]] = []
        self._wall_time = 0.0
        self._stats_start = self.clock.now
        self._origin: tuple[Any, Any, dict[str, Any]] | None = None

    # -- time --------------------------------------------------------------

    @property
    def now(self) -> float:
        return self.clock._now

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
        self.event_queue.push(event, t)
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
        """Schedule a plain function call. Lighter than a full :class:`Event` and not logged."""
        return self.schedule(_CallEvent(fn, label, priority), time=time, delay=delay)

    def _schedule_internal(
        self,
        fn: Callable[[], Any],
        label: str,
        *,
        delay: float = 0.0,
        priority: int = Priority.NORMAL,
    ) -> Event:
        ev = _CallEvent(fn, label, priority)
        q = self.event_queue
        seq = q._seq
        q._seq = seq + 1
        q._live += 1
        t = self.clock._now + delay
        ev._seq = seq
        ev.timestamp = t
        ev.status = _SCHEDULED
        heappush(q._heap, (t, priority, seq, ev))
        return ev

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
        if event.status is EventStatus.CREATED:
            raise EventStateError(f"{event!r} was never scheduled")
        if priority is not None:
            event.priority = int(priority)
        self.event_queue.push(event, self._resolve_time(time, delay))
        return event

    def cancel(self, event: Event) -> bool:
        """Cancel a pending event. Returns ``False`` if it was not pending."""
        if event.status is not EventStatus.SCHEDULED:
            return False
        self.event_queue.discard(event)
        return True

    def pending_events(
        self, limit: int | None = None, *, include_internal: bool = False
    ) -> list[Event]:
        events = [e for e in self.event_queue if include_internal or not e.internal]
        return events if limit is None else events[:limit]

    def peek(self) -> float:
        """Time of the next pending event (``inf`` if none)."""
        return self.event_queue.peek_time()

    # -- execution ---------------------------------------------------------

    def _execute(self, event: Event) -> None:
        self.clock._now = event.timestamp
        event.status = EventStatus.EXECUTED
        self.events_processed += 1
        if self.log is not None and not event.internal:
            self.log.append(
                LogRecord(event.timestamp, event.event_type, metadata=dict(event.payload))
            )
        event.execute(self)

    def step(self) -> Event | None:
        """Execute the next event. Returns it, or ``None`` if the queue is empty."""
        event = self.event_queue.pop()
        if event is not None:
            self._execute(event)
        return event

    def stop(self) -> None:
        """Ask :meth:`run` to return after the current event."""
        self._stop_requested = True

    def _replay(self, n_events: int) -> int:
        """Execute exactly ``n_events`` events, ignoring stop requests (checkpoint restore)."""
        pop = self.event_queue.pop
        done = 0
        while done < n_events:
            event = pop()
            if event is None:
                break
            self._execute(event)
            done += 1
        self._stop_requested = False
        return done

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save a checkpoint that :meth:`load_checkpoint` restores by deterministic replay.

        See :mod:`simulsi.core.checkpoint` for what is stored and the requirements.
        """
        from simulsi.core.checkpoint import save_checkpoint

        return save_checkpoint(self, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, model: Any = None) -> Simulation:
        """Rebuild a simulation saved with :meth:`save_checkpoint`, in exactly the saved state."""
        from simulsi.core.checkpoint import load_checkpoint

        return load_checkpoint(path, model=model)

    def run(
        self,
        until: TimeLike | None = None,
        *,
        max_events: int | None = None,
    ) -> SimulationResult:
        """Process events in order and return a :class:`SimulationResult`.

        With ``until``, every event whose timestamp is ``<= until`` runs and the
        clock then stands exactly at ``until`` (even if the queue emptied
        earlier). Without it, the run lasts until no events remain. Runs can
        be resumed by calling :meth:`run` again with a later ``until``.
        """
        horizon = math.inf
        if until is not None:
            horizon = self.clock.to_time(until)
            if horizon < self.clock.now:
                raise SchedulingError(f"until={horizon} is before now={self.clock.now}")
        start = self.clock.now
        wall0 = _wall.perf_counter()
        self._stop_requested = False
        limit = math.inf if max_events is None else max_events
        executed = 0
        pop_due = self.event_queue.pop_due
        clock = self.clock
        executed_status = EventStatus.EXECUTED
        # Same steps as _execute, inlined: this loop runs once per event.
        while executed < limit and not self._stop_requested:
            event = pop_due(horizon)
            if event is None:
                break
            clock._now = event.timestamp
            event.status = executed_status
            self.events_processed += 1
            if event.internal:
                event.execute(self)
            else:
                if self.log is not None:
                    self.log.append(
                        LogRecord(event.timestamp, event.event_type, metadata=dict(event.payload))
                    )
                event.execute(self)
            executed += 1
        stopped_early = self._stop_requested or executed >= limit
        if not stopped_early and math.isfinite(horizon):
            self.clock.advance_to(horizon)
        self._wall_time += _wall.perf_counter() - wall0
        self._check_blocked()
        return self._finish(start)

    def _check_blocked(self) -> None:
        if self.event_queue:
            return
        blocked = [p for p in self._processes if p.target is not None]
        if blocked:
            names = ", ".join(sorted({p.name for p in blocked})[:5])
            self.warn(
                f"{len(blocked)} process(es) blocked with no pending events "
                f"(possible deadlock or starvation): {names}"
            )

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
            metrics=self.flat_metrics(),
            details=self.details(),
            warnings=list(self.warnings),
            log=self.log,
            series=self.series(),
        )

    def series(self) -> dict[str, list[tuple[float, float]]]:
        """Recorded step-function time series (empty when ``record_series=False``)."""
        out: dict[str, list[tuple[float, float]]] = {}
        if not self.metrics.record_series:
            return out
        for name, r in self.resources.items():
            for key, tw in (
                ("busy", r.busy),
                ("queue_length", r.queue_length),
                ("capacity", r.effective_level),
            ):
                if tw.series is not None:
                    out[f"resource.{name}.{key}"] = list(tw.series)
        for name, q in self.queues.items():
            if q.length.series is not None:
                out[f"queue.{name}.length"] = list(q.length.series)
        for name, g in self.metrics.gauges.items():
            if g.series is not None:
                out[name] = list(g.series)
        return out

    def flat_metrics(self) -> dict[str, float]:
        flat: dict[str, float] = {
            "sim.events": float(self.events_processed),
            "sim.observed_time": self.clock.now - self._stats_start,
        }
        for etype in sorted(set(self._created) | set(self._disposed)):
            flat[f"entity.{etype}.created"] = float(self._created.get(etype, 0))
            flat[f"entity.{etype}.disposed"] = float(self._disposed.get(etype, 0))
        flat.update(self.metrics.flat())
        for r in self.resources.values():
            flat.update(r.flat_metrics())
        for q in self.queues.values():
            flat.update(q.flat_metrics())
        return flat

    def details(self) -> dict[str, Any]:
        return {
            "entities": {
                etype: {
                    "created": self._created.get(etype, 0),
                    "disposed": self._disposed.get(etype, 0),
                }
                for etype in sorted(set(self._created) | set(self._disposed))
            },
            "metrics": self.metrics.summary(),
            "resources": {name: r.summary() for name, r in self.resources.items()},
            "queues": {name: q.summary() for name, q in self.queues.items()},
        }

    def reset_statistics(self) -> None:
        """Discard statistics collected so far, typically at the end of a warm-up period."""
        self.metrics.reset()
        for r in self.resources.values():
            r.reset_stats()
        for q in self.queues.values():
            q.reset_stats()
        self._created.clear()
        self._disposed.clear()
        self._stats_start = self.clock.now

    def warmup(self, duration: DurationLike) -> None:
        """Reset statistics once ``duration`` has elapsed (schedules :meth:`reset_statistics`)."""
        self._schedule_internal(
            self.reset_statistics,
            "_warmup",
            delay=self.clock.duration(duration),
            priority=Priority.URGENT,
        )

    # -- processes ---------------------------------------------------------

    def process(
        self, generator: ProcessGenerator, *, name: str | None = None, entity: Entity | None = None
    ) -> Process:
        """Start a generator as a process at the current time."""
        p = Process(self, generator, name=name, entity=entity)
        self._processes[p] = None
        self._processes_started += 1
        return p

    def _process_finished(self, process: Process) -> None:
        self._processes.pop(process, None)

    @property
    def active_process(self) -> Process | None:
        return self._active_process

    def timeout(self, delay: DurationLike, value: Any = None) -> Timeout:
        return Timeout(self, self.clock.duration(delay), value)

    def signal(self, name: str = "signal") -> Signal:
        return Signal(self, name)

    def all_of(self, *waitables: Waitable | Iterable[Waitable]) -> AllOf:
        return AllOf(self, _flatten(waitables))

    def any_of(self, *waitables: Waitable | Iterable[Waitable]) -> AnyOf:
        return AnyOf(self, _flatten(waitables))

    # -- resources and queues ----------------------------------------------

    def add_resource(self, resource: Resource) -> Resource:
        existing = self.resources.get(resource.name)
        if existing is not None and existing is not resource:
            raise ResourceUsageError(f"a different resource named {resource.name!r} already exists")
        resource._bind(self)
        self.resources[resource.name] = resource
        return resource

    def resource(
        self,
        name: str,
        capacity: int = 1,
        *,
        discipline: Discipline = "fifo",
        preemptive: bool = False,
    ) -> Resource:
        return self.add_resource(
            Resource(name, capacity, discipline=discipline, preemptive=preemptive)
        )

    def add_queue(self, queue: Queue[Any]) -> Queue[Any]:
        existing = self.queues.get(queue.name)
        if existing is not None and existing is not queue:
            raise ResourceUsageError(f"a different queue named {queue.name!r} already exists")
        queue._bind(self)
        self.queues[queue.name] = queue
        return queue

    def queue(
        self, name: str, *, discipline: Discipline = "fifo", capacity: float = math.inf
    ) -> Queue[Any]:
        return self.add_queue(Queue(name, discipline=discipline, capacity=capacity))

    def request(
        self,
        resource: Resource,
        *,
        priority: float = 0,
        entity: Entity | None = None,
        patience: DurationLike | None = None,
    ) -> Request:
        """Ask for one unit of ``resource``; yield the result to wait for it."""
        if resource.sim is not self:
            self.add_resource(resource)
        return resource.request(
            priority=priority,
            entity=entity,
            patience=None if patience is None else self.clock.duration(patience),
        )

    def release(self, target: Resource | Request) -> None:
        """Release a request, or the most recent unit of ``resource`` held by the current process."""
        if isinstance(target, Request):
            target.resource.release(target)
            return
        proc = self._active_process
        if proc is not None:
            for req in reversed(proc.held):
                if req.resource is target:
                    target.release(req)
                    return
        raise ResourceUsageError(
            f"the current process holds no unit of {target.name!r}; pass the Request to release()"
        )

    def use(
        self, resource: Resource, duration: DurationLike, *, priority: float = 0
    ) -> Generator[Any, Any, Request]:
        """Acquire ``resource``, hold it for ``duration``, release it: ``yield from sim.use(...)``.

        The unit is released even if the process is interrupted while holding it.
        """
        req: Request = yield self.request(resource, priority=priority)
        try:
            yield self.timeout(duration)
        except GeneratorExit:
            raise
        except BaseException:
            if req.released_at is None:
                resource.release(req)
            raise
        resource.release(req)
        return req

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
        etype = entity.entity_type
        n = self._id_counters.get(etype, 0) + 1
        self._id_counters[etype] = n
        self._created[etype] = self._created.get(etype, 0) + 1
        if entity._auto_id:
            entity.id = f"{etype}-{n}"
            entity._auto_id = False
        entity.created_at = self.clock.now
        entity._sim = self
        self.entities[entity.id] = entity
        if self.log is not None:
            self.log.append(
                LogRecord(
                    self.clock.now,
                    "entity.created",
                    entity=entity.id,
                    new_state=entity.state,
                    metadata={"entity_type": etype},
                )
            )
        return entity

    def dispose(self, entity: Entity, state: str = "disposed") -> None:
        """Mark an entity as having left the system; records its time in system."""
        if entity.disposed_at is not None:
            raise EventStateError(f"{entity!r} was already disposed")
        if entity._sim is not self:
            raise EventStateError(f"{entity!r} is not registered with this simulation")
        if entity.state != state:
            entity.set_state(state)
        entity.disposed_at = self.clock.now
        self.entities.pop(entity.id, None)
        etype = entity.entity_type
        self._disposed[etype] = self._disposed.get(etype, 0) + 1
        self.metrics.observe(
            f"entity.{etype}.time_in_system", entity.disposed_at - entity.created_at
        )

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

    def snapshot(self, *, max_events: int = 50, max_entities: int = 500) -> dict[str, Any]:
        """A JSON-serialisable view of the current state (for debugging and UIs)."""
        return {
            "name": self.name,
            "now": self.clock.now,
            "seed": self.seed,
            "events_processed": self.events_processed,
            "pending_event_count": len(self.event_queue),
            "pending_events": [e.describe() for e in self.pending_events(max_events)],
            "active_entity_count": len(self.entities),
            "active_entities": [e.describe() for e in list(self.entities.values())[:max_entities]],
            "active_processes": len(self._processes),
            "resources": {name: r.describe() for name, r in self.resources.items()},
            "queues": {name: q.describe() for name, q in self.queues.items()},
            "metrics": self.flat_metrics(),
            "warnings": list(self.warnings),
        }

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def __repr__(self) -> str:
        return f"Simulation(name={self.name!r}, now={self.now}, pending={len(self.event_queue)})"


def _flatten(items: tuple[Waitable | Iterable[Waitable], ...]) -> list[Waitable]:
    out: list[Waitable] = []
    for item in items:
        if isinstance(item, Waitable):
            out.append(item)
        else:
            out.extend(item)
    return out
