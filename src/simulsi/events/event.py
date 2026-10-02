"""Events and the event queue."""

from __future__ import annotations

import enum
import heapq
import math
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation


class EventStatus(enum.Enum):
    CREATED = "created"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    EXECUTED = "executed"


class Priority(enum.IntEnum):
    """Conventional priorities. Lower values run first among same-time events."""

    URGENT = -100
    HIGH = -10
    NORMAL = 0
    LOW = 10


EventCallback = Callable[["Simulation", "Event"], Any]

_EMPTY: dict[str, Any] = {}


class Event:
    """Something that happens at a point in simulation time.

    Use it directly with a ``callback`` or subclass it and override
    :meth:`execute`. ``event_id`` and ``timestamp`` are assigned when the event
    is scheduled.

    Ordering is fully deterministic: events execute by ``(timestamp,
    priority, scheduling order)``; lower priority values run first.
    """

    __slots__ = (
        "__dict__",
        "_seq",
        "callback",
        "event_id",
        "event_type",
        "internal",
        "payload",
        "priority",
        "status",
        "timestamp",
    )

    def __init__(
        self,
        event_type: str = "event",
        payload: dict[str, Any] | None = None,
        callback: EventCallback | None = None,
        priority: int = Priority.NORMAL,
    ) -> None:
        self.event_type = event_type
        self.payload: dict[str, Any] = {} if payload is None else payload
        self.callback = callback
        self.priority = int(priority)
        self.event_id = -1
        self.timestamp = math.nan
        self.status = EventStatus.CREATED
        self.internal = False
        self._seq = -1

    def execute(self, sim: Simulation) -> None:
        """Run the event's behaviour. Subclasses override this."""
        if self.callback is not None:
            self.callback(sim, self)

    @property
    def pending(self) -> bool:
        return self.status is EventStatus.SCHEDULED

    def describe(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "event_type": self.event_type,
            "status": self.status.value,
            "payload": dict(self.payload),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.event_id}, type={self.event_type!r}, "
            f"t={self.timestamp}, priority={self.priority}, status={self.status.value})"
        )


class _CallEvent(Event):
    """Internal event that invokes a zero-argument function. Not logged."""

    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[], Any], event_type: str, priority: int) -> None:
        # Hot path: assign slots directly instead of calling Event.__init__.
        self.fn = fn
        self.event_type = event_type
        self.priority = priority
        self.internal = True
        self.event_id = -1
        self.status = EventStatus.CREATED
        self.callback = None
        self.payload = _EMPTY
        self.timestamp = math.nan
        self._seq = -1

    def execute(self, sim: Simulation) -> None:
        self.fn()


class EventQueue:
    """Binary-heap event queue with lazy cancellation.

    Heap entries are ``(time, priority, seq, event)``. ``seq`` is unique and
    monotonically increasing, so ties are broken by scheduling order and
    events themselves are never compared. Cancelling or rescheduling marks the
    old entry stale instead of searching the heap (O(1)); stale entries are
    discarded when they reach the top.
    """

    __slots__ = ("_heap", "_live", "_seq")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, Event]] = []
        self._seq = 0
        self._live = 0

    def push(self, event: Event, time: float) -> None:
        seq = self._seq
        self._seq = seq + 1
        if event.status is not EventStatus.SCHEDULED:
            self._live += 1
        event._seq = seq
        event.timestamp = time
        event.status = EventStatus.SCHEDULED
        heapq.heappush(self._heap, (time, event.priority, seq, event))

    def discard(self, event: Event) -> None:
        """Mark a scheduled event as cancelled; its heap entry becomes stale."""
        if event.status is EventStatus.SCHEDULED:
            event.status = EventStatus.CANCELLED
            self._live -= 1

    def _drop_stale(self) -> None:
        heap = self._heap
        while heap:
            _, _, seq, ev = heap[0]
            if ev.status is EventStatus.SCHEDULED and ev._seq == seq:
                return
            heapq.heappop(heap)

    def pop(self) -> Event | None:
        heap = self._heap
        while heap:
            _, _, seq, ev = heapq.heappop(heap)
            if ev.status is EventStatus.SCHEDULED and ev._seq == seq:
                self._live -= 1
                return ev
        return None

    def pop_due(self, horizon: float) -> Event | None:
        """Pop the next live event if its time is ``<= horizon``, else return ``None``."""
        heap = self._heap
        while heap:
            entry = heap[0]
            ev = entry[3]
            if ev.status is not EventStatus.SCHEDULED or ev._seq != entry[2]:
                heapq.heappop(heap)
                continue
            if entry[0] > horizon:
                return None
            heapq.heappop(heap)
            self._live -= 1
            return ev
        return None

    def peek_time(self) -> float:
        """Time of the next live event, or ``inf`` when empty."""
        self._drop_stale()
        return self._heap[0][0] if self._heap else math.inf

    def peek(self) -> Event | None:
        self._drop_stale()
        return self._heap[0][3] if self._heap else None

    def __len__(self) -> int:
        return self._live

    def __bool__(self) -> bool:
        return self._live > 0

    def __iter__(self) -> Iterator[Event]:
        """Live events in execution order (sorts a copy; intended for inspection)."""
        live = [
            entry
            for entry in self._heap
            if entry[3].status is EventStatus.SCHEDULED and entry[3]._seq == entry[2]
        ]
        live.sort(key=lambda e: (e[0], e[1], e[2]))
        return (entry[3] for entry in live)

    def clear(self) -> None:
        for _, _, _, ev in self._heap:
            if ev.status is EventStatus.SCHEDULED:
                ev.status = EventStatus.CANCELLED
        self._heap.clear()
        self._live = 0
