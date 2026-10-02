"""Queues (buffers) that hold items between process steps."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from simulsi.core.trace import LogRecord
from simulsi.errors import CapacityError, ResourceUsageError
from simulsi.metrics.collectors import Counter, Tally, TimeWeighted
from simulsi.processes.process import Waitable
from simulsi.queues.discipline import Discipline, OrderedBuffer

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation

T = TypeVar("T")


class _Slot(Generic[T]):
    __slots__ = ("item", "priority", "put_at")

    def __init__(self, item: T, priority: float, put_at: float) -> None:
        self.item = item
        self.priority = priority
        self.put_at = put_at


class Put(Waitable):
    __slots__ = ("item", "priority")

    def __init__(self, sim: Simulation, item: Any, priority: float) -> None:
        super().__init__(sim)
        self.item = item
        self.priority = priority


class Get(Waitable):
    __slots__ = ()


class Queue(Generic[T]):
    """An ordered buffer with optional finite capacity and blocking put/get.

    ``yield queue.put(item)`` waits while the queue is full; ``item = yield
    queue.get()`` waits while it is empty. Items leave in the order given by
    ``discipline`` (``"fifo"``, ``"lifo"``, ``"priority"`` or a key function
    applied to the item).

    Accounting invariant (checked in tests): ``puts == gets + removed + len(queue)``,
    so items can never disappear silently.
    """

    def __init__(
        self,
        name: str = "queue",
        *,
        discipline: Discipline = "fifo",
        capacity: float = math.inf,
        sim: Simulation | None = None,
    ) -> None:
        if not (capacity > 0):
            raise CapacityError(f"queue capacity must be > 0, got {capacity!r}")
        self.name = name
        self.capacity = capacity
        self.discipline = discipline
        self._buffer: OrderedBuffer[_Slot[T]] = OrderedBuffer(discipline)
        self._getters: deque[Get] = deque()
        self._putters: deque[Put] = deque()
        self.sim: Simulation | None = None
        if sim is not None:
            sim.add_queue(self)

    def _bind(self, sim: Simulation) -> None:
        if self.sim is not None:
            if self.sim is not sim:
                raise ResourceUsageError(f"queue {self.name!r} belongs to another simulation")
            return
        self.sim = sim
        series = sim.metrics.record_series
        self.length = TimeWeighted(f"{self.name}.length", sim.clock, 0, record_series=series)
        self.wait_times = Tally(f"{self.name}.wait", keep_values=sim.metrics.keep_values)
        self.puts = Counter(f"{self.name}.puts", sim.clock)
        self.gets = Counter(f"{self.name}.gets", sim.clock)
        self.removed = Counter(f"{self.name}.removed", sim.clock)
        self.blocked_puts = Counter(f"{self.name}.blocked_puts", sim.clock)

    def _require_sim(self) -> Simulation:
        if self.sim is None:
            raise ResourceUsageError(f"queue {self.name!r} is not attached; use sim.add_queue()")
        return self.sim

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def items(self) -> list[T]:
        return [s.item for s in self._buffer]

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    # -- operations ------------------------------------------------------------

    def put(self, item: T, *, priority: float = 0) -> Put:
        sim = self._require_sim()
        req = Put(sim, item, priority)
        if self._putters or self.is_full:
            self.blocked_puts.increment()
            self._putters.append(req)
        else:
            self._accept(req)
        return req

    def get(self) -> Get:
        sim = self._require_sim()
        req = Get(sim)
        if self._buffer and not self._getters:
            req.succeed(self._take())
            self._admit_putters()
        else:
            self._getters.append(req)
        return req

    def try_get(self) -> T | None:
        """Non-blocking get: the next item, or ``None`` if empty."""
        self._require_sim()
        if not self._buffer or self._getters:
            return None
        item = self._take()
        self._admit_putters()
        return item

    def remove(self, item: T) -> bool:
        """Remove a specific item (e.g. a customer abandoning the line)."""
        sim = self._require_sim()
        for slot in self._buffer:
            if slot.item is item:
                self._buffer.remove(slot)
                self.removed.increment()
                self.length.record(len(self._buffer))
                if sim.log is not None:
                    sim.log.append(LogRecord(sim.now, "queue.remove", resource=self.name))
                self._admit_putters()
                return True
        return False

    def cancel(self, req: Get | Put) -> bool:
        """Withdraw a pending get or put."""
        pool: deque[Any] = self._getters if isinstance(req, Get) else self._putters
        if req.triggered or req not in pool:
            return False
        pool.remove(req)
        req._value = None
        if isinstance(req, Put):
            self._admit_putters()
        return True

    def _accept(self, req: Put) -> None:
        sim = self._require_sim()
        slot = _Slot(req.item, req.priority, sim.now)
        self.puts.increment()
        if sim.log is not None:
            sim.log.append(
                LogRecord(sim.now, "queue.put", entity=_item_id(req.item), resource=self.name)
            )
        req.succeed(None)
        if self._getters:
            # Hand over directly: the item never sits in the buffer.
            self.wait_times.observe(0.0)
            self.gets.increment()
            if sim.log is not None:
                sim.log.append(
                    LogRecord(sim.now, "queue.get", entity=_item_id(slot.item), resource=self.name)
                )
            self._getters.popleft().succeed(slot.item)
            return
        self._buffer.push(slot, slot.priority, slot.item)
        self.length.record(len(self._buffer))

    def _take(self) -> T:
        sim = self._require_sim()
        slot = self._buffer.pop()
        self.length.record(len(self._buffer))
        self.wait_times.observe(sim.now - slot.put_at)
        self.gets.increment()
        if sim.log is not None:
            sim.log.append(
                LogRecord(sim.now, "queue.get", entity=_item_id(slot.item), resource=self.name)
            )
        return slot.item

    def _admit_putters(self) -> None:
        while self._putters and not self.is_full:
            self._accept(self._putters.popleft())

    # -- statistics --------------------------------------------------------------

    def reset_stats(self) -> None:
        self.length.reset()
        self.wait_times.reset()
        for c in (self.puts, self.gets, self.removed, self.blocked_puts):
            c.reset()

    def summary(self) -> dict[str, Any]:
        if self.sim is None:
            return {"bound": False}
        return {
            "length_now": len(self._buffer),
            "mean_length": self.length.mean,
            "max_length": self.length.max,
            "wait": self.wait_times.summary(),
            "puts": self.puts.value,
            "gets": self.gets.value,
            "removed": self.removed.value,
            "blocked_puts": self.blocked_puts.value,
            "throughput": self.gets.rate,
            "waiting_getters": len(self._getters),
            "waiting_putters": len(self._putters),
        }

    def flat_metrics(self) -> dict[str, float]:
        if self.sim is None:
            return {}
        s = self.summary()
        p = f"queue.{self.name}."
        out = {
            p + k: float(s[k])
            for k in ("mean_length", "max_length", "puts", "gets", "removed", "throughput")
        }
        out[p + "length_final"] = float(s["length_now"])
        out[p + "wait.mean"] = float(s["wait"]["mean"])
        out[p + "wait.max"] = float(s["wait"]["max"])
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "length": len(self._buffer),
            "capacity": None if math.isinf(self.capacity) else self.capacity,
            "discipline": self.discipline if isinstance(self.discipline, str) else "custom",
            "items": [_item_id(s.item) or repr(s.item) for s in list(self._buffer)[:50]],
            "waiting_getters": len(self._getters),
            "waiting_putters": len(self._putters),
        }

    def __repr__(self) -> str:
        return f"Queue({self.name!r}, length={len(self)})"


def _item_id(item: Any) -> str | None:
    from simulsi.entities.entity import Entity

    return item.id if isinstance(item, Entity) else None
