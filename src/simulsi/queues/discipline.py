"""Queue disciplines shared by resources and queues."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")

DisciplineName = Literal["fifo", "lifo", "priority"]
Discipline = DisciplineName | Callable[[Any], Any]

_DISCIPLINES = ("fifo", "lifo", "priority")


def check_discipline(discipline: Discipline) -> None:
    if not callable(discipline) and discipline not in _DISCIPLINES:
        raise ValueError(
            f"unknown queue discipline {discipline!r}; use one of {_DISCIPLINES} or a key function"
        )


class OrderedBuffer(Generic[T]):
    """Items ordered by a discipline; ties always fall back to arrival order.

    * ``"fifo"``     - first in, first out
    * ``"lifo"``     - last in, first out
    * ``"priority"`` - lowest ``priority`` value first, FIFO among equals
    * a callable     - ``key(subject)`` ascending, FIFO among equals

    Removal from the middle (reneging, cancellation) is O(1) via lazy deletion.
    """

    __slots__ = ("_entries", "_heap", "_seq", "discipline")

    def __init__(self, discipline: Discipline = "fifo") -> None:
        check_discipline(discipline)
        self.discipline = discipline
        self._heap: list[list[Any]] = []
        self._entries: dict[int, list[Any]] = {}
        self._seq = 0

    def _key(self, priority: float, subject: Any) -> Any:
        d = self.discipline
        if d == "fifo":
            return 0
        if d == "lifo":
            return -self._seq
        if d == "priority":
            return priority
        assert callable(d)
        return d(subject)

    def push(self, item: T, priority: float = 0, subject: Any = None) -> None:
        key = self._key(priority, item if subject is None else subject)
        entry = [key, self._seq, item, True]
        self._seq += 1
        self._entries[id(item)] = entry
        heapq.heappush(self._heap, entry)

    def _clean(self) -> None:
        heap = self._heap
        while heap and not heap[0][3]:
            heapq.heappop(heap)

    def peek(self) -> T | None:
        self._clean()
        return self._heap[0][2] if self._heap else None

    def pop(self) -> T:
        self._clean()
        if not self._heap:
            raise IndexError("pop from empty buffer")
        entry = heapq.heappop(self._heap)
        del self._entries[id(entry[2])]
        item: T = entry[2]
        return item

    def remove(self, item: T) -> bool:
        entry = self._entries.pop(id(item), None)
        if entry is None:
            return False
        entry[3] = False
        return True

    def __contains__(self, item: object) -> bool:
        return id(item) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __iter__(self) -> Iterator[T]:
        """Items in service order (sorts a copy)."""
        live = sorted(e for e in self._heap if e[3])
        return (e[2] for e in live)
