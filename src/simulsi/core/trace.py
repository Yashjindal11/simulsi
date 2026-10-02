"""Structured simulation trace (opt-in event log)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LogRecord:
    timestamp: float
    event_type: str
    entity: str | None = None
    resource: str | None = None
    old_state: str | None = None
    new_state: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLog:
    """Append-only list of :class:`LogRecord` objects.

    ``max_records`` bounds memory for very long runs; once reached, further
    records are counted in ``dropped`` instead of stored.
    """

    def __init__(self, max_records: int | None = None) -> None:
        self.records: list[LogRecord] = []
        self.max_records = max_records
        self.dropped = 0

    def append(self, record: LogRecord) -> None:
        if self.max_records is not None and len(self.records) >= self.max_records:
            self.dropped += 1
            return
        self.records.append(record)

    def filter(
        self,
        *,
        event_type: str | None = None,
        entity: str | None = None,
        resource: str | None = None,
    ) -> list[LogRecord]:
        return [
            r
            for r in self.records
            if (event_type is None or r.event_type == event_type)
            and (entity is None or r.entity == entity)
            and (resource is None or r.resource == resource)
        ]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[LogRecord]:
        return iter(self.records)
