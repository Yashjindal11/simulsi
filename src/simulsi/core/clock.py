"""Simulation clock.

Internally simulation time is always a ``float`` measured in abstract *time
units*. A clock may optionally be anchored to the calendar via ``epoch`` (the
``datetime`` that corresponds to ``t = 0``) and ``unit`` (the ``timedelta``
that one time unit represents). This keeps the hot path cheap while letting
users schedule with ``timedelta`` / ``datetime`` values when that is natural.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from simulsi.errors import SchedulingError

TimeLike = int | float | timedelta | datetime
DurationLike = int | float | timedelta


class Clock:
    """Holds the current simulation time and converts user-facing time values."""

    __slots__ = ("_now", "epoch", "start", "unit")

    def __init__(
        self,
        start: float = 0.0,
        *,
        epoch: datetime | None = None,
        unit: timedelta = timedelta(minutes=1),
    ) -> None:
        if unit <= timedelta(0):
            raise ValueError("clock unit must be a positive timedelta")
        start_f = float(start)
        if not math.isfinite(start_f):
            raise ValueError("clock start must be finite")
        self.start = start_f
        self.epoch = epoch
        self.unit = unit
        self._now = start_f

    @property
    def now(self) -> float:
        return self._now

    def advance_to(self, time: float) -> None:
        if time < self._now:
            raise SchedulingError(f"clock cannot move backwards: {time} < {self._now}")
        self._now = time

    def reset(self) -> None:
        self._now = self.start

    # -- conversions -------------------------------------------------------

    def duration(self, value: DurationLike) -> float:
        """Convert a duration (number of time units or ``timedelta``) to float time units."""
        if isinstance(value, timedelta):
            return value / self.unit
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"expected a number or timedelta, got {type(value).__name__}")
        return float(value)

    def to_time(self, value: TimeLike) -> float:
        """Convert an absolute time to float time units.

        Numbers are taken as-is, ``timedelta`` values are offsets from ``t = 0``
        and ``datetime`` values require an ``epoch``.
        """
        if isinstance(value, datetime):
            if self.epoch is None:
                raise SchedulingError("datetime values need a clock epoch (Simulation(epoch=...))")
            return (value - self.epoch) / self.unit
        return self.duration(value)

    def to_timedelta(self, time: float) -> timedelta:
        return self.unit * time

    def to_datetime(self, time: float | None = None) -> datetime:
        """Calendar time for ``time`` (default: now). Requires an epoch."""
        if self.epoch is None:
            raise ValueError("clock has no epoch; pass epoch=datetime(...) to the Simulation")
        return self.epoch + self.unit * (self._now if time is None else time)

    def __repr__(self) -> str:
        return f"Clock(now={self._now!r})"
