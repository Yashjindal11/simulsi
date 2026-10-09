"""Continuous quantities that change at a rate between events (tanks, batteries, queues of fluid).

A :class:`Level` holds a value that moves linearly at its current ``rate``
between events: ``value(t) = value(t0) + rate * (t - t0)``, kept inside
``[low, high]``. Processes change the rate (a pump starts, a charger
plugs in) or the value itself (a tanker delivers), and wait for the level
to cross a threshold with :meth:`Level.when`. Crossings are computed
exactly and scheduled as events, so nothing is polled and no step size
is involved.

>>> from simulsi import Simulation
>>> sim = Simulation()
>>> battery = Level(sim, "battery", init=80, rate=-2.0, low=0, high=100)
>>> def watch(sim):
...     t = yield battery.when(20, "down")
...     battery.set_rate(+10.0)  # start charging
...     yield battery.when(100, "up")
>>> _ = sim.process(watch(sim))
>>> _ = sim.run(until=50)
>>> battery.value, round(battery.time_average(), 2)
(100.0, 63.6)
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Literal

from simulsi.processes.process import Signal

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation
    from simulsi.events.event import Event

Direction = Literal["up", "down", "any"]
EPS = 1e-9


class Level:
    """A continuously changing quantity with a piecewise-constant rate."""

    def __init__(
        self,
        sim: Simulation,
        name: str,
        *,
        init: float = 0.0,
        rate: float = 0.0,
        low: float = -math.inf,
        high: float = math.inf,
    ) -> None:
        if not low <= init <= high:
            raise ValueError(f"init {init} must be within [{low}, {high}]")
        self.sim = sim
        self.name = name
        self.low, self.high = float(low), float(high)
        self._v0 = float(init)
        self._t0 = sim.now
        self._rate = float(rate)
        self._start = sim.now
        self._area = 0.0
        self._min = self._max = float(init)
        self._pending: list[tuple[float, Direction, Signal, Event | None]] = []
        self._bound_event: Event | None = None
        self._schedule_bound()
        sim.on_finish(self._finish)

    # -- state -------------------------------------------------------------------

    @property
    def rate(self) -> float:
        """Current rate, 0 while pinned at a bound it is pushing against."""
        v = self.value
        if (v >= self.high - EPS and self._rate > 0) or (v <= self.low + EPS and self._rate < 0):
            return 0.0
        return self._rate

    @property
    def value(self) -> float:
        return self._at(self.sim.now)

    def _at(self, t: float) -> float:
        v = self._v0 + self._rate * (t - self._t0)
        return min(self.high, max(self.low, v))

    def _advance(self) -> None:
        """Fold the path so far into the running statistics and restart from now."""
        now = self.sim.now
        if now > self._t0:
            self._area += self._integral(self._t0, now)
            for x in (self._at(self._t0), self._at(now)):
                self._min, self._max = min(self._min, x), max(self._max, x)
        self._v0, self._t0 = self._at(now), now

    def _integral(self, a: float, b: float) -> float:
        """Exact area under the clamped linear path between ``a`` and ``b``."""
        r, v0 = self._rate, self._at(a)
        if r == 0:
            return v0 * (b - a)
        bound = self.high if r > 0 else self.low
        t_hit = a + (bound - v0) / r if math.isfinite(bound) else math.inf
        if t_hit >= b:
            return (v0 + self._at(b)) / 2 * (b - a)
        return (v0 + bound) / 2 * (t_hit - a) + bound * (b - t_hit)

    # -- changes -------------------------------------------------------------------

    def set_rate(self, rate: float) -> None:
        """Change how fast the level moves (units per time unit; negative drains)."""
        self._advance()
        self._rate = float(rate)
        self._reschedule()

    def add(self, amount: float) -> float:
        """Jump by ``amount`` (clamped to the bounds); returns the amount actually added."""
        self._advance()
        before = self._v0
        self._v0 = min(self.high, max(self.low, before + amount))
        self._min, self._max = min(self._min, self._v0), max(self._max, self._v0)
        self._reschedule()
        return self._v0 - before

    def when(self, threshold: float, direction: Direction = "any") -> Signal:
        """A signal that triggers (with the time) when the level reaches ``threshold``.

        ``"down"`` waits for the level to be at or below the threshold,
        ``"up"`` at or above, ``"any"`` for it to touch the value from either
        side. Triggers immediately if already there.
        """
        if direction not in ("up", "down", "any"):
            raise ValueError("direction must be 'up', 'down' or 'any'")
        sig = Signal(self.sim, f"{self.name}.when({threshold:g})")
        if self._reached(threshold, direction, self.value):
            sig.succeed(self.sim.now)
            return sig
        entry: tuple[float, Direction, Signal, Event | None] = (threshold, direction, sig, None)
        self._pending.append(entry)
        self._reschedule()
        return sig

    @staticmethod
    def _reached(threshold: float, direction: Direction, v: float) -> bool:
        if direction == "down":
            return v <= threshold + EPS
        if direction == "up":
            return v >= threshold - EPS
        return abs(v - threshold) <= EPS

    def _crossing_time(self, threshold: float, direction: Direction) -> float:
        r, v = self._rate, self.value
        if r == 0 or not (self.low - EPS <= threshold <= self.high + EPS):
            return math.inf
        if (direction == "down" and r > 0) or (direction == "up" and r < 0):
            return math.inf
        dt = (threshold - v) / r
        return self.sim.now + dt if dt >= 0 else math.inf

    def _reschedule(self) -> None:
        sim = self.sim
        kept: list[tuple[float, Direction, Signal, Event | None]] = []
        for threshold, direction, sig, event in self._pending:
            if event is not None:
                sim.cancel(event)
            if self._reached(threshold, direction, self.value):
                sig.succeed(sim.now)
                continue
            t = self._crossing_time(threshold, direction)
            new_event = None
            if math.isfinite(t):
                new_event = sim._schedule_internal(
                    functools.partial(self._fire, threshold, direction), "_level", delay=t - sim.now
                )
            kept.append((threshold, direction, sig, new_event))
        self._pending = kept
        self._schedule_bound()

    def _fire(self, threshold: float, direction: Direction) -> None:
        for i, (th, d, sig, _event) in enumerate(self._pending):
            if th == threshold and d == direction:
                del self._pending[i]
                sig.succeed(self.sim.now)
                return

    def _schedule_bound(self) -> None:
        """Record the moment the level pins at a bound (it then stays there until the rate turns)."""
        if self._bound_event is not None:
            self.sim.cancel(self._bound_event)
            self._bound_event = None
        r, v = self._rate, self.value
        bound = self.high if r > 0 else self.low if r < 0 else math.nan
        if math.isfinite(bound) and abs(v - bound) > EPS:
            self._bound_event = self.sim._schedule_internal(
                self._advance, "_level_bound", delay=(bound - v) / r
            )

    # -- statistics ------------------------------------------------------------------

    def time_average(self) -> float:
        now = self.sim.now
        span = now - self._start
        if span <= 0:
            return self.value
        return (self._area + self._integral(self._t0, now)) / span

    def _finish(self, sim: Simulation) -> None:
        self._advance()
        prefix = f"level.{self.name}"
        sim.metrics.set(f"{prefix}.mean", self.time_average())
        sim.metrics.set(f"{prefix}.final", self.value)
        sim.metrics.set(f"{prefix}.min", self._min)
        sim.metrics.set(f"{prefix}.max", self._max)

    def __repr__(self) -> str:
        return f"Level({self.name!r}, value={self.value:g}, rate={self.rate:g})"
