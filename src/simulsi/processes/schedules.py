"""Time-varying inputs: shift (capacity) schedules and non-stationary arrivals.

* :func:`capacity_schedule` - set a resource's capacity from a table such as
  ``[(0, 2), (480, 4), (960, 1)]``, optionally repeating every ``period``.
* :class:`PiecewiseRate` - an arrival rate that is constant within each
  interval of a table (e.g. hourly call volumes), optionally periodic.
* :func:`arrivals` - a source process for a Poisson process with a constant,
  piecewise-constant or arbitrary time-varying rate (NHPP).

Times in tables are measured from when the schedule or source is created
(normally ``t = 0``).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Generator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation
    from simulsi.processes.process import Process
    from simulsi.resources.resource import Resource


def _table(
    rows: Sequence[tuple[float, float]], period: float | None, what: str
) -> tuple[list[float], list[float]]:
    pts = sorted((float(t), float(v)) for t, v in rows)
    if not pts:
        raise ValueError(f"{what} table is empty")
    if pts[0][0] != 0.0:
        raise ValueError(f"{what} table must start at time 0, got {pts[0][0]}")
    times = [t for t, _ in pts]
    if len(set(times)) != len(times):
        raise ValueError(f"{what} table has duplicate times")
    if period is not None:
        if not (period > 0) or math.isinf(period):
            raise ValueError(f"period must be positive and finite, got {period!r}")
        if times[-1] >= period:
            raise ValueError(f"{what} table times must be < period ({period})")
    return times, [v for _, v in pts]


def _segments(
    times: list[float], period: float | None, start: float
) -> Generator[tuple[int, float, float], None, None]:
    """Yield ``(index, begin, end)`` for every table segment from offset ``start`` on."""
    n = len(times)
    if period is None:
        i = bisect_right(times, start) - 1
        while i < n:
            yield i, times[i], times[i + 1] if i + 1 < n else math.inf
            i += 1
        return
    k = math.floor(start / period)
    i = bisect_right(times, start - k * period) - 1
    while True:
        base = k * period
        end = base + (times[i + 1] if i + 1 < n else period)
        yield i, base + times[i], end
        i += 1
        if i == n:
            i, k = 0, k + 1


def capacity_schedule(
    sim: Simulation,
    resource: Resource,
    table: Sequence[tuple[float, int]],
    *,
    period: float | None = None,
    name: str | None = None,
) -> Process:
    """Change ``resource`` capacity at the times in ``table`` (``(time, capacity)`` rows).

    With ``period`` the table repeats (e.g. ``period=1440`` for a daily shift
    pattern in minutes). Lowering capacity never interrupts work in progress:
    units in use finish their job and the new limit applies to later grants.
    """
    times, caps = _table([(t, float(c)) for t, c in table], period, "capacity")
    for c in caps:
        if c != int(c) or c < 0:
            raise ValueError(f"capacities must be non-negative integers, got {c}")
    if resource.sim is not sim:
        sim.add_resource(resource)
    origin = sim.now

    def run() -> Any:
        for i, begin, _ in _segments(times, period, 0.0):
            at = origin + begin
            if at > sim.now:
                yield at - sim.now
            resource.set_capacity(int(caps[i]))
            if period is None and i == len(times) - 1:
                return

    return sim.process(run(), name=name or f"schedule:{resource.name}")


class PiecewiseRate:
    """A rate that is constant on each interval of ``table`` (``(start_time, rate)`` rows).

    >>> rate = PiecewiseRate([(0, 2.0), (8, 6.0), (17, 1.0)], period=24)
    >>> rate(9.5), rate(30.0), rate.max
    (6.0, 2.0, 6.0)
    """

    def __init__(self, table: Sequence[tuple[float, float]], *, period: float | None = None):
        self.times, self.rates = _table(table, period, "rate")
        if any(r < 0 or math.isinf(r) or math.isnan(r) for r in self.rates):
            raise ValueError("rates must be finite and >= 0")
        self.period = period

    def __call__(self, t: float) -> float:
        if self.period is not None:
            t = t - math.floor(t / self.period) * self.period
        if t < 0:
            return 0.0
        return self.rates[bisect_right(self.times, t) - 1]

    @property
    def max(self) -> float:
        return max(self.rates)

    def mean(self) -> float:
        """Average rate over one period (or the first ``times[-1]`` units without a period)."""
        horizon = self.period if self.period is not None else self.times[-1]
        if horizon == 0:
            return self.rates[0]
        bounds = [*self.times, horizon]
        area = sum(r * (bounds[i + 1] - bounds[i]) for i, r in enumerate(self.rates))
        return area / horizon

    def next_arrival(self, t: float, work: float) -> float:
        """The time at which the integrated rate from ``t`` reaches ``work`` (inf if never)."""
        if self.max == 0:
            return math.inf
        for i, _, end in _segments(self.times, self.period, max(t, 0.0)):
            r = self.rates[i]
            if r > 0:
                need = work / r
                if t + need <= end:
                    return t + need
                work -= r * (end - t)
            elif math.isinf(end):
                return math.inf
            t = end
        return math.inf  # pragma: no cover - the generator above never ends when periodic

    def __repr__(self) -> str:
        rows = list(zip(self.times, self.rates, strict=True))
        return f"PiecewiseRate({rows}, period={self.period})"


RateLike = float | PiecewiseRate | Callable[[float], float]


def arrivals(
    sim: Simulation,
    rate: RateLike,
    spawn: Callable[[Simulation, int], Any],
    *,
    stream: str = "arrivals",
    limit: int | None = None,
    until: float | None = None,
    max_rate: float | None = None,
    name: str = "arrivals",
) -> Process:
    """Start a source that calls ``spawn(sim, i)`` at each arrival of a Poisson process.

    ``rate`` may be a number (homogeneous Poisson process), a
    :class:`PiecewiseRate` (exact, one draw per arrival) or any function of
    time ``rate(t)`` with an upper bound ``max_rate`` (generated by thinning).
    If ``spawn`` returns a generator it is started as a process. Stops after
    ``limit`` arrivals or at time ``until`` (measured like the rate table,
    from now).
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    origin = sim.now
    rs = sim.stream(stream)
    if isinstance(rate, PiecewiseRate):
        table = rate

        def next_time(t: float) -> float:
            return table.next_arrival(t, rs.exponential(1.0))

    elif callable(rate):
        if max_rate is None or not (max_rate > 0) or math.isinf(max_rate):
            raise ValueError("a rate function needs a finite positive max_rate for thinning")
        fn, bound = rate, float(max_rate)

        def next_time(t: float) -> float:
            while True:
                t += rs.exponential(1.0 / bound)
                if until is not None and t > until:
                    return math.inf
                r = fn(t)
                if r > bound * (1 + 1e-9):
                    raise ValueError(f"rate({t}) = {r} exceeds max_rate = {bound}")
                if rs.random() * bound < r:
                    return t

    else:
        lam = float(rate)
        if not (lam > 0) or math.isinf(lam):
            raise ValueError(f"rate must be positive and finite, got {rate!r}")

        def next_time(t: float) -> float:
            return t + rs.exponential(1.0 / lam)

    def run() -> Any:
        count = 0
        t = 0.0
        while limit is None or count < limit:
            t = next_time(t)
            if math.isinf(t) or (until is not None and t > until):
                return
            yield origin + t - sim.now
            out = spawn(sim, count)
            if isinstance(out, Generator):
                sim.process(out)
            count += 1

    return sim.process(run(), name=name)
