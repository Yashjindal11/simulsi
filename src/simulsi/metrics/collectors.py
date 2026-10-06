"""Statistic collectors and the metrics registry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from simulsi.core.clock import Clock

DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99)


class Tally:
    """Summary statistics for a stream of observations (e.g. waiting times).

    Mean and variance use Welford's numerically stable update. When
    ``keep_values`` is true the raw observations are retained so quantiles
    and histograms can be computed exactly.
    """

    __slots__ = (
        "_m2",
        "_mean",
        "bins",
        "count",
        "keep_values",
        "max",
        "min",
        "name",
        "total",
        "values",
    )

    def __init__(
        self, name: str, *, keep_values: bool = True, bins: int | Sequence[float] | None = None
    ) -> None:
        self.name = name
        self.keep_values = keep_values
        self.bins = bins
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.values: list[float] = []

    def observe(self, value: float) -> None:
        x = float(value)
        self.count += 1
        self.total += x
        delta = x - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (x - self._mean)
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x
        if self.keep_values:
            self.values.append(x)

    @property
    def mean(self) -> float:
        return self._mean if self.count else math.nan

    @property
    def variance(self) -> float:
        return self._m2 / (self.count - 1) if self.count > 1 else math.nan

    @property
    def std(self) -> float:
        v = self.variance
        return math.sqrt(v) if not math.isnan(v) else math.nan

    def quantile(self, q: float) -> float:
        if not self.keep_values:
            raise ValueError(f"tally {self.name!r} does not keep values; quantiles unavailable")
        if not self.values:
            return math.nan
        return float(np.quantile(self.values, q))

    def histogram(self) -> dict[str, list[float]]:
        if not self.values:
            return {"edges": [], "counts": []}
        counts, edges = np.histogram(
            self.values, bins=self.bins if self.bins is not None else "auto"
        )
        return {"edges": edges.tolist(), "counts": counts.astype(float).tolist()}

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min if self.count else math.nan,
            "max": self.max if self.count else math.nan,
            "sum": self.total,
        }
        if self.keep_values and self.values:
            qs = np.quantile(self.values, DEFAULT_QUANTILES)
            for q, v in zip(DEFAULT_QUANTILES, qs, strict=True):
                out[f"p{round(q * 100)}"] = float(v)
        if self.bins is not None:
            out["histogram"] = self.histogram()
        return out


class TimeWeighted:
    """A piecewise-constant quantity integrated over simulation time.

    Typical uses: queue length, number of busy servers, work in progress.
    ``mean`` is the time average ``(1/T) * integral of value dt``. If
    ``record_series`` is set, every change is stored as ``(time, value)`` for
    plotting.
    """

    __slots__ = (
        "_area",
        "_area2",
        "_clock",
        "_last_t",
        "_start",
        "max",
        "min",
        "name",
        "series",
        "value",
    )

    def __init__(
        self,
        name: str,
        clock: Clock,
        initial: float = 0.0,
        *,
        record_series: bool = True,
        start: float | None = None,
    ) -> None:
        self.name = name
        self._clock = clock
        self.value = float(initial)
        self.series: list[tuple[float, float]] | None = [] if record_series else None
        self.reset()
        if start is not None and start < clock.now:
            # Backdate: the level is taken to have held `initial` since `start`.
            self._start = self._last_t = start
            if self.series is not None:
                self.series[0] = (start, self.value)

    def reset(self) -> None:
        now = self._clock.now
        self._start = now
        self._last_t = now
        self._area = 0.0
        self._area2 = 0.0
        self.min = self.value
        self.max = self.value
        if self.series is not None:
            self.series.clear()
            self.series.append((now, self.value))

    def record(self, value: float) -> None:
        now = self._clock._now
        dt = now - self._last_t
        if dt > 0:
            v = self.value
            self._area += v * dt
            self._area2 += v * v * dt
            self._last_t = now
        x = float(value)
        self.value = x
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x
        if self.series is not None:
            if self.series and self.series[-1][0] == now:
                self.series[-1] = (now, x)
            else:
                self.series.append((now, x))

    def add(self, delta: float) -> None:
        self.record(self.value + delta)

    def area(self, now: float | None = None) -> float:
        t = self._clock.now if now is None else now
        return self._area + self.value * max(0.0, t - self._last_t)

    def elapsed(self, now: float | None = None) -> float:
        t = self._clock.now if now is None else now
        return t - self._start

    @property
    def mean(self) -> float:
        span = self.elapsed()
        return self.area() / span if span > 0 else self.value

    @property
    def variance(self) -> float:
        span = self.elapsed()
        if span <= 0:
            return 0.0
        area2 = self._area2 + self.value**2 * max(0.0, self._clock.now - self._last_t)
        return max(0.0, area2 / span - self.mean**2)

    def summary(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": math.sqrt(self.variance),
            "min": self.min,
            "max": self.max,
            "final": self.value,
            "duration": self.elapsed(),
        }


class Counter:
    __slots__ = ("_clock", "_start", "name", "value")

    def __init__(self, name: str, clock: Clock, *, start: float | None = None) -> None:
        self.name = name
        self._clock = clock
        self.reset()
        if start is not None and start < clock.now:
            self._start = start

    def reset(self) -> None:
        self.value = 0.0
        self._start = self._clock.now

    def increment(self, by: float = 1.0) -> None:
        self.value += by

    @property
    def rate(self) -> float:
        span = self._clock.now - self._start
        return self.value / span if span > 0 else math.nan

    def summary(self) -> dict[str, Any]:
        return {"value": self.value, "rate": self.rate}


class Metrics:
    """Registry of named collectors.

    * ``observe(name, x)``   - one observation of a quantity (``Tally``)
    * ``histogram(name, x)`` - same, and the summary includes a histogram
    * ``record(name, x)``    - set a time-weighted level, e.g. WIP (``TimeWeighted``)
    * ``increment(name)``    - count occurrences; summary includes rate per time unit
    * ``set(name, x)``       - a final scalar (e.g. computed cost)
    """

    def __init__(
        self, clock: Clock, *, keep_values: bool = True, record_series: bool = True
    ) -> None:
        self._clock = clock
        self._start = clock.now
        self.keep_values = keep_values
        self.record_series = record_series
        self.tallies: dict[str, Tally] = {}
        self.gauges: dict[str, TimeWeighted] = {}
        self.counters: dict[str, Counter] = {}
        self.values: dict[str, float] = {}

    def tally(self, name: str, *, bins: int | Sequence[float] | None = None) -> Tally:
        t = self.tallies.get(name)
        if t is None:
            t = self.tallies[name] = Tally(name, keep_values=self.keep_values, bins=bins)
        return t

    def gauge(self, name: str, initial: float = 0.0) -> TimeWeighted:
        """Get or create a time-weighted level. A new gauge is backdated to the
        start of the statistics window with value ``initial``."""
        g = self.gauges.get(name)
        if g is None:
            g = self.gauges[name] = TimeWeighted(
                name, self._clock, initial, record_series=self.record_series, start=self._start
            )
        return g

    def counter(self, name: str) -> Counter:
        c = self.counters.get(name)
        if c is None:
            c = self.counters[name] = Counter(name, self._clock, start=self._start)
        return c

    def observe(self, name: str, value: float) -> None:
        self.tally(name).observe(value)

    def histogram(self, name: str, value: float, bins: int | Sequence[float] = 20) -> None:
        t = self.tally(name, bins=bins)
        if t.bins is None:
            t.bins = bins
        t.observe(value)

    def record(self, name: str, value: float) -> None:
        """Set a level. If the gauge is new, it counts as 0 before this first record."""
        g = self.gauges.get(name)
        if g is None:
            g = self.gauge(name, value if self._clock.now <= self._start else 0.0)
        g.record(value)

    def increment(self, name: str, by: float = 1.0) -> None:
        self.counter(name).increment(by)

    def set(self, name: str, value: float) -> None:
        self.values[name] = float(value)

    def reset(self) -> None:
        """Discard statistics gathered so far (e.g. at the end of a warm-up period)."""
        self._start = self._clock.now
        for t in self.tallies.values():
            t.reset()
        for g in self.gauges.values():
            g.reset()
        for c in self.counters.values():
            c.reset()
        self.values.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "tallies": {k: v.summary() for k, v in sorted(self.tallies.items())},
            "gauges": {k: v.summary() for k, v in sorted(self.gauges.items())},
            "counters": {k: v.summary() for k, v in sorted(self.counters.items())},
            "values": dict(sorted(self.values.items())),
        }

    def flat(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, t in sorted(self.tallies.items()):
            for k, v in t.summary().items():
                if k != "histogram":
                    out[f"{name}.{k}"] = float(v)
        for name, g in sorted(self.gauges.items()):
            out[f"{name}.mean"] = g.mean
            out[f"{name}.max"] = g.max
            out[f"{name}.final"] = g.value
        for name, c in sorted(self.counters.items()):
            out[name] = c.value
            out[f"{name}.rate"] = c.rate
        out.update(self.values)
        return out
