"""Trace-driven input: replay recorded arrivals and values instead of sampling.

* :func:`load_trace` - read a CSV into a list of records (numbers converted).
* :func:`trace_arrivals` - start a source that creates an arrival at each
  recorded time, passing the record to ``spawn``.
* :class:`TraceReplay` - hand out recorded values (e.g. service times) in
  order, as a drop-in for ``distribution.sample(stream)``.

Trace-driven runs are useful for validating a model against history: with
the real arrivals and service times, the model should reproduce the real
waits. Every replication of a trace-driven input is identical, so vary
something else (or use sampled inputs) for confidence intervals.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simulsi.errors import SimulsiError

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation
    from simulsi.processes.process import Process
    from simulsi.randomness.stream import RandomStream


class TraceExhausted(SimulsiError):
    """A :class:`TraceReplay` ran out of values."""


def _convert(cell: str) -> Any:
    try:
        return int(cell)
    except ValueError:
        pass
    try:
        return float(cell)
    except ValueError:
        return cell


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV file with a header row; numeric cells become ``int``/``float``."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [{k: _convert(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def trace_arrivals(
    sim: Simulation,
    records: Iterable[float | Mapping[str, Any]],
    spawn: Callable[[Simulation, Any], Any],
    *,
    time: str = "time",
    name: str = "trace",
) -> Process:
    """Create one arrival per record at its recorded time (measured from now).

    ``records`` are numbers (the times) or mappings with a ``time`` field.
    ``spawn(sim, record)`` is called at each arrival; if it returns a
    generator, that is started as a process. Times must be non-decreasing
    and non-negative.
    """
    rows = list(records)
    times: list[float] = []
    for i, r in enumerate(rows):
        t = float(r[time]) if isinstance(r, Mapping) else float(r)
        if t < 0 or (times and t < times[-1]):
            raise ValueError(f"trace times must be non-negative and sorted (record {i}: {t})")
        times.append(t)
    origin = sim.now

    def run() -> Any:
        for t, record in zip(times, rows, strict=True):
            delay = origin + t - sim.now
            if delay > 0:
                yield delay
            out = spawn(sim, record)
            if isinstance(out, Generator):
                sim.process(out)

    return sim.process(run(), name=name)


class TraceReplay:
    """Recorded values handed out in order.

    ``service = TraceReplay(times)`` then ``service.sample()`` (or
    ``service.sample(stream)``, so it can stand in for a distribution) gives
    the next value. Raises :class:`TraceExhausted` at the end unless
    ``cycle=True``. Create it inside the model's build function so every run
    starts from the first value.
    """

    def __init__(self, values: Sequence[Any], *, cycle: bool = False) -> None:
        if len(values) == 0:
            raise ValueError("a trace needs at least one value")
        self.values = list(values)
        self.cycle = cycle
        self.position = 0

    @property
    def remaining(self) -> int | float:
        return float("inf") if self.cycle else len(self.values) - self.position

    def sample(self, stream: RandomStream | None = None) -> Any:
        if self.position >= len(self.values):
            if not self.cycle:
                raise TraceExhausted(f"trace of {len(self.values)} values is exhausted")
            self.position = 0
        value = self.values[self.position]
        self.position += 1
        return value

    def __len__(self) -> int:
        return len(self.values)
