"""Warm-up (initial transient) detection.

Steady-state estimates are biased by the start-up period of a run (an empty
system has short queues). :func:`mser` implements MSER (White 1997; MSER-5
with batches of 5, Franklin & White 2008): delete the first ``d``
observations where ``d`` minimises the marginal standard error of the
remaining mean. :func:`suggest_warmup` applies it to a model: it averages a
recorded time series across replications (reducing noise, as in Welch's
method), and reports the truncation point as a simulation time.

MSER is a heuristic. If the suggested truncation is close to the maximum
allowed fraction, the run is probably too short to reach steady state - the
result flags this rather than returning a misleading number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from simulsi.core.model import Model
from simulsi.randomness.stream import derive_seed

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MserResult:
    truncation: int
    """Number of original observations to delete."""
    statistic: float
    n: int
    batch_size: int
    reliable: bool
    note: str = ""


def mser(
    values: Sequence[float] | FloatArray, batch_size: int = 5, max_fraction: float = 0.5
) -> MserResult:
    """MSER-``batch_size`` truncation point for one output sequence (in observation order)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    k = len(x) // batch_size
    if k < 4:
        return MserResult(0, math.nan, len(x), batch_size, False, "too few observations for MSER")
    b = x[: k * batch_size].reshape(k, batch_size).mean(axis=1)
    # suffix sums give mean and SSE of b[d:] for every d in O(k)
    rev = b[::-1]
    csum = np.cumsum(rev)[::-1]
    csum2 = np.cumsum(rev * rev)[::-1]
    m = k - np.arange(k)
    sse = csum2 - csum * csum / m
    stat = sse / (m * m)
    limit = max(1, int(k * max_fraction))
    d = int(np.argmin(stat[:limit]))
    reliable = d < limit - 1
    note = (
        ""
        if reliable
        else "minimum at the search limit: the run may be too short to reach steady state"
    )
    return MserResult(d * batch_size, float(stat[d]), len(x), batch_size, reliable, note)


def _resample(points: Sequence[tuple[float, float]], edges: FloatArray) -> FloatArray:
    """Time-average a step function over each bin ``[edges[i], edges[i+1])``."""
    t = np.asarray([p[0] for p in points], dtype=float)
    v = np.asarray([p[1] for p in points], dtype=float)
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # value in force at lo, then every change inside the bin
        j0 = max(0, int(np.searchsorted(t, lo, side="right")) - 1)
        j1 = int(np.searchsorted(t, hi, side="left"))
        times = np.concatenate([[lo], t[j0 + 1 : j1], [hi]])
        vals = v[j0:j1] if j1 > j0 else v[j0 : j0 + 1]
        widths = np.diff(times)
        out[i] = float(np.dot(widths, vals[: len(widths)]) / (hi - lo))
    return out


@dataclass
class WarmupAdvice:
    series: str
    warmup: float
    duration: float
    replications: int
    bins: int
    reliable: bool
    note: str = ""
    averaged: list[float] = field(default_factory=list)
    bin_edges: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def suggest_warmup(
    model: Model,
    params: Mapping[str, Any] | None = None,
    *,
    series: str | None = None,
    replications: int = 5,
    seed: int = 0,
    duration: float | None = None,
    bins: int = 200,
    batch_size: int = 5,
) -> WarmupAdvice:
    """Suggest a warm-up for ``model`` from a recorded time series.

    ``series`` is a key of ``SimulationResult.series`` (default: the first
    resource ``queue_length``). Each replication is run *without* warm-up, the
    series is time-averaged into ``bins`` equal bins, averaged across
    replications, and MSER-``batch_size`` picks the truncation bin.
    """
    horizon = duration if duration is not None else model.duration
    if horizon is None:
        raise ValueError("a duration is needed (model has none)")
    edges = np.linspace(0.0, float(horizon), bins + 1)
    rows = []
    key = series
    for r in range(replications):
        res = model.simulate(
            params,
            seed=derive_seed(seed, "replication", r),
            duration=horizon,
            warmup=0.0,
            record_series=True,
        )
        if key is None:
            candidates = sorted(k for k in res.series if k.endswith(".queue_length"))
            if not candidates:
                raise ValueError(
                    f"no queue_length series recorded; choose one of {sorted(res.series)}"
                )
            key = candidates[0]
        if key not in res.series:
            raise KeyError(f"no series {key!r}; recorded: {sorted(res.series)}")
        rows.append(_resample(res.series[key], edges))
    averaged = np.mean(np.vstack(rows), axis=0)
    m = mser(averaged, batch_size=batch_size)
    width = float(horizon) / bins
    assert key is not None
    return WarmupAdvice(
        series=key,
        warmup=m.truncation * width,
        duration=float(horizon),
        replications=replications,
        bins=bins,
        reliable=m.reliable,
        note=m.note,
        averaged=averaged.tolist(),
        bin_edges=edges.tolist(),
    )
