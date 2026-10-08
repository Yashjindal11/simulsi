"""Ground operations: the turnaround critical path, minimum turn times and minimum connection times."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulsi.core.simulation import Simulation
from simulsi.errors import ConfigError
from simulsi.processes.process import Signal
from simulsi.randomness.stream import RandomStream, derive_seed


@dataclass(frozen=True)
class TurnaroundTask:
    """A ground task with a triangular duration (minutes) that starts once ``after`` are done.

    ``disruption`` is the chance the task overruns, by an exponential extra
    time with mean ``disruption_mean`` (a late fuel truck, a missing
    caterer, a slow wheelchair assist).
    """

    name: str
    low: float
    mode: float
    high: float
    after: tuple[str, ...] = ()
    disruption: float = 0.0
    disruption_mean: float = 10.0

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ConfigError(f"task {self.name}: need low <= mode <= high")


NARROWBODY = (
    TurnaroundTask("deboard", 6, 8, 12),
    TurnaroundTask("unload_bags", 8, 10, 15),
    TurnaroundTask("clean", 7, 10, 15, ("deboard",)),
    TurnaroundTask("cater", 6, 8, 14, ("deboard",), 0.05, 10),
    TurnaroundTask("fuel", 8, 12, 18, ("deboard",), 0.05, 12),
    TurnaroundTask("load_bags", 8, 11, 16, ("unload_bags",)),
    TurnaroundTask("board", 12, 15, 24, ("clean", "cater", "fuel"), 0.1, 6),
    TurnaroundTask("close_doors", 2, 3, 5, ("board", "load_bags")),
    TurnaroundTask("pushback", 2, 3, 6, ("close_doors",)),
)
"""A typical A320/737 turnaround (minutes), fuelling with passengers off."""


@dataclass
class TurnaroundResult:
    totals: np.ndarray
    criticality: dict[str, float]
    mean_duration: dict[str, float]
    mean_slack: dict[str, float]
    tasks: tuple[TurnaroundTask, ...] = field(repr=False, default=())

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.totals, q))

    def min_turn(self, reliability: float = 0.95) -> float:
        """The scheduled turn time that is long enough on ``reliability`` of days."""
        return math.ceil(self.quantile(reliability))

    def table(self) -> list[dict[str, Any]]:
        return [
            {
                "task": t.name,
                "after": ", ".join(t.after) or "-",
                "mean_minutes": round(self.mean_duration[t.name], 1),
                "critical_share": round(self.criticality[t.name], 3),
                "mean_slack": round(self.mean_slack[t.name], 1),
            }
            for t in self.tasks
        ]

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        q = {p: self.quantile(p) for p in (0.5, 0.8, 0.95)}
        head = (
            f"turnaround: median {q[0.5]:.1f} min, p80 {q[0.8]:.1f}, p95 {q[0.95]:.1f} "
            f"over {len(self.totals)} simulated turns"
        )
        varying = {k: v for k, v in self.criticality.items() if v < 0.999} or self.criticality
        bottleneck = max(varying, key=varying.__getitem__)
        return (
            head
            + "\n"
            + format_table(self.table())
            + f"\nbottleneck: of the tasks that run in parallel, {bottleneck} sets the turn time "
            + f"in {self.criticality[bottleneck]:.0%} of turns"
        )


def turnaround(
    tasks: Sequence[TurnaroundTask] = NARROWBODY,
    *,
    replications: int = 2000,
    seed: int = 0,
    crews: dict[str, int] | None = None,
) -> TurnaroundResult:
    """Simulate a turnaround as a network of ground tasks and find the critical path.

    Each task is a simulsi process that waits for its predecessors, then
    takes a sampled duration. ``crews`` optionally limits shared teams
    (``{"clean": 1}`` means cleaning can only start when the cleaning team
    is free - useful for banked schedules). The criticality index of a task
    is the share of turns where it lies on the longest (critical) path.

    >>> r = turnaround(replications=200)
    >>> 35 < r.quantile(0.5) < 60
    True
    """
    names = [t.name for t in tasks]
    if len(set(names)) != len(names):
        raise ConfigError("duplicate task names")
    for t in tasks:
        missing = set(t.after) - set(names)
        if missing:
            raise ConfigError(f"task {t.name}: unknown predecessors {sorted(missing)}")
    _check_acyclic(tasks)
    totals = []
    critical = dict.fromkeys(names, 0)
    durations = dict.fromkeys(names, 0.0)
    slack = dict.fromkeys(names, 0.0)
    by_name = {t.name: t for t in tasks}
    for r in range(replications):
        start, end = _one_turn(tasks, derive_seed(seed, "turn", r), crews or {})
        total = max(end.values())
        totals.append(total)
        # walk back along the latest-finishing predecessor
        node = max(end, key=end.__getitem__)
        while True:
            critical[node] += 1
            preds = by_name[node].after
            if not preds:
                break
            node = max(preds, key=end.__getitem__)
        latest_start = _latest_starts(tasks, end, start, total)
        for name in names:
            durations[name] += end[name] - start[name]
            slack[name] += latest_start[name] - start[name]
    reps = replications
    return TurnaroundResult(
        np.array(totals),
        {k: v / reps for k, v in critical.items()},
        {k: v / reps for k, v in durations.items()},
        {k: v / reps for k, v in slack.items()},
        tuple(tasks),
    )


def _one_turn(
    tasks: Sequence[TurnaroundTask], seed: int, crews: dict[str, int]
) -> tuple[dict[str, float], dict[str, float]]:
    """Start and end time of every task in one simulated turnaround."""
    sim = Simulation(seed=seed, record_series=False, keep_values=False)
    rs = sim.stream("tasks")
    done: dict[str, Signal] = {t.name: sim.signal(t.name) for t in tasks}
    start: dict[str, float] = {}
    end: dict[str, float] = {}
    teams = {k: sim.resource(f"team.{k}", v) for k, v in crews.items()}

    def run(t: TurnaroundTask) -> Any:
        if t.after:
            yield sim.all_of([done[a] for a in t.after])
        req = None
        if t.name in teams:
            req = yield sim.request(teams[t.name])
        start[t.name] = sim.now
        d = rs.triangular(t.low, t.mode, t.high)
        if t.disruption > 0 and rs.random() < t.disruption:
            d += rs.exponential(t.disruption_mean)
        yield d
        if req is not None:
            sim.release(req)
        end[t.name] = sim.now
        done[t.name].succeed(sim.now)

    for t in tasks:
        sim.process(run(t), name=t.name)
    sim.run()
    return start, end


def _check_acyclic(tasks: Sequence[TurnaroundTask]) -> None:
    after = {t.name: set(t.after) for t in tasks}
    done: set[str] = set()
    while after:
        ready = [n for n, deps in after.items() if deps <= done]
        if not ready:
            raise ConfigError(f"turnaround tasks have a cycle among {sorted(after)}")
        for n in ready:
            done.add(n)
            del after[n]


def _latest_starts(
    tasks: Sequence[TurnaroundTask], end: dict[str, float], start: dict[str, float], total: float
) -> dict[str, float]:
    succ: dict[str, list[str]] = {t.name: [] for t in tasks}
    for t in tasks:
        for a in t.after:
            succ[a].append(t.name)
    latest_finish: dict[str, float] = {}
    for name in reversed(_topo(tasks)):
        latest_finish[name] = min(
            (latest_finish[s] - (end[s] - start[s]) for s in succ[name]), default=total
        )
    return {n: latest_finish[n] - (end[n] - start[n]) for n in latest_finish}


def _topo(tasks: Sequence[TurnaroundTask]) -> list[str]:
    after = {t.name: set(t.after) for t in tasks}
    order: list[str] = []
    while after:
        ready = sorted(n for n, deps in after.items() if deps <= set(order))
        order.extend(ready)
        for n in ready:
            del after[n]
    return order


def recommend_min_turn(
    tasks: Sequence[TurnaroundTask] = NARROWBODY,
    *,
    reliability: float = 0.95,
    replications: int = 2000,
    seed: int = 0,
) -> float:
    """Shortest scheduled turn that is achieved on ``reliability`` of turns."""
    return turnaround(tasks, replications=replications, seed=seed).min_turn(reliability)


@dataclass
class MCTResult:
    mct: float
    reliability: float
    table: list[dict[str, float]]

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        return (
            format_table(self.table)
            + f"\nminimum connection time for {self.reliability:.0%}: {self.mct:g} min"
        )


def recommend_mct(
    *,
    reliability: float = 0.95,
    walk: tuple[float, float, float] = (8.0, 12.0, 25.0),
    gate_close: float = 15.0,
    inbound_delay: tuple[float, float] = (0.25, 20.0),
    outbound_delay: tuple[float, float] = (0.2, 15.0),
    inbound_samples: Sequence[float] | None = None,
    outbound_samples: Sequence[float] | None = None,
    samples: int = 20000,
    seed: int = 0,
    candidates: Sequence[float] = (25, 30, 35, 40, 45, 50, 60, 75, 90),
) -> MCTResult:
    """The minimum connection time that ``reliability`` of passengers make.

    A passenger makes the connection when the inbound arrival delay, plus
    the walk (triangular ``walk`` minutes) and the gate closing
    ``gate_close`` minutes before departure, fits in the scheduled
    connection plus the outbound's own delay. Delays are "probability,
    mean" (exponential), or empirical samples - for example arrival delays
    from a :class:`~simulsi.aviation.Forecast`.
    """
    rs = RandomStream(seed)

    def draw(spec: tuple[float, float], given: Sequence[float] | None) -> np.ndarray:
        if given is not None:
            arr = np.asarray(given, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                raise ConfigError("empty delay samples")
            return np.array([arr[rs.integers(0, len(arr))] for _ in range(samples)])
        p, mean = spec
        return np.array([rs.exponential(mean) if rs.random() < p else 0.0 for _ in range(samples)])

    din = np.maximum(draw(inbound_delay, inbound_samples), 0.0)
    dout = np.maximum(draw(outbound_delay, outbound_samples), 0.0)
    w = np.array([rs.triangular(*walk) for _ in range(samples)])
    need = din - dout + w + gate_close
    rows = [
        {"connection_minutes": float(c), "made": round(float(np.mean(need <= c)), 4)}
        for c in candidates
    ]
    mct = float(math.ceil(np.quantile(need, reliability) / 5) * 5)
    return MCTResult(mct, reliability, rows)
