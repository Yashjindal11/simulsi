"""Reproducible micro- and macro-benchmarks (used by ``simulsi benchmark``).

Numbers depend on the machine and Python version; record them with the
environment (see :func:`simulsi.experiments.provenance.environment`).
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from simulsi.core.simulation import Simulation
from simulsi.experiments.experiment import Experiment
from simulsi.models.queueing import mmc
from simulsi.randomness.distributions import Exponential


@dataclass
class BenchmarkResult:
    name: str
    size: int
    events: int
    seconds: float
    peak_memory_mb: float | None = None

    @property
    def events_per_second(self) -> float:
        return self.events / self.seconds if self.seconds > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["events_per_second"] = self.events_per_second
        return d


def _timer_chain(n_events: int) -> Simulation:
    """Pure engine overhead: ``n_events`` callback events, 100 interleaved chains."""
    sim = Simulation(seed=1, keep_values=False, record_series=False)
    chains = 100
    per_chain = n_events // chains
    stream = sim.stream("delays")

    def make(remaining: list[int]) -> Callable[[], None]:
        def tick() -> None:
            remaining[0] -= 1
            if remaining[0] > 0:
                sim.call_at(tick, delay=stream.exponential(1.0))

        return tick

    for _ in range(chains):
        sim.call_at(make([per_chain]), delay=stream.exponential(1.0))
    return sim


def _process_queue(n_customers: int) -> Simulation:
    """Process-based M/M/2 queue: generator processes, resource contention, metrics."""
    sim = Simulation(seed=1, keep_values=False, record_series=False, keep_entity_history=False)
    server = sim.resource("server", 2)
    inter, svc = Exponential(rate=1.8), Exponential(rate=1.0)
    a, s = sim.stream("a"), sim.stream("s")

    def customer(sim: Simulation) -> Any:
        yield from sim.use(server, svc.sample(s))

    def source(sim: Simulation) -> Any:
        for _ in range(n_customers):
            yield inter.sample(a)
            sim.process(customer(sim))

    sim.process(source(sim))
    return sim


SCENARIOS: dict[str, Callable[[int], Simulation]] = {
    "events": _timer_chain,
    "processes": _process_queue,
}


def run_engine_benchmark(name: str, size: int, *, memory: bool = False) -> BenchmarkResult:
    sim = SCENARIOS[name](size)
    if memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    sim.run()
    dt = time.perf_counter() - t0
    peak = None
    if memory:
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = peak_b / 1e6
    return BenchmarkResult(name, size, sim.events_processed, dt, peak)


def run_experiment_benchmark(
    replications: int, workers: int, duration: float = 2_000.0
) -> BenchmarkResult:
    m = mmc.with_options(duration=duration, warmup=0.0)
    exp = Experiment(m, replications=replications, seed=0, workers=workers)
    t0 = time.perf_counter()
    res = exp.run()
    dt = time.perf_counter() - t0
    return BenchmarkResult(
        f"experiment(workers={workers})", replications, sum(r.events for r in res.records), dt
    )


def default_suite(
    sizes: tuple[int, ...] = (10_000, 100_000, 1_000_000),
    *,
    memory: bool = True,
    workers: tuple[int, ...] = (1, 4),
    replications: int = 16,
) -> list[BenchmarkResult]:
    out = []
    for name in SCENARIOS:
        for n in sizes:
            out.append(run_engine_benchmark(name, n))
            if memory and n <= 100_000:
                mem = run_engine_benchmark(name, n, memory=True)
                out[-1].peak_memory_mb = mem.peak_memory_mb
    for w in workers:
        out.append(run_experiment_benchmark(replications, w))
    return out
