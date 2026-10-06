"""Same M/M/2 queue in SimulSI and SimPy, timed side by side.

    pip install simpy
    python benchmarks/compare_simpy.py [--customers 100000] [--repeat 3]

Both versions use NumPy exponential draws, a capacity-2 resource and a
generator process per customer, and record each customer's wait. The
SimulSI run also collects its usual resource statistics (utilization,
queue length, wait tallies), which SimPy does not; that is part of the
comparison, not hidden from it. Reports the best of ``--repeat`` runs.
"""

from __future__ import annotations

import argparse
import platform
import time
from collections.abc import Callable
from typing import Any

import numpy as np

ARRIVAL, SERVICE, SERVERS = 1.8, 1.0, 2


def simulsi_mm2(n: int, seed: int = 1) -> tuple[float, float]:
    from simulsi import Simulation

    sim = Simulation(seed=seed, keep_values=False, record_series=False, keep_entity_history=False)
    server = sim.resource("server", SERVERS)
    rng = np.random.default_rng(seed)
    waits: list[float] = []

    def customer(sim: Simulation, service: float) -> Any:
        t0 = sim.now
        req = server.request()
        yield req
        waits.append(sim.now - t0)
        yield service
        server.release(req)

    def source(sim: Simulation) -> Any:
        for _ in range(n):
            yield float(rng.exponential(1 / ARRIVAL))
            sim.process(customer(sim, float(rng.exponential(1 / SERVICE))))

    sim.process(source(sim))
    t = time.perf_counter()
    sim.run()
    return time.perf_counter() - t, float(np.mean(waits))


def simpy_mm2(n: int, seed: int = 1) -> tuple[float, float]:
    import simpy

    env = simpy.Environment()
    server = simpy.Resource(env, SERVERS)
    rng = np.random.default_rng(seed)
    waits: list[float] = []

    def customer(env: simpy.Environment, service: float) -> Any:
        t0 = env.now
        with server.request() as req:
            yield req
            waits.append(env.now - t0)
            yield env.timeout(service)

    def source(env: simpy.Environment) -> Any:
        for _ in range(n):
            yield env.timeout(float(rng.exponential(1 / ARRIVAL)))
            env.process(customer(env, float(rng.exponential(1 / SERVICE))))

    env.process(source(env))
    t = time.perf_counter()
    env.run()
    return time.perf_counter() - t, float(np.mean(waits))


def best(fn: Callable[[int], tuple[float, float]], n: int, repeat: int) -> tuple[float, float]:
    runs = [fn(n) for _ in range(repeat)]
    return min(r[0] for r in runs), runs[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--customers", type=int, default=100_000)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    import simpy

    import simulsi

    n = args.customers
    s_t, s_w = best(simulsi_mm2, n, args.repeat)
    p_t, p_w = best(simpy_mm2, n, args.repeat)
    print(
        f"Python {platform.python_version()}, SimulSI {simulsi.__version__}, "
        f"SimPy {simpy.__version__}, {n:,} customers, best of {args.repeat}"
    )
    print("| library | seconds | customers/s | mean wait |")
    print("|---|---:|---:|---:|")
    for name, t, w in (("SimulSI", s_t, s_w), ("SimPy", p_t, p_w)):
        print(f"| {name} | {t:.3f} | {n / t:,.0f} | {w:.4f} |")
    print(f"\nSimulSI / SimPy time ratio: {s_t / p_t:.2f}")


if __name__ == "__main__":
    main()
