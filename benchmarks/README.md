# Benchmarks

```bash
python benchmarks/run_benchmarks.py          # full suite, ~1 minute
python benchmarks/run_benchmarks.py --quick  # 10k / 100k only
simulsi benchmark                            # same measurements from the CLI
```

The script writes [`results/latest.json`](results/latest.json) (raw numbers
plus the environment) and [`results/latest.md`](results/latest.md).

## What is measured

| Benchmark | Description | Size means |
|---|---|---|
| `events` | Pure engine overhead: callback events in 100 interleaved chains with exponential delays (heap push/pop, clock, dispatch). | events |
| `processes` | An M/M/2 queue written with generator processes, a resource with contention, and metrics: about 3.85 events per customer. | customers |
| `experiment(workers=N)` | 16 replications of the built-in M/M/1 model (20,000 time units each), serial and in worker processes. | replications |
| `peak_memory_mb` | `tracemalloc` peak during a separate run (sizes up to 100k only, because tracing slows the run). | |

## Results

Measured on 2026-10-02 on the development machine. These are single
measurements, not averages over repeated runs; expect some run-to-run
variation. Re-run the script on your hardware rather than relying on them.

- Python 3.12.15 (CPython), NumPy 2.5.3, SimulSI 0.1.0.dev0
- macOS 26.6.2 on Apple silicon (arm64), 10 logical CPUs

| name | size | events | seconds | events/s | peak MB |
|---|---:|---:|---:|---:|---:|
| events | 10,000 | 10,000 | 0.023 | 437,190 | 0.02 |
| events | 100,000 | 100,000 | 0.228 | 438,656 | 0.02 |
| events | 1,000,000 | 1,000,000 | 2.281 | 438,466 | - |
| processes | 10,000 | 38,501 | 0.154 | 250,000 | 0.31 |
| processes | 100,000 | 385,581 | 1.595 | 241,692 | 0.54 |
| processes | 1,000,000 | 3,850,935 | 16.34 | 235,737 | - |
| experiment (1 worker) | 16 | 1,119,489 | 4.59 | 244,072 | - |
| experiment (2 workers) | 16 | 1,119,489 | 3.82 | 293,357 | - |
| experiment (4 workers) | 16 | 1,119,489 | 3.40 | 329,285 | - |

## Observations

* Throughput is flat from 10k to 1M events: the heap is O(log n) and
  finished processes and timeouts are freed by reference counting. An
  earlier version kept a reference cycle per timeout, so the cyclic garbage
  collector slowed long runs by about 30%. That was found with this suite and
  fixed.
* Memory stays small because statistics are incremental. Keeping raw
  observations (`keep_values`), time series (`record_series`), entity
  histories or an event log (`trace=True`) costs memory proportional to run
  length.
* Parallel speed-up is modest here (about 1.35x with 4 workers): each
  replication takes about 0.3 s, while starting a worker process with the
  `spawn` method (macOS, Windows) and importing NumPy/SciPy costs about a
  second. Longer replications parallelise better. Results are identical to
  serial runs either way (tested).
* `tests/performance/test_performance.py` guards against regressions with
  generous floors (about 10x below these numbers) and a linear-scaling check.

## Tracking in CI

The [Benchmarks workflow](https://github.com/Yashjindal11/simulsi/actions/workflows/benchmarks.yml)
runs [`track.py`](track.py) on every push to `main` that touches the code:
engine throughput (events and process models, best of 3), a small serial
experiment and the SimulSI / SimPy time ratio. It compares them with the
previous successful run, writes the table to the job summary, keeps the
JSON as an artifact for 90 days and raises a warning when a number gets
more than 25% worse. Shared runners are noisy, so it warns rather than
fails; treat a regression as real when it persists.

```bash
python benchmarks/track.py --output now.json --previous before.json
```

## Comparison with SimPy

[`compare_simpy.py`](compare_simpy.py) runs the same M/M/2 queue (arrival
rate 1.8, service rate 1.0, a generator process per customer, NumPy draws)
in SimulSI and in [SimPy](https://simpy.readthedocs.io/), and checks that
both produce the same mean wait.

```bash
pip install simpy
python benchmarks/compare_simpy.py --customers 100000 --repeat 3
```

Measured on 2026-10-06 (Python 3.12.15, SimulSI 0.3.0.dev, SimPy 4.1.2,
Apple silicon, best of 3):

| library | seconds | customers/s | mean wait |
|---|---:|---:|---:|
| SimulSI | 1.35 | 73,800 | 4.6536 |
| SimPy | 0.84 | 118,700 | 4.6536 |

The two give identical results, and SimulSI takes about 1.6x as long. Part of
the gap is extra work SimulSI does on every request and release: it records
time-weighted utilization and queue length, a wait-time tally and counters,
and tracks which process holds each unit so unreleased resources can be
reported. SimPy leaves that bookkeeping to the model. The rest is per-event
overhead (lazy cancellation, the event status machine). Profiling for this
comparison led to hot-path changes that cut the gap from 1.85x and made the
`processes` benchmark about 14% faster.

If raw speed on very large models matters most, SimPy (or a compiled
engine) is the better choice. SimulSI is aimed at the experiment layer:
replications, statistics, scenarios and analysis.
