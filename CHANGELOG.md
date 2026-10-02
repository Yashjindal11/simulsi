# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-02

### Added

- Preemptive resources (`preemptive=True` with the priority discipline);
  evicted processes receive `Interrupt(Preempted(...))`.
- Sobol first-order and total-effect sensitivity indices
  (`simulsi.analysis.sobol_indices`) with bootstrap CIs, validated against
  the Ishigami function.
- Checkpoints of running simulations (`sim.save_checkpoint`,
  `Simulation.load_checkpoint`) restored by verified deterministic replay.
- PyPI publishing workflow (trusted publishing).

### Fixed

- Hold times in event-log records were wrong for units granted at t = 0.

## [0.1.0] - 2026-10-02

First public release.

### Added

- Discrete-event engine: float clock with `timedelta`/`datetime` conversion,
  heap-based event queue with deterministic `(time, priority, sequence)`
  ordering, scheduling, rescheduling, cancellation, `run(until=...)`
  that stops exactly at the horizon, snapshots and a structured event log
  (JSON/CSV/Parquet export).
- Generator-based processes with timeouts, signals, `all_of`/`any_of`,
  waiting on other processes, interrupts and error propagation.
- Resources with FIFO/LIFO/priority/custom disciplines, reneging,
  capacity changes, downtime, automatic statistics; blocking queues.
- Entities with attributes, state history and time-in-system metrics.
- Seeded random streams (PCG64, named sub-streams via `SeedSequence`) and 12
  distributions with validation and configuration specs.
- Metrics: tallies, time-weighted gauges, counters, histograms, warm-up reset.
- Models (`Model`, `@model`, `Parameter`), scenarios, full-factorial grids.
- Experiments with common random numbers, provenance (git commit,
  environment), parallel workers, checkpoint/resume and JSON/CSV/Parquet
  export.
- Monte Carlo (random and Latin hypercube sampling, vectorised models,
  simulation models), statistics (t/bootstrap/Wilson intervals, replication
  advice, convergence, batch means, paired and Welch differences), scenario
  comparison, sensitivity analysis (OAT, finite differences, Pearson,
  Spearman, SRC).
- Failure/recovery processes, scheduled disruptions, cost models and an
  optimisation `Objective` adapter.
- Model validation (smoke run with deadlock / unreleased-resource / state
  checks) and flow/state graph introspection (Mermaid, DOT).
- Optional matplotlib/Plotly visualisation.
- YAML experiment configuration (strict schema, safe loading, path checks)
  and the `simulsi` CLI: `init`, `validate`, `run`, `experiment`, `analyze`,
  `benchmark`, `visualize`, `ui`.
- Optional React + TypeScript dashboard (`simulsi ui`): load experiments,
  view parameters and provenance, run models, compare scenarios, inspect
  distributions and convergence, trace single runs (timeline, queues,
  utilization) and export JSON/CSV. Served locally on 127.0.0.1.
- Six domain examples: bank, hospital, warehouse, manufacturing,
  transportation, aviation - plus a minimal M/M/c example.
- Benchmark suite (`benchmarks/run_benchmarks.py`, `simulsi benchmark`) with
  measured results, and performance regression tests.
- Documentation: user guide, architecture, research workflow, extension guide,
  FAQ; documentation code blocks are executed by the test suite.

### Known limitations

- No preemptive resources, continuous-time dynamics or single-run
  checkpoint/restore (experiments are checkpointed per replication).
- Sensitivity analysis is local / correlation-based; no Sobol indices yet.
- Not yet published on PyPI; install from source.
