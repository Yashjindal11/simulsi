# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Flowchart models in YAML (`simulsi.flowchart`): sources, resource and delay
  stations, probabilistic routing, patience, shifts, parameters (`$name`)
  and presets; usable anywhere a model reference is accepted.
- Built-in models `airport_turnaround` (parallel ground-handling tasks and
  the critical path) and `disruption_recovery` (delay vs cancel vs spares
  after a hub storm).
- Multi-objective optimisation: `pareto_search`, `pareto_front`,
  `ParetoResult.plot()` and `simulsi pareto`.
- "Try it in your browser" docs page running SimulSI with Pyodide.
- Six more built-in models: `emergency_department`, `restaurant`,
  `ev_charging`, `inventory` ((s, S) policy), `theme_park` and `warehouse`.
- Validation tests for the built-in models against theory (final-size
  equation, Webster, service-time tails) and limiting cases.
- Dashboard: side-by-side replay of two parameter sets (or presets) on one
  clock with a metric diff, and charts for every series a model records.

## [0.5.0] - 2026-10-07

### Added

- Six new built-in models: `airline` (delay propagation), `epidemic`
  (stochastic SEIR with hospital capacity and lockdowns), `supply_chain`
  (bullwhip effect), `ride_hailing` (pickup feedback and surge pricing),
  `cloud_autoscaling` (cold starts and scaling policies) and
  `traffic_signal` (fixed vs actuated timing, checked against Webster).
- Model presets: named what-if scenarios (`Model(presets=...)`,
  `model.preset_scenarios()`), shown in `describe()` and loadable in the
  dashboard.
- `simulsi models` and `simulsi whatif` (parameter sweeps and preset
  comparisons with CIs and a bar chart).
- A model gallery in the docs.

## [0.4.0] - 2026-10-07

### Added

- Morris elementary-effects screening (`analysis.morris_screening`).
- Ranking and selection with procedure KN (`analysis.select_best`).
- Built-in optimisation: `optimization.optimize` with grid, Latin hypercube
  and Bayesian (Gaussian process + expected improvement) search, confirmed by
  ranking and selection.
- Gaussian-process surrogates: `GaussianProcess`, `fit_surrogate`, `Surrogate`
  (leave-one-out accuracy, no extrapolation).
- Self-contained HTML reports: `ExperimentResult.report()`, `simulsi report`
  and a dashboard download.
- Jupyter display (`_repr_html_`) for results, comparisons, sensitivity,
  Monte Carlo, fits, selection and optimisation.
- Dashboard: an *Explore* page (grid sweeps, Morris screening, Monte Carlo),
  a *Response* tab for multi-scenario results, and replay of traced runs.
- Tutorial notebooks in `examples/notebooks/` (executed in the test suite).
- A SimPy migration guide (`docs/simpy-migration.md`).
- Docker image with the CLI and dashboard (`Dockerfile`; published to
  `ghcr.io/yashjindal11/simulsi` for each release).
- Benchmark tracking in CI (`benchmarks/track.py`, Benchmarks workflow) with
  warnings on regressions.

## [0.3.0] - 2026-10-06

### Added

- Per-run warm-up override (`Model.simulate(warmup=)`, `simulsi run --warmup`);
  the model warm-up is scaled proportionally for shorter runs, with a note.
- Dashboard runs experiments as background jobs with a progress bar
  (`POST /api/run` returns 202, `GET /api/jobs/{id}`).
- Multiple-comparison adjustment in `compare(adjust="bonferroni"|"holm"|"bh")`,
  the CLI (`--adjust`), configs and the dashboard.
- Warm-up detection with MSER-5 (`simulsi.analysis.mser`, `suggest_warmup`,
  `simulsi warmup`).
- Sequential experiments: `Experiment.run_until(relative_precision, ...)` and
  `simulsi experiment --until-precision`.
- Antithetic variates (`Experiment(antithetic=True)`, inverse-transform
  sampling modes on `RandomStream`) and control variates
  (`ExperimentResult.control_variate`, `statistics.control_variate`).
- Documentation site with API reference (mkdocs-material, GitHub Pages).
- Modelling primitives: `Container` (`sim.container`) for levels and stock;
  filtered gets on queues (`queue.get(filter=...)`); `sim.wait_until(predicate)`;
  shift schedules (`capacity_schedule`); non-stationary arrivals
  (`arrivals` with `PiecewiseRate` or a rate function); `batch`, `split`/`join`,
  `Router` and `shortest_queue`.
- Input modelling: `fit_distribution` (MLE fits ranked by AIC/BIC/KS, with
  `FitReport.format()` and `plot()`), `load_values` and `simulsi fit`;
  trace-driven input with `load_trace`, `trace_arrivals` and `TraceReplay`.

### Changed

- Faster process engine (about 14% more events per second on process models);
  `benchmarks/compare_simpy.py` compares against SimPy.
- Worker process pools are reused across experiments in the same process
  (`shutdown_workers()` to release them).

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
