# Roadmap

What is planned, what was considered and deliberately left out, and why.
Suggestions are welcome as [issues](https://github.com/Yashjindal11/simulsi/issues)
(there is a *Model request* template for new built-in models).

## Next

- **Faster core.** The process engine takes about 1.6x as long as SimPy on
  a like-for-like queue (see the
  [benchmarks](https://github.com/Yashjindal11/simulsi/tree/main/benchmarks)).
  The plan is to keep the package pure Python and remove more per-event
  work. A compiled extension (Cython or mypyc) would complicate wheels for
  every platform and the in-browser demo, so it is only an option if the
  pure-Python route stalls.
- **Distributed experiments.** Experiments already run on local worker
  processes. An optional Ray or Dask backend for clusters fits the existing
  `workers=` design, but it is a dependency-heavy feature that needs real
  cluster testing first.
- **Event-driven `wait_until`.** Predicates are checked after every event.
  An opt-in form that re-checks only when given resources or containers
  change would make many pending conditions cheap.
- **Spatial and network models.** Movement on grids and road or airline
  networks with travel times, for richer epidemic, ride-hailing and
  logistics models.
- **Continuous dynamics.** Levels that change continuously between events
  (tanks, batteries, fluids) with threshold-crossing events.
- **More flowchart features.** Batching, split/join and resource schedules
  driven by parameters in YAML flowcharts, and a visual flowchart view in
  the dashboard.

## Done recently

See the [changelog](https://github.com/Yashjindal11/simulsi/blob/main/CHANGELOG.md):
YAML flowchart models; side-by-side replay; fifteen built-in models with
what-if presets; Pareto search; input-uncertainty and rare-event analysis;
the cross-entropy method; an in-browser demo; model plugins through entry
points; validation of the built-in models against theory.

## Out of scope

- A graphical drag-and-drop model editor. Flowchart YAML plus the dashboard
  covers simple models; anything more is clearer in Python.
- Real-time or hardware-in-the-loop simulation.
- Domain data sets. All models ship with synthetic data only.
