# Roadmap

What is planned, what was considered and deliberately left out, and why.
Suggestions are welcome as [issues](https://github.com/Yashjindal11/simulsi/issues)
(there is a *Model request* template for new built-in models).

## Next

- **Faster core.** The process engine takes about 1.7x as long as SimPy on
  a like-for-like M/M/2 queue (see the
  [benchmarks](https://github.com/Yashjindal11/simulsi/tree/main/benchmarks)).
  A profile shows no single hot spot left: the gap is spread over the
  statistics simulsi collects by default on every grant and release
  (time-weighted busy units and queue length, waits), the process
  bookkeeping that makes deadlock warnings and checkpoints possible, and
  the event queue. The package stays pure Python; a compiled extension
  (Cython or mypyc) would complicate wheels for every platform and the
  in-browser demo, so it is only an option if users hit real limits.
- **Ray backend.** Experiments run on any `concurrent.futures` executor
  (local processes or Dask). Ray has no such executor built in, so it
  needs a small adapter and real cluster testing.
- **Flowchart view.** A visual view of YAML flowcharts in the dashboard.
- **Crew pairings across days.** The airline twin simulates one day, so
  crew rest rules that span nights are out of its reach today.

## Done recently

See the [changelog](https://github.com/Yashjindal11/simulsi/blob/main/CHANGELOG.md):
event-driven `wait_until(on=[...])`; continuous `Level`s with exact
threshold crossings; movement on networks and grids (`simulsi.spatial`);
batch and parallel (split/join) stations in YAML flowcharts; experiments
on Dask clusters; animated GIF replays;
airline operations on real data (delay causes, airport good and bad days,
cancellations, storm cancellations, probability recalibration, passenger
rebooking, ferried spares, a live status feed and recovery and reserve
planning in the dashboard);
airline operations (day-ahead and live forecasts, recovery search, reserve
and buffer planning, calibration and backtesting);
YAML flowchart models; side-by-side replay; fifteen built-in models with
what-if presets; Pareto search; input-uncertainty and rare-event analysis;
the cross-entropy method; an in-browser demo; model plugins through entry
points; validation of the built-in models against theory.

## Out of scope

- A graphical drag-and-drop model editor. Flowchart YAML plus the dashboard
  covers simple models; anything more is clearer in Python.
- Real-time or hardware-in-the-loop simulation.
- Domain data sets. All models ship with synthetic data only.
