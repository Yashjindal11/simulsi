# Visualization

Plotting is optional: install `simulsi[viz]` (matplotlib) or
`simulsi[plotly]`. Every function returns a figure and never calls `show()`;
pass `backend="plotly"` for interactive figures and `ax=` to draw into an
existing matplotlib axis. `save_figure(fig, "out.png")` handles both backends
(`.html` for Plotly).

## Single runs

Time series are recorded automatically (disable with
`Simulation(record_series=False)` for very long runs) and the event log with
`trace=True`.

```python
from simulsi.models import mmc
from simulsi.visualization import (plot_entity_trajectories, plot_queue_length, plot_series,
                                   plot_timeline, plot_utilization, save_figure)

model = mmc.with_options(duration=200, warmup=0, sim_options={"record_series": True})
run = model.simulate({"servers": 2, "arrival_rate": 1.8}, seed=1, trace=True)

save_figure(plot_queue_length(run), "queue.png")        # waiting line over time
save_figure(plot_utilization(run), "utilization.png")   # busy units vs capacity
save_figure(plot_series(run, ["resource.server.busy"]), "busy.png")
save_figure(plot_timeline(run.log), "timeline.png")     # events by type over time
```

`plot_entity_trajectories(log)` draws a Gantt chart of entity states (from
`entity.created` / `entity.state` records), for models that use entities:

```python
from simulsi import Simulation, model


@model(duration=60)
def desk(sim: Simulation, p):
    counter = sim.resource("counter", 1)

    def visitor(sim, v):
        v.set_state("queued")
        req = yield sim.request(counter)
        v.set_state("served")
        yield 4
        sim.release(req)
        sim.dispose(v)

    def source(sim):
        while True:
            yield sim.stream("a").exponential(3)
            v = sim.entity("visitor")
            sim.process(visitor(sim, v), entity=v)

    sim.process(source(sim))


traced = desk.simulate(seed=2, trace=True)
save_figure(plot_entity_trajectories(traced.log, max_entities=15, end_time=60), "trajectories.png")
```

## Experiments

```python
from simulsi import Experiment, Scenario, compare
from simulsi.visualization import plot_comparison, plot_convergence, plot_distribution

exp = Experiment(mmc.with_options(duration=1_000, warmup=100),
                 [Scenario("baseline", {"arrival_rate": 1.6, "servers": 2}),
                  Scenario("three", {"arrival_rate": 1.6, "servers": 3})],
                 replications=12, seed=1).run()
waits = exp.values("resource.server.wait.mean", "baseline")
save_figure(plot_distribution(waits, title="Mean wait per replication"), "dist.png")
save_figure(plot_convergence(waits), "convergence.png")     # running mean with CI band
cmp = compare(exp, "baseline", metrics=["resource.server.wait.mean", "resource.server.utilization"])
save_figure(plot_comparison(cmp, relative=True), "comparison.png")   # % change with CIs
```

`plot_sensitivity(result)` draws a tornado chart from any
`SensitivityResult`.

## HTML reports and notebooks

`result.report("report.html")` writes a single self-contained HTML file (no
JavaScript, no external resources): provenance, scenario definitions,
summaries with confidence intervals, a chart per metric, differences from the
baseline with adjusted p-values, failed replications and a reporting
checklist. `simulsi report results/my-study -o report.html` does the same
from saved results, and the dashboard has an *HTML report* button.

```python
html = exp.report(metrics=["resource.server.wait.mean"], adjust="holm")
print(len(html) > 0)
```

In Jupyter, results display as tables: `ExperimentResult`,
`SimulationResult`, `Comparison`, `SensitivityResult`, `MonteCarloResult`,
`FitReport`, `SelectionResult` and `OptimizationResult` all implement
`_repr_html_`.

## From the command line

```bash
simulsi visualize results/my-study --out plots                # distributions, convergence, comparison
simulsi visualize examples/queue.py --out plots --backend plotly   # one traced run
```

## Web dashboard

`simulsi ui` starts a local dashboard (React + TypeScript, served by a small
standard-library HTTP server bound to `127.0.0.1`). It is optional - nothing
in the Python package depends on it.

```bash
simulsi ui results/my-study results/other-study   # preload saved experiments
```

The dashboard lets you:

1. load saved experiments (from the command line or by dropping an
   `experiment.json` onto the page - it is read in the browser and sent only
   to the local SimulSI server);
2. view parameters, provenance and per-scenario summaries with CIs;
3. run built-in models and models loaded with `simulsi ui --model file.py`
   with your own parameters, scenarios, replications and seed;
4. compare scenarios against a baseline (differences, CIs, multiple-comparison
   adjustment);
5. inspect any metric's per-replication distribution and convergence;
6. **explore** a model: grid sweeps (full factorial, opened in a *Response*
   tab that plots a metric against a parameter with CIs, one line per value
   of another parameter), Morris sensitivity screening, and Monte Carlo over
   uncertain inputs with output distributions and Spearman correlations;
7. trace a single run and **replay** it: a time slider (or Play) shows busy
   and idle units, waiting entities, queue and container levels, every level
   the model records (for example the epidemic's S/E/I/R counts) and the
   latest events at each moment, next to the full event log. Tick *Compare
   two parameter sets* (or pick two presets) to run both with the same seed
   and replay them side by side on one clock, with a metric-by-metric diff;
8. export results as JSON, CSV or an HTML report.

Runs, grid sweeps and analyses run as background jobs, so the page stays
responsive and shows progress.

Building it from source needs Node 20+: `cd web/frontend && npm ci && npm run build`.
Wheels built in CI include the compiled dashboard.
