# CLI and configuration

```text
simulsi init [DIR] [--force]
simulsi validate TARGET [-p KEY=VALUE ...] [--smoke T]
simulsi run TARGET [-p KEY=VALUE ...] [--seed N] [--duration T] [--warmup T] [-m METRIC ...] [--trace FILE] [--json]
simulsi experiment CONFIG [-r N] [-w N] [--seed N] [-o DIR] [--checkpoint FILE] [--adjust METHOD] [-q]
simulsi analyze RESULTS [-m METRIC ...] [--baseline NAME] [--confidence C] [--precision P] [--adjust METHOD] [--json]
simulsi benchmark [--sizes N ...] [--workers N ...] [--replications N] [--no-memory] [--json]
simulsi visualize TARGET [-o DIR] [-m METRIC ...] [--backend matplotlib|plotly] [--format png|svg|pdf]
simulsi ui [RESULTS ...] [--model REF ...] [--host H] [--port P] [--no-browser]
```

`python -m simulsi ...` works too. Exit codes: `0` success, `1` runtime
failure (e.g. failed replications), `2` invalid input or configuration.

## Model references

Wherever a model is expected you can write:

| Reference | Meaning |
|---|---|
| `builtin:mmc` | a model shipped with SimulSI |
| `examples/queue.py` | the module's `model` attribute, or its only `Model` |
| `examples/queue.py:queue` | a named attribute of a file |
| `mypackage.models:clinic` | an attribute of an importable module |
| `experiment.yaml` | (`run`, `validate`) the model and base parameters of a config |

Parameters given with `-p key=value` are parsed as YAML scalars, so
`-p servers=3 -p rate=1.5 -p enabled=true` gives an int, a float and a bool.

## Commands

* **init** writes `model.py`, `experiment.yaml` and `README.md`.
* **validate** checks the configuration schema, parameter values and runs a
  short smoke simulation per scenario (see [model validation](concepts.md#model-validation)).
* **run** runs one replication and prints the metrics (`--json` for
  everything, `--trace log.csv|json|parquet` for the event log).
* **experiment** runs every scenario x replication, prints a summary with
  CIs and a comparison with the baseline, applies the cost model and saves
  `experiment.json`, `replications.csv`, `summary.csv` (and Parquet if asked).
* **analyze** reloads saved results: summary, comparison, provenance, and
  with `--precision 0.05` how many replications each metric needs.
* **benchmark** measures engine throughput (10k / 100k / 1M events),
  process-based model throughput and parallel experiment speed on your
  machine, with tracemalloc peak memory.
* **visualize** plots saved results (distributions, convergence, comparison)
  or runs a model once with tracing and plots queues, utilization, timeline
  and trajectories.
* **ui** starts the local web dashboard ([details](visualization.md#web-dashboard)).

## Experiment configuration

```yaml
model: model.py:service_desk     # required; see model references
name: service-desk-staffing      # optional, defaults to the model name
description: Staffing study for the downtown branch

simulation:
  seed: 42                       # experiment seed (replication seeds derive from it)
  duration: 480                  # overrides the model's duration
  warmup: 30                     # overrides the model's warm-up

experiment:
  replications: 30               # 1 .. 1,000,000
  workers: 4                     # worker processes (1 = serial)
  common_random_numbers: true    # same seeds for every scenario (paired comparisons)
  confidence: 0.95
  on_error: raise                # or "record" to keep going and store errors
  multiple_comparisons: holm     # none (default), bonferroni, holm or bh

parameters:                      # base values for every scenario
  arrival_rate: 0.9

scenarios:                       # each derives from the base parameters
  - name: high_demand
    description: Saturday peak
    parameters: {arrival_rate: 1.3}
  - name: extra_server
    parameters: {servers: 3}

grid:                            # optional full factorial instead of a single baseline
  servers: [2, 3, 4]
  arrival_rate: [0.8, 1.0, 1.2]

baseline: baseline               # scenario used for comparisons
metrics:                         # what to report (default: everything)
  - resource.server.wait.mean
  - resource.server.utilization

cost:                            # optional cost model, applied per replication
  terms:
    - {name: staff, kind: cost, metric: resource.server.capacity_time, rate: 0.5}
    - {name: waiting, kind: cost, metric: resource.server.wait.total, rate: 0.2}
    - {name: sla, kind: cost, metric: resource.server.wait.mean, above: 5, fixed: 100}
    - {name: fees, kind: revenue, metric: resource.server.releases, rate: 1.5}

output:
  directory: results/service-desk   # relative to, and kept inside, the config's directory (use -o for elsewhere)
  formats: [json, csv, parquet]
```

Distribution-valued parameters use specs such as
`{distribution: lognormal, mu: 1.2, sigma: 0.4}` (registered kinds:
`constant`, `uniform`, `normal`, `exponential`, `poisson`, `binomial`,
`gamma`, `lognormal`, `triangular`, `empirical`, `categorical`).

### Safety

* YAML is read with `yaml.safe_load`; Python-specific tags are rejected.
* Unknown keys anywhere are errors (typos never pass silently).
* Configuration is never executed. The `model:` reference is the only link to
  code, and file references must resolve inside the configuration's
  directory unless you pass `--allow-outside`.
* Results are written only where you ask (`output.directory` may not escape
  the configuration's directory); nothing is uploaded.
