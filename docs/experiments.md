# Experiments and analysis

All examples on this page use the built-in M/M/c queue (`simulsi.models.mmc`),
whose steady-state behaviour is known exactly (Erlang C), so you can see the
statistics doing their job.

## Models

A `Model` bundles a build function `build(sim, params)` with a parameter
schema, a run length and an optional warm-up:

```python
from simulsi import Model, Parameter, model
from simulsi.models import mmc

print(mmc.describe()["parameters"][0])
short = mmc.with_options(duration=2_000, warmup=200)
r = short.simulate({"servers": 2, "arrival_rate": 1.5}, seed=1)
print(r.metrics["resource.server.wait.mean"])
```

* `Parameter(name, default, kind, low=, high=, choices=, description=, unit=)`;
  `kind` is one of `int`, `float`, `bool`, `str`, `probability`,
  `distribution` (accepts `{"distribution": ...}` specs) or `any`.
  `Model(parameters={"servers": 2})` infers kinds from defaults.
* Unknown parameter names are errors (`strict=True`), which catches typos in
  scenarios and configuration files.
* `model.simulate(params, seed=...)` runs one replication;
  `model.create(...)` builds a simulation without running it (for stepping
  and inspection); `model.evaluate(params, metric=..., replications=...)`
  returns a mean with common random numbers.

## Scenarios

```python
from simulsi import Scenario, grid
from simulsi.scenarios import one_at_a_time

baseline = Scenario("baseline", {"arrival_rate": 1.5, "servers": 2})
scenarios = [
    baseline,
    baseline.derive("high_demand", arrival_rate=1.8),
    baseline.derive("capacity_expansion", servers=3),
    baseline.scale("demand_plus_20pct", arrival_rate=1.2),   # numeric parameters only
]
matrix = grid({"servers": [2, 3, 4], "arrival_rate": [1.0, 1.5, 1.8]})
print(len(matrix), matrix[0].name)          # 9 'servers=2,arrival_rate=1'
print([s.name for s in one_at_a_time(baseline, {"servers": [1, 3]})])
```

Scenarios hold only overrides; the model supplies defaults. Failure,
disruption, weather or demand-shock scenarios are expressed through model
parameters (e.g. `mtbf_hours`, `weather_gates_closed`) - see the examples.

## Experiments

```python
from simulsi import Experiment, Scenario
from simulsi.models import mmc

model = mmc.with_options(duration=2_000, warmup=200)
exp = Experiment(
    model,
    [Scenario("baseline", {"arrival_rate": 1.5, "servers": 2}),
     Scenario("three_servers", {"arrival_rate": 1.5, "servers": 3})],
    replications=10,
    seed=42,
    workers=1,                    # >1 runs replications in worker processes
)
result = exp.run()
print(result.format_summary(["resource.server.wait.mean", "resource.server.utilization"]))
row = result.summary(["resource.server.wait.mean"])[0]
print(row["mean"], row["ci_low"], row["ci_high"])
```

Every record keeps scenario, replication, seed, resolved parameters, metrics,
runtime, events and warnings; `result.metadata` keeps experiment id, model
name and version, git commit, timestamp, seeds, environment and runtime.

* **Seeds** - replication `r` uses `derive_seed(seed, "replication", r)`,
  shared across scenarios (common random numbers). Set
  `common_random_numbers=False` for independent scenarios.
* **Parallelism** - `workers=N` uses a process pool. Results are identical to
  a serial run. The model must be importable (module-level build function);
  guard scripts with `if __name__ == "__main__":`. Starting workers costs
  around a second, so the pool is kept and reused by later experiments in the
  same Python process (call `simulsi.experiments.shutdown_workers()` after
  editing a model file in a notebook).
* **Run until precise** - `exp.run_until(0.05, ["resource.server.wait.mean"])`
  keeps adding replications (reusing earlier ones) until every listed metric
  in every scenario has a relative CI half-width of 5%, or `max_replications`
  is reached; `result.metadata.stopping` records what happened.
* **Checkpoints** - `exp.run(checkpoint="run.jsonl")` appends each finished
  replication; re-running with the same file skips completed work and refuses
  a file written by a different experiment definition.
* **Errors** - `on_error="record"` keeps going and stores the error message
  in the failed record (`result.errors`).
* **Export** - `result.save(dir, formats=("json", "csv", "parquet"))`,
  `ExperimentResult.load(dir)`, `result.to_dataframe()`.

## Statistics

```python
import numpy as np

from simulsi.statistics import (batch_means, bootstrap_ci, mean_ci, proportion_ci,
                                required_replications, summarize)

waits = result.values("resource.server.wait.mean", "baseline")
print(summarize(waits).to_dict()["half_width"])
print(mean_ci(waits, 0.95), bootstrap_ci(waits, np.median, seed=1))
print(proportion_ci(successes=18, n=20))                       # Wilson interval
advice = required_replications(waits, relative_precision=0.05)
print(advice.sufficient, advice.required_n, advice.note)
```

* `summarize` - n, mean, std, variance, min/max, median, quantiles, t-interval.
* `bootstrap_ci` - percentile bootstrap for any statistic (median, p95, ...).
* `required_replications` - estimates how many replications give a target
  relative half-width (Law's sequential approximation). It is an estimate:
  run that many and check again.
* `convergence` - running mean and half-width after each replication.
* `batch_means` - CI from **one long run** of a steady-state model. Within-run
  observations are autocorrelated; batching reduces that, and the result
  reports the lag-1 autocorrelation of batch means so you can check.
* `paired_difference` / `welch_difference` - CIs for differences.

All intervals assume independent replications (true for SimulSI
replications) and quantify *sampling error only*. They say nothing about
whether the model represents the real system.

## Comparing scenarios

```python
from simulsi import compare

cmp = compare(result, "baseline", metrics=["resource.server.wait.mean"])
print(cmp.format())
row = cmp.get("resource.server.wait.mean", "three_servers")
print(row.absolute_difference, row.percentage_difference, row.ci_low, row.ci_high, row.method)
```

Columns: metric, baseline, scenario, both means, absolute and percentage
difference, CI, p-value, method (`paired-t` with common random numbers, else
`welch-t`) and whether the CI excludes zero.

With many metrics and scenarios some "significant" differences will be
chance. Pass `adjust=` to correct for that:

```python
holm = compare(result, "baseline", adjust="holm")   # or "bonferroni", "bh"
print(holm.format())                                # adds a p_adj column
```

* `bonferroni` - every interval is widened to level `1 - alpha/m`, so all
  intervals hold *simultaneously*; conservative.
* `holm` - same family-wise guarantee, more powerful; adjusts p-values only.
* `bh` - Benjamini-Hochberg; controls the expected share of false
  discoveries rather than the chance of any.

With `holm`/`bh`, `significant` means `p_adjusted < 1 - confidence` and the
intervals are the ordinary per-comparison ones.

## Monte Carlo

```python
from simulsi import monte_carlo
from simulsi.randomness import Normal, Triangular, Uniform


def project_cost(labour_hours, rate, materials):
    return labour_hours * rate + materials


mc = monte_carlo(
    project_cost,
    {"labour_hours": Triangular(800, 1000, 1500), "rate": Uniform(60, 75), "materials": Normal(40_000, 5_000)},
    iterations=20_000,
    seed=1,
)
s = mc.summary()
print(s.mean, s.ci_low, s.ci_high, mc.percentile(95))
p, (lo, hi) = mc.probability("value", ">", 120_000)
print(f"P(cost > 120k) = {p:.3f} [{lo:.3f}, {hi:.3f}]")
```

* `vectorized=True` passes whole NumPy arrays to the function - orders of
  magnitude faster for formula-style models.
* A function with an `rng` argument receives its own per-iteration stream.
* `sampling="lhs"` uses Latin hypercube sampling (stratified; needs
  invertible distributions).
* `objective=` derives an `"objective"` output from each iteration.
* Passing a simulation `Model` runs one simulation per iteration: input
  uncertainty propagated through a stochastic model.

## Sensitivity analysis

```python
from simulsi import monte_carlo
from simulsi.analysis import correlation_sensitivity, finite_difference, one_at_a_time
from simulsi.randomness import Uniform

model = mmc.with_options(duration=1_000, warmup=100)
outputs = ["resource.server.wait.mean"]
base = {"arrival_rate": 1.5, "servers": 2}
print(one_at_a_time(model, {"servers": [3], "arrival_rate": [1.2, 1.8]}, outputs, base=base,
                    replications=5).format())
print(finite_difference(model, ["arrival_rate", "service_rate"], outputs, base=base,
                        replications=5).format())

mc = monte_carlo(model, {"arrival_rate": Uniform(1.0, 1.8), "service_rate": Uniform(0.9, 1.1)},
                 iterations=30, seed=3, fixed={"servers": 2})
print(correlation_sensitivity(mc, outputs).format())
```

* **One-at-a-time** - change one parameter, keep the rest at base; paired CI
  of the change and an elasticity.
* **Finite differences** - local derivatives with common random numbers;
  central by default.
* **Correlation** - Pearson, Spearman rank, and standardised regression
  coefficients with the regression R^2 (if R^2 is low, a linear summary is
  misleading). These screening methods miss interactions.
* **Sobol indices** - variance-based global sensitivity: the first-order index
  is the share of output variance explained by a parameter alone, and the
  total-effect index includes its interactions. Bootstrap CIs are included.
  Cost: `n * (d + 2)` evaluations.

```python
import numpy as np

from simulsi.analysis import sobol_indices


def ishigami(x1, x2, x3):            # a standard test function with known indices
    return np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3**4 * np.sin(x1)


u = Uniform(-np.pi, np.pi)
sobol = sobol_indices(ishigami, {"x1": u, "x2": u, "x3": u}, n=4096, vectorized=True, seed=1)
print(sobol.format())   # x3 has ~0 first-order but ~0.24 total effect: it acts only via interaction
```

`sobol_indices` also accepts a simulation `Model` (every design row uses
common random numbers); keep `n` modest, because each row is a simulation run.

## Cost models

```python
from simulsi.cost import CostModel

costs = (
    CostModel()
    .fixed("facility", 500)
    .resource("server", per_capacity_time=0.8)      # staffing, per server per time unit
    .waiting("server", per_time=0.3)                # customer waiting cost
    .penalty("sla", "resource.server.wait.mean", above=1.0, amount=200)
    .revenue("fees", "resource.server.releases", 2.0)
)
result.derive(costs.metrics)                        # adds cost.total, cost.profit, cost.<term>
print(result.format_summary(["cost.total", "cost.profit"]))
```

Cost models also load from configuration (`cost: {terms: [...]}`).

## Optimisation interface

SimulSI deliberately ships no optimiser. `Objective` turns a model into a
function any optimiser can call:

```python
from scipy.optimize import minimize_scalar

from simulsi.optimization import Objective

obj = Objective(model, None, ["servers"], fixed={"arrival_rate": 1.5}, replications=3,
                transform=lambda m: costs.calculate(m).total_cost)
for servers in (2, 3, 4):
    print(servers, obj([servers]))
best = minimize_scalar(lambda x: obj([x]), bounds=(2, 5), method="bounded", options={"maxiter": 6})
print(obj.best().parameters)
```

Every evaluation uses the same replication seeds, invalid parameter vectors
return `inf` instead of raising, integer parameters are rounded, results are
cached and `obj.history` keeps every evaluation. Confirm the chosen design
with a fresh experiment using a different seed - optimising over noise can
otherwise flatter the winner.
