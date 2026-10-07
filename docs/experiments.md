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

## Variance reduction

Two standard techniques buy narrower intervals for the same number of runs.

**Antithetic variates** run replications in pairs: the second run of each
pair draws `1 - u` wherever the first drew `u`, so a run with unusually short
inter-arrival times is paired with one with unusually long ones.

```python
from simulsi import Experiment, Scenario
from simulsi.models import mmc

exp = Experiment(mmc.with_options(duration=500, warmup=50),
                 Scenario("baseline", {"arrival_rate": 0.7}),
                 replications=20, seed=5, antithetic=True)
anti = exp.run()
print(anti.summary(["resource.server.utilization"])[0]["half_width"])
print(len(anti.values("resource.server.utilization")))      # 10 pair means
print(len(anti.raw_values("resource.server.utilization")))  # 20 runs
```

* `replications` must be even. Pair `k` shares seed
  `replication_seed(scenario, k)`; summaries, comparisons and `run_until` use
  the *pair means* as the independent observations.
* Antithetic runs sample by inversion (`RandomStream(seed, mode="inverse")`
  and `"antithetic"`), so each uniform maps to exactly one variate. Draws
  differ from native runs with the same seed, but the distributions are the
  same.
* It helps when outputs move monotonically with the inputs (utilization,
  throughput, mean waits); it can hurt for non-monotone responses. Check the
  pair correlation with `raw_values`.

**Control variates** adjust an output using a quantity whose true mean is
known, such as the observed mean service time:

```py
est = result.control_variate("resource.server.wait.mean",
                             {"service.mean": 1.0})    # true mean service time
print(est.mean, est.ci_low, est.ci_high, est.variance_reduction)
```

The estimator regresses the output on the controls across replications
(`simulsi.statistics.control_variate`), and the interval accounts for the
estimated coefficients (n - q - 1 degrees of freedom). The control must be
recorded by the model, for example with `sim.metrics.observe("service", s)`.

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

**Morris screening** is the cheap first pass when there are many inputs: `r`
random one-at-a-time trajectories cost `r * (d + 1)` runs.

```python
from simulsi.analysis import morris_screening

screen = morris_screening(model, {"arrival_rate": (1.0, 1.8), "service_rate": (0.9, 1.1),
                                  "servers": (2, 4)},
                          r=8, outputs=outputs, seed=2)
print(screen.format())
```

`morris-mu_star` (mean absolute effect, inputs scaled to [0, 1]) ranks
importance; a `morris-sigma` comparable to `mu_star` signals non-linearity or
interactions. Drop the inputs with negligible `mu_star`, then run Sobol on
the rest.

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

## Choosing the best scenario

With several alternatives, `select_best` (procedure KN, Kim and Nelson 2001)
keeps adding replications only to the scenarios still in contention and
stops once one is best with the requested confidence:

```python
from simulsi.analysis import select_best

choice = select_best(
    model,
    {f"servers={c}": {"servers": c, "arrival_rate": 1.5} for c in (2, 3, 4)},
    "resource.server.wait.mean",
    indifference=0.05,        # differences below this are treated as ties
    confidence=0.95,
)
print(choice.format())
```

The guarantee: with probability at least `confidence`, the selected scenario
is the best or within `indifference` of it. Clearly worse scenarios drop out
after a few runs, so this is usually much cheaper than running every scenario
to the same precision. If `max_replications` is reached first, the result
says so (`converged=False`).

## Optimisation and surrogates

`optimize` searches a parameter box and then confirms the winner:

```python
from simulsi.optimization import optimize

best = optimize(
    model, None, {"servers": (1, 6)},
    fixed={"arrival_rate": 1.5}, method="bayes", budget=6, replications=3,
    transform=lambda m: costs.calculate(m).total_cost,
)
print(best.format())
```

* `method="grid"` tries every combination (integers: every value; floats:
  `grid_levels` points), `"random"` a Latin hypercube of `budget` points, and
  `"bayes"` fits a Gaussian process after a small initial design and picks
  each next point by expected improvement.
* Every candidate is the mean of `replications` runs with shared seeds.
* With a `metric` and `indifference=...`, the best `confirm_top` candidates
  are re-run with new seeds and compared by `select_best`, so the reported
  optimum is not just the luckiest estimate.

A surrogate (metamodel) approximates a model from a few dozen runs:

```python
from simulsi.optimization import fit_surrogate

surface = fit_surrogate(model, {"arrival_rate": (1.0, 1.8), "servers": (2, 4)},
                        ["resource.server.wait.mean"], n=20)
print(surface.accuracy())        # leave-one-out R^2; below ~0.8, add runs
mean, sd = surface.predict({"arrival_rate": 1.4, "servers": 3}, return_std=True)
print(mean, sd)
```

Predictions are refused outside the fitted ranges. Treat them as a guide to
where to run real experiments, not as results.

## Trade-offs: Pareto fronts

When objectives conflict (cost vs service, on-time rate vs passenger
delay) there is no single best design, only a front of designs where
improving one objective worsens another. `pareto_search` evaluates
candidates with common random numbers and marks that front:

```python
from simulsi.models import mmc
from simulsi.optimization import pareto_search

front = pareto_search(
    mmc.with_options(duration=2_000, warmup=200),
    {"resource.server.wait.mean": "min", "resource.server.utilization": "max"},
    grid={"servers": [1, 2, 3, 4]},
    fixed={"arrival_rate": 0.8},
    replications=3,
)
print(front.format(all_designs=True))
```

Candidates come from `grid=` (full factorial), `ranges=` (a Latin
hypercube of `budget` points) or `scenarios=` (named parameter sets).
`front.plot()` draws two objectives with the front highlighted. When a
dominated design lies within the confidence intervals of a front design, the
result says so: more replications may change the front. From the command
line:

```bash
simulsi pareto builtin:disruption_recovery -o cost:min -o otp:max \
    --vary policy=delay,cancel,spares --vary spares=1,2,4 --plot front.png
```

## Plugging in other optimisers

`Objective` turns a model into a function any optimiser can call
(scipy.optimize, OR-Tools, evolutionary or Bayesian libraries):

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
