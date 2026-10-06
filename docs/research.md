# Research workflow

SimulSI is designed so that a simulation study can be repeated by someone
else - or by you in six months - and give the same numbers.

## A reproducible study, step by step

1. **Version the model.** Put the model in a git repository and set
   `Model(version="...")`. Bump the version whenever logic changes.
2. **Validate before you trust.** Run `validate_model` (or `simulsi
   validate`) for each scenario. Then check the model against something
   known: a closed-form case (as the M/M/c tests do with Erlang C), historical
   data, or a hand-worked trace (`trace=True`).
3. **Choose warm-up and run length.** For steady-state questions, run
   `suggest_warmup(model, params)` (or `simulsi warmup model.py`): it
   averages a time series over replications and applies MSER-5. Check the
   averaged series too; if the result is flagged unreliable, the run is too
   short to reach steady state. For terminating systems (a working day, a week of
   operations) the run length is the real horizon and no warm-up is needed.
4. **Pilot, then size the experiment.** Run ~10 replications and ask
   `result.replication_advice(metric, relative_precision=0.05)` how many you
   need. Re-check after running them.
5. **Compare with common random numbers.** Keep
   `common_random_numbers=True` so scenario differences are estimated with
   paired intervals.
6. **Record everything.** `result.save(dir)` writes the records together with
   seeds, resolved parameters, model version, git commit (and whether the
   tree was dirty), timestamp, runtime and environment.

```python
from simulsi import Experiment, ExperimentResult, Scenario
from simulsi.models import mmc

model = mmc.with_options(duration=2_000, warmup=200)
pilot = Experiment(model, Scenario("baseline", {"arrival_rate": 0.9}), replications=10, seed=2026).run()
advice = pilot.replication_advice("resource.server.wait.mean", relative_precision=0.10)
print(advice.relative_half_width, advice.required_n)

study = Experiment(model, Scenario("baseline", {"arrival_rate": 0.9}),
                   replications=max(10, advice.required_n or 10), seed=2026).run()
study.save("results/mm1-study")
again = ExperimentResult.load("results/mm1-study")
md = again.metadata
print(md.experiment_id, md.model_name, md.model_version, md.seed, md.git, md.environment["python"])
rec = again.records[0]
replay = model.simulate(rec.parameters, seed=rec.seed)       # any replication, re-run alone
assert replay.metrics["resource.server.wait.mean"] == rec.metrics["resource.server.wait.mean"]
```

## What the statistics do and do not tell you

* **Replications are independent.** Different replications use independent
  seeds, so t-intervals across replications are valid (with the usual
  normal-approximation caveats for very skewed metrics with few replications -
  use more replications or `bootstrap_ci`).
* **Observations within one run are not independent.** Do not compute a
  t-interval over all customers' waiting times in one run; use replication
  means, or `batch_means` for a single long run.
* **Intervals cover sampling error only.** They measure how precisely the
  model's expected output has been estimated, not how well the model matches
  reality. Validation is a separate, domain-specific task.
* **Comparisons are not causal claims about the world.** A significant
  difference means the *model* responds to the change.
* **Many comparisons inflate false positives.** With 20 metrics at 95%,
  expect about one spurious "significant" difference. Pre-register the
  metrics that matter, or adjust: `compare(..., adjust="holm")`
  (`simulsi analyze --adjust holm`).
* **Optimising over noise flatters the winner.** Re-evaluate the selected
  design with fresh seeds before reporting it.

## Design of experiments

* `grid(...)` / `Experiment.grid(...)` - full factorial designs.
* `one_at_a_time(...)` scenarios and the `analysis.one_at_a_time` method.
* `monte_carlo(..., sampling="lhs")` - Latin hypercube designs over
  uncertain inputs, followed by `correlation_sensitivity` (Spearman, SRC
  with R^2).
* `finite_difference(...)` - local gradients with common random numbers.

## Reporting checklist

- model version and git commit, SimulSI version (`result.metadata`)
- run length, warm-up and how they were chosen
- number of replications and why it is enough
- confidence level and interval method (paired-t / Welch / bootstrap)
- validation performed and its limits
- all scenario definitions (`result.metadata.scenarios`)

## Reading list

* A. M. Law, *Simulation Modeling and Analysis*, 5th ed., McGraw-Hill, 2015 -
  output analysis, replication sizing (the approximation used by
  `required_replications`), common random numbers.
* J. Banks et al., *Discrete-Event System Simulation*, 5th ed., Pearson, 2010.
* B. L. Nelson, *Foundations and Methods of Stochastic Simulation*, 2nd ed., Springer, 2021.
* A. Saltelli et al., *Global Sensitivity Analysis: The Primer*, Wiley, 2008.
