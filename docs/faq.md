# FAQ

## How does SimulSI compare with other tools?

SimulSI is a young project. Established tools have years of use and
features SimulSI lacks; nothing here claims SimulSI is better. The question
is which fits your workflow.

| | SimulSI | SimPy | Salabim | Commercial DES suites (AnyLogic, Arena, Simio, ...) |
|---|---|---|---|---|
| Modelling style | generator processes, events, resources, queues | generator processes, events, resources, stores | components/processes, queues, states, built-in animation | graphical flowcharts + code |
| Experiments, replications, seeds | built in (`Experiment`, CRN, provenance, checkpoints, parallel) | user code | user code / some helpers | built in |
| Output statistics (CIs, replication advice, comparison) | built in | user code | monitors with statistics | built in |
| Monte Carlo & sensitivity | built in (random/LHS, OAT, FD, correlation/SRC) | user code | user code | varies / add-ons |
| Animation | no (plots + dashboard only) | no | yes | yes |
| Maturity | alpha | mature, widely used | mature | mature, commercial |
| Licence | MIT | MIT | MIT | proprietary |

**SimPy** is the reference Python discrete-event library: small, stable and
well documented. SimulSI's process style will look familiar to SimPy users
(it was designed independently, with a few differences: `yield 5` holds
for 5 time units, requests have built-in `patience`, resources record
statistics automatically, interrupts withdraw pending requests). If you
only need an engine and like assembling the analysis yourself, SimPy is an
excellent choice.

**Salabim** offers rich built-in monitoring and animation. Choose it if
visual animation of the model matters.

**Commercial suites** give graphical modelling, animation, optimisation
and support for large industrial models.

SimulSI's niche is the **experimentation layer**: reproducible seeding
designed for common random numbers, scenarios and grids, paired comparisons,
replication sizing, Monte Carlo, sensitivity analysis, cost models,
provenance, validated YAML configuration, a CLI and an optional dashboard,
all in one MIT-licensed Python package with no services or keys.

## Does it use AI or call any service?

No. There is no LLM, no API key, no telemetry and no network access.

## How fast is it?

It is a pure-Python engine. On the development machine (an Apple silicon
laptop, Python 3.12) it processes roughly 440k simple events per second, and
about 240k events per second for process-based queueing models. Measured
numbers, and how to reproduce them,
are in [benchmarks/README.md](https://github.com/Yashjindal11/simulsi/blob/main/benchmarks/README.md). For many replications,
use `workers=N`.

## Why are my results different from yesterday?

They should not be with the same seed, model and parameters. Check:

* the seed (`result.metadata.seed`, `record.seed`);
* the model version and git commit recorded in `result.metadata`;
* global randomness inside the model (`random.random()`, `np.random.*`) -
  use `sim.stream("name")` instead;
* iteration over unordered collections (`set`) that affects scheduling order;
* wall-clock time used inside the model.

## Why does adding a resource change my arrivals?

It should not, if every random quantity comes from a *named* stream
(`sim.stream("arrivals")`). If you draw everything from `sim.rng`, any extra
draw shifts all later numbers. Named streams are what make common random
numbers effective.

## Should I use one long run or many replications?

Replications are simpler and give valid confidence intervals directly. For
steady-state metrics of slowly mixing systems, one long run with
`batch_means` avoids repeating the warm-up. Check the reported lag-1
autocorrelation.

## How many replications do I need?

Run a pilot (~10), then `result.replication_advice(metric,
relative_precision=0.05)` or `simulsi analyze results --precision 0.05`.

## Can I pause and resume a simulation?

You can call `run(until=...)` repeatedly, and save a running simulation with
`sim.save_checkpoint(path)` and restore it later with
`Simulation.load_checkpoint(path)` (restore replays the run deterministically
and verifies the result). Experiments can be checkpointed per replication.
See [Architecture: checkpointing](architecture.md#checkpointing).

## Can a resource preempt a lower-priority user?

Yes: `sim.resource(..., discipline="priority", preemptive=True)`. The evicted
process receives an `Interrupt` with a `Preempted` cause and can re-queue its
remaining work; see [Concepts: preemption](concepts.md#preemption).

## Does it work on Windows?

Yes. CI runs the test suite on Linux, macOS and Windows. Parallel experiments
use the `spawn` start method on Windows and macOS, so scripts must guard
their entry point with `if __name__ == "__main__":`.
