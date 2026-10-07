# SimulSI

[![CI](https://github.com/Yashjindal11/simulsi/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashjindal11/simulsi/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/simulsi)](https://pypi.org/project/simulsi/)
[![Python](https://img.shields.io/pypi/pyversions/simulsi)](https://pypi.org/project/simulsi/)
[![Docs](https://github.com/Yashjindal11/simulsi/actions/workflows/docs.yml/badge.svg)](https://yashjindal11.github.io/simulsi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Simulation Intelligence** - *Model the system. Simulate the future.*

SimulSI is a general-purpose, reproducible simulation and scenario
experimentation framework for Python. You describe a system - customers and
tellers, patients and doctors, jobs and machines, aircraft and gates, requests
and servers - as discrete-event processes; SimulSI runs it, measures it, and
helps you answer *what if?* with replications, confidence intervals, scenario
comparisons, Monte Carlo and sensitivity analysis.

```python
from simulsi import Experiment, Scenario, Simulation, model
from simulsi.randomness import Exponential


@model(duration=480, parameters={"arrival_rate": 1.0, "tellers": 3})
def bank(sim: Simulation, p):
    tellers = sim.resource("teller", capacity=p.tellers)
    arrivals, service = sim.stream("arrivals"), sim.stream("service")

    def customer(sim):
        yield sim.request(tellers)                       # wait for a teller
        yield Exponential(mean=2.6).sample(service)      # being served
        sim.release(tellers)

    def source(sim):
        while True:
            yield Exponential(rate=p.arrival_rate).sample(arrivals)
            sim.process(customer(sim))

    sim.process(source(sim))


result = Experiment(
    bank,
    [Scenario("baseline"), Scenario("four_tellers", {"tellers": 4})],
    replications=30,
    seed=42,
).run()
print(result.format_summary(["resource.teller.wait.mean", "resource.teller.utilization"]))
print(result.compare("baseline", metrics=["resource.teller.wait.mean"]).format())
```

## Why SimulSI?

Discrete-event engines tell you what happened in *one* run. Decisions need
more: how sure are we, which scenario is better, which input matters most, and
can someone else reproduce this? SimulSI is built **experiment-first**:

| | |
|---|---|
| **Reproducible by construction** | Every run is determined by its seed. Named random streams (`sim.stream("arrivals")`) are derived from the seed with NumPy's `SeedSequence`, so adding a resource never shifts another stream. Serial and parallel runs give identical numbers. |
| **Scenario comparison** | Scenarios share replication seeds (*common random numbers*) and comparisons use paired confidence intervals, which are usually much tighter than comparing independent runs. |
| **Statistics built in** | t and bootstrap CIs, Wilson intervals for probabilities, "have I run enough replications?" advice, convergence curves, batch means for long runs. |
| **Monte Carlo & sensitivity** | Propagate input uncertainty (random or Latin hypercube sampling, vectorised or per simulation run). One-at-a-time, finite-difference, correlation and standardised-regression sensitivity. |
| **Provenance** | Each experiment records id, model name/version, git commit, timestamp, seeds, parameters, runtime and environment. Results export to JSON, CSV and Parquet. |
| **Model introspection** | `snapshot()` of live state, structured event logs, observed flow graphs (arrival -> queue -> service -> departure), entity state machines, and model validation with a smoke run (unreleased resources, possible deadlocks, zero capacities, unreached states). |
| **Decision support** | Cost/revenue models on top of metrics; an `Objective` adapter that hands models to scipy.optimize, OR-Tools, evolutionary or Bayesian optimisers without depending on any of them. |
| **Local-first** | No LLMs, API keys, telemetry or network access. Core dependencies: NumPy, SciPy, pydantic, PyYAML. Plotting, pandas, Parquet and the web dashboard are optional. |

SimulSI is not a replacement for mature engines such as SimPy or Salabim, or
for commercial packages; see
[how SimulSI differs](docs/faq.md#how-does-simulsi-compare-with-other-tools).

## Installation

Python 3.11+, from [PyPI](https://pypi.org/project/simulsi/):

```bash
pip install simulsi             # core (includes the CLI and the web dashboard)
pip install "simulsi[viz]"      # + matplotlib plots
pip install "simulsi[all]"      # + pandas, Parquet, matplotlib, Plotly
```

With Docker (dashboard and CLI, no Python needed; the image is published
for each release):

```bash
docker run --rm -p 127.0.0.1:8642:8642 ghcr.io/yashjindal11/simulsi        # dashboard on http://localhost:8642
docker run --rm -v "$PWD:/work" ghcr.io/yashjindal11/simulsi run model.py  # any CLI command
```

From source (for development):

```bash
git clone https://github.com/Yashjindal11/simulsi.git && cd simulsi
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # everything needed to run the tests
```

## Command line

```bash
simulsi init my-study                          # starter model.py + experiment.yaml
simulsi validate my-study/experiment.yaml      # schema check + model smoke run
simulsi run examples/queue.py -p servers=3     # one replication, metrics table
simulsi experiment my-study/experiment.yaml    # scenarios x replications, comparison, saved results
simulsi analyze my-study/results/service-desk --precision 0.05
simulsi visualize my-study/results/service-desk --out plots
simulsi models                                 # built-in models and their what-if presets
simulsi whatif builtin:epidemic --presets      # how outputs change across scenarios
simulsi benchmark                              # events/sec on this machine
simulsi ui my-study/results/service-desk       # local dashboard on 127.0.0.1
```

Experiments are described in YAML - data only, validated against a strict
schema:

```yaml
model: model.py:service_desk
simulation: {seed: 42, duration: 480}
experiment: {replications: 30, workers: 4}
parameters: {arrival_rate: 0.9}
scenarios:
  - {name: high_demand, parameters: {arrival_rate: 1.3}}
  - {name: extra_server, parameters: {servers: 3}}
```

## Built-in models

Nine ready-to-run models, each with *what-if presets* that show how the
system reacts when a parameter changes:

| Model | What changes when you turn the knobs |
|---|---|
| `builtin:airline` | schedule buffer, crews, gates, spares and storms vs on-time performance and delay propagation |
| `builtin:airport_turnaround` | banked vs depeaked schedules, crews and refuelling rules vs turnaround critical path and delays |
| `builtin:disruption_recovery` | after a hub storm: delay vs cancel vs spare aircraft, by cost, passenger delay and recovery time |
| `builtin:epidemic` | R0, vaccination, beds and lockdown policy vs attack rate, hospital overflow and deaths |
| `builtin:supply_chain` | lead times, forecasting and information sharing vs the bullwhip effect |
| `builtin:ride_hailing` | fleet size, rush hours and surge pricing vs service level and driver earnings |
| `builtin:cloud_autoscaling` | cold starts, scaling policy and traffic bursts vs SLO, errors and cost |
| `builtin:traffic_signal` | cycle length, green split, fixed vs actuated control (checked against Webster) |
| `builtin:mmc` | the M/M/c queue (checked against Erlang C) |

```bash
simulsi models airline                                      # parameters and presets
simulsi whatif builtin:airline --presets                    # compare every preset
simulsi whatif builtin:traffic_signal --vary cycle=30,60,90,120
```

See the [model gallery](docs/models.md) for what each one shows. Simple process
models can also be written as [YAML flowcharts](docs/flowcharts.md), no Python
needed: `simulsi whatif examples/flowcharts/clinic.yaml --presets`.

## Examples

| Example | Shows |
|---|---|
| [`queue.py`](examples/queue.py) | Minimal M/M/c model checked against Erlang C |
| [`bank_queue.py`](examples/bank_queue.py) | Reneging customers, staffing scenarios, replication advice |
| [`hospital.py`](examples/hospital.py) | Priority triage, several resources, service levels |
| [`warehouse.py`](examples/warehouse.py) | Time-varying demand, picking/packing buffers, backlog |
| [`manufacturing.py`](examples/manufacturing.py) | Breakdowns with a repair crew, blocking, cost model |
| [`transportation.py`](examples/transportation.py) | Shuttle loop, boarding capacity, left-behind passengers |
| [`aviation.py`](examples/aviation.py) | Synthetic airport gates, taxiway holds, tows, weather disruption |

```bash
python examples/hospital.py
python scripts/run_examples.py      # all of them, quick mode
```

## Documentation

The full documentation, including an API reference for every public class
and function, is at **https://yashjindal11.github.io/simulsi/**. The same
pages are in [`docs/`](docs):

- [Getting started](docs/getting-started.md) - installation, quickstart, first experiment
- [Core concepts](docs/concepts.md) - simulation, events, entities, resources, queues, processes, randomness, metrics, disruptions, validation, introspection
- [Experiments and analysis](docs/experiments.md) - scenarios, experiments, Monte Carlo, statistics, variance reduction, comparison, sensitivity, ranking and selection, cost, optimisation, surrogates
- [Visualization](docs/visualization.md) - plots, HTML reports, notebooks and the web dashboard
- [CLI and configuration](docs/cli.md)
- [Architecture](docs/architecture.md) - design, module map, performance, limitations
- [Extending SimulSI](docs/extending.md)
- [Research workflow](docs/research.md) - reproducibility, provenance, statistical practice
- [Flowchart models](docs/flowcharts.md) - build process models in YAML, no Python
- [Built-in model gallery](docs/models.md) - airline, epidemic, supply chain, ride hailing, cloud, traffic, M/M/c
- [Examples and notebooks](docs/examples.md) - walkthroughs of the domain examples and tutorial notebooks
- [Coming from SimPy](docs/simpy-migration.md) - concept map and a side-by-side port
- [FAQ](docs/faq.md) - including how SimulSI compares with other tools
- [Benchmarks](benchmarks/README.md) - measured numbers and how to reproduce them

## Status

SimulSI is alpha software (0.x) and the API may still change. It is tested on
Python 3.11-3.13 (Linux, macOS, Windows in CI) with unit, property-based
(Hypothesis), statistical, integration, CLI and performance tests - including
checks of simulated M/M/c queues against closed-form Erlang-C results.

## Contributing

Issues and pull requests are welcome - see [CONTRIBUTING.md](CONTRIBUTING.md)
and the [Code of Conduct](CODE_OF_CONDUCT.md). For security reports see
[SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Yash Jindal
