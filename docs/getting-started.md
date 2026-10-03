# Getting started

## Installation

SimulSI needs Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install "simulsi[viz]"       # core + matplotlib, from PyPI
```

| Extra | Adds |
|---|---|
| *(none)* | engine, experiments, statistics, CLI (NumPy, SciPy, pydantic, PyYAML) |
| `viz` | matplotlib plots |
| `plotly` | interactive Plotly figures |
| `pandas` | `ExperimentResult.to_dataframe()` |
| `parquet` | Parquet export (pandas + pyarrow) |
| `all` | all of the above |
| `dev` | test, lint and type-check tooling |

Check the install:

```bash
simulsi --version
simulsi run builtin:mmc --duration 1000
```

## Your first simulation

A simulation has a clock, an event queue and components. *Processes* are
Python generators: each `yield` waits for something - a duration, a resource,
an item in a queue - and the engine resumes the generator when it is ready.

```python
from simulsi import Simulation

sim = Simulation(seed=42)
server = sim.resource("server", capacity=1)


def customer(sim, name, arrive, service):
    yield arrive                      # hold until the arrival time
    req = yield sim.request(server)   # wait for the server
    print(f"{sim.now:5.1f}  {name} starts after waiting {req.waiting_time:.1f}")
    yield service                     # being served
    sim.release(req)


for i, (arrive, service) in enumerate([(0, 4), (1, 3), (2, 2)]):
    sim.process(customer(sim, f"c{i}", arrive, service))

result = sim.run(until=20)
print(result.metrics["resource.server.utilization"])   # 9 busy time units / 20 = 0.45
```

`result.metrics` is a flat dictionary of everything SimulSI measured
automatically (utilization, waiting times, queue lengths, throughput, ...)
plus anything you recorded yourself.

## Randomness, the reproducible way

Never use `random` or `numpy.random` globals in a model. Ask the simulation
for a named stream; it is derived from the seed and the name only.

```python
from simulsi import Simulation
from simulsi.randomness import Exponential

sim = Simulation(seed=7)
arrivals = sim.stream("arrivals")
gap = Exponential(rate=0.5)
print([round(gap.sample(arrivals), 3) for _ in range(3)])   # same numbers every run
```

## Your first model and experiment

Wrap the setup in a *model* - a build function plus parameters and a run
length - so it can be replicated, compared and analysed:

```python
from simulsi import Experiment, Parameter, Scenario, Simulation, model
from simulsi.randomness import Exponential


@model(
    duration=1000.0,
    warmup=100.0,
    parameters=[
        Parameter("arrival_rate", 0.8, "float", low=0.0),
        Parameter("servers", 1, "int", low=1),
    ],
)
def clinic(sim: Simulation, p):
    desk = sim.resource("desk", capacity=p.servers)
    arrivals, service = sim.stream("arrivals"), sim.stream("service")

    def patient(sim):
        yield from sim.use(desk, Exponential(mean=1.0).sample(service))

    def source(sim):
        while True:
            yield Exponential(rate=p.arrival_rate).sample(arrivals)
            sim.process(patient(sim))

    sim.process(source(sim))


experiment = Experiment(
    clinic,
    [Scenario("baseline"), Scenario("two_desks", {"servers": 2})],
    replications=10,
    seed=1,
)
result = experiment.run()
print(result.format_summary(["resource.desk.wait.mean", "resource.desk.utilization"]))
print(result.compare("baseline", metrics=["resource.desk.wait.mean"]).format())
```

Each replication gets an independent seed derived from the experiment seed;
both scenarios use the same replication seeds, so the comparison is paired.
Save everything (results + provenance) with `result.save("results/clinic")`.

## From the command line

```bash
simulsi init my-study && cd my-study
simulsi validate experiment.yaml
simulsi experiment experiment.yaml
simulsi analyze results/service-desk --precision 0.05
simulsi visualize results/service-desk --out plots
```

Next: [Core concepts](concepts.md) and [Experiments and analysis](experiments.md).
