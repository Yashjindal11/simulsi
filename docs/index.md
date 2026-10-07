# SimulSI

**Simulation Intelligence** - *Model the system. Simulate the future.*

SimulSI is a general-purpose, reproducible discrete-event simulation and
scenario experimentation framework for Python: model queues, resources,
entities and disruptions as processes, then run reproducible experiments with
confidence intervals, paired scenario comparisons, Monte Carlo and sensitivity
analysis.

```bash
pip install simulsi            # add [viz] for plots, [all] for pandas/Parquet/Plotly
simulsi init my-study && cd my-study && simulsi experiment experiment.yaml
```

New here? Start with [Getting started](getting-started.md), then
[Core concepts](concepts.md) and [Experiments and analysis](experiments.md).
Every public class and function is documented in the API reference.

## Topics

| Topic | Page |
|---|---|
| Why SimulSI? | [README](https://github.com/Yashjindal11/simulsi/blob/main/README.md#why-simulsi) |
| Installation | [Getting started](getting-started.md#installation) |
| Quickstart | [Getting started](getting-started.md#your-first-simulation) |
| Core concepts | [Concepts](concepts.md) |
| Events | [Concepts: events](concepts.md#events) |
| Entities | [Concepts: entities](concepts.md#entities) |
| Resources | [Concepts: resources](concepts.md#resources) |
| Queues | [Concepts: queues](concepts.md#queues) |
| Processes | [Concepts: processes](concepts.md#processes) |
| Randomness | [Concepts: randomness](concepts.md#randomness) |
| Metrics | [Concepts: metrics](concepts.md#metrics) |
| Failures and disruptions | [Concepts: failures](concepts.md#failures-and-disruptions) |
| Snapshots, logs, graphs | [Concepts: introspection](concepts.md#state-snapshots-event-logs-and-graphs) |
| Model validation | [Concepts: validation](concepts.md#model-validation) |
| Models and scenarios | [Experiments](experiments.md#models) |
| Experiments | [Experiments](experiments.md#experiments) |
| Statistics | [Experiments: statistics](experiments.md#statistics) |
| Scenario comparison | [Experiments: comparison](experiments.md#comparing-scenarios) |
| Monte Carlo | [Experiments: Monte Carlo](experiments.md#monte-carlo) |
| Sensitivity analysis | [Experiments: sensitivity](experiments.md#sensitivity-analysis) |
| Cost models | [Experiments: cost](experiments.md#cost-models) |
| Optimisation | [Experiments: optimisation and surrogates](experiments.md#optimisation-and-surrogates) |
| Visualization & dashboard | [Visualization](visualization.md) |
| CLI & YAML configuration | [CLI](cli.md) |
| Architecture | [Architecture](architecture.md) |
| Extending SimulSI | [Extending](extending.md) |
| Research workflow | [Research](research.md) |
| Examples | [Examples](examples.md) |
| FAQ & comparison with other tools | [FAQ](faq.md) |
| Benchmarks | [Benchmarks](https://github.com/Yashjindal11/simulsi/blob/main/benchmarks/README.md) |
| Contributing | [CONTRIBUTING](https://github.com/Yashjindal11/simulsi/blob/main/CONTRIBUTING.md) |
