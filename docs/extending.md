# Extending SimulSI

SimulSI's extension points are plain Python objects - no plugin registry is
needed except for distributions used in configuration files.

## Custom distributions

Subclass `Distribution` and implement `sample`. Give it a `kind` to make it
usable from YAML (`{"distribution": "<kind>", ...}`); add `mean`/`variance`
if they are known (statistical tests use them).

```python
import math
from dataclasses import dataclass
from typing import ClassVar

from simulsi.randomness import Distribution, RandomStream, from_spec


@dataclass(frozen=True)
class Weibull(Distribution[float]):
    shape: float
    scale: float = 1.0
    kind: ClassVar[str] = "weibull"

    def __post_init__(self):
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("weibull shape and scale must be > 0")

    def sample(self, stream: RandomStream) -> float:
        return self.scale * float(stream.generator.weibull(self.shape))

    def sample_n(self, stream, n):
        return self.scale * stream.generator.weibull(self.shape, n)

    @property
    def mean(self) -> float:
        return self.scale * math.gamma(1 + 1 / self.shape)


d = from_spec({"distribution": "weibull", "shape": 1.5, "scale": 100})
print(d.sample(RandomStream(1)), d.mean)
```

For one-offs, `Custom(lambda stream: ...)` wraps any function (not usable in
configuration).

## Custom queue disciplines

Any key function works as a discipline; ties fall back to arrival order.

```python
from simulsi import Simulation

sim = Simulation(seed=1)
# shortest-processing-time-first: requests carry the job in `entity`
machine = sim.resource("machine", 1, discipline=lambda req: req.entity["work"])
buffer = sim.queue("jobs", discipline=lambda job: job["due"])   # earliest due date
```

## Custom events

Subclass `Event` and override `execute`; call `super().__init__` with the
event type and payload.

```python
from simulsi import Event, Simulation


class Restock(Event):
    def __init__(self, units):
        super().__init__("restock", {"units": units})

    def execute(self, sim):
        sim.metrics.increment("restocked", self.payload["units"])
        sim.schedule(Restock(self.payload["units"]), delay=24)


sim = Simulation(seed=1)
sim.schedule(Restock(100), time=0)
print(sim.run(until=72).metrics["restocked"])   # 4 restocks at t = 0, 24, 48, 72
```

## Custom metrics

Anything you can compute can be a metric: `sim.metrics.observe/record/
increment/set`, or a finish hook that derives values from collectors:

```python
from simulsi.models import mmc


def build(sim, p):
    mmc.build(sim, p)
    sim.on_finish(lambda s: s.metrics.set("server.idle_fraction",
                                          1 - s.resources["server"].utilization))


from simulsi import Model

extended = Model(build, name="mmc_idle", duration=500.0, parameters=list(mmc.parameters.values()))
print(extended.simulate(seed=1).metrics["server.idle_fraction"])
```

After an experiment, `result.derive(fn)` adds metrics computed from each
replication's metrics (that is how cost models plug in).

## Composing models

A build function can call other build functions, wrap them, or add
disruptions - models are just Python:

```python
from simulsi import Model
from simulsi.processes import FailureProcess
from simulsi.randomness import Exponential


def with_failures(sim, p):
    mmc.build(sim, p)
    FailureProcess(sim, sim.resources["server"], Exponential(mean=p.mtbf), p.repair)


unreliable = Model(
    with_failures, name="mmc_unreliable", duration=2_000.0,
    parameters={**{k: v for k, v in mmc.parameters.items()}, "mtbf": 200.0, "repair": 10.0},
)
print(unreliable.simulate(seed=1).metrics["resource.server.availability"])
```

## Connecting optimisers

`simulsi.optimization.Objective` exposes `__call__(x) -> float`, `bounds` and
`history`, which is what scipy.optimize, scikit-optimize, Optuna, DEAP or an
OR-Tools callback expect. See [experiments.md](experiments.md#optimisation-interface).

## New analysis methods

`SensitivityResult` holds `SensitivityRow(parameter, output, method, value,
ci_low, ci_high, setting, detail)` rows; a new method (e.g. Sobol indices) only
needs to produce rows to get ranking, formatting and tornado plots for free.
Statistical functions in `simulsi.statistics` take plain arrays - add new ones
there with tests against a reference implementation.
