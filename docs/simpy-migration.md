# Coming from SimPy

SimulSI uses the same core idea as [SimPy](https://simpy.readthedocs.io/):
processes are Python generators that `yield` what they wait for. Most SimPy
models port line by line. What changes is around the model: parameters,
replications, statistics, comparisons and reports are built in, so the
bookkeeping you usually write yourself goes away.

## Concept map

| SimPy | SimulSI | Notes |
|---|---|---|
| `env = simpy.Environment()` | `sim = Simulation(seed=1)` | Every simulation has a seed; `sim.stream("name")` gives independent, reproducible random streams. |
| `env.now` | `sim.now` | |
| `env.process(gen)` | `sim.process(gen)` | Returns a `Process` you can `yield` (join) or `interrupt()`. |
| `yield env.timeout(d)` | `yield d` (or `yield sim.timeout(d, value)`) | A plain number is a delay. |
| `env.run(until=t)` | `sim.run(until=t)` | Returns a `SimulationResult` with all metrics. |
| `simpy.Resource(env, capacity=c)` | `sim.resource("name", c)` | Utilization, waits and queue length are recorded automatically. |
| `with res.request() as req: yield req` | `req = yield sim.request(res)` ... `res.release(req)`, or `yield from sim.use(res, duration)` | `sim.use` releases even if the process is interrupted. |
| `simpy.PriorityResource` | `sim.resource(..., discipline="priority")` + `sim.request(res, priority=p)` | Lower value = served first, FIFO among equals. |
| `simpy.PreemptiveResource` | `sim.resource(..., discipline="priority", preemptive=True)` | The preempted holder gets an `Interrupt` whose cause is `Preempted`. |
| `simpy.Store(env, capacity)` | `sim.queue("name", capacity=...)` | `yield q.put(x)`, `x = yield q.get()`; FIFO, LIFO, priority or a key function. |
| `simpy.FilterStore` | `q.get(filter=lambda item: ...)` | |
| `simpy.Container(env, capacity, init)` | `sim.container("name", capacity, init=...)` | Level statistics and stockouts are recorded. |
| `env.event()` / `event.succeed(v)` | `sim.signal()` / `signal.succeed(v)` | |
| `env.all_of([...])`, `env.any_of([...])` | `sim.all_of(...)`, `sim.any_of(...)` | |
| `proc.interrupt(cause)` / `simpy.Interrupt` | `proc.interrupt(cause)` / `simulsi.Interrupt` | |
| `res.request()` raced against `env.timeout(patience)` | `sim.request(res, patience=p)` | Reneging is counted (`resource.<name>.reneges`). |
| polling loops (`while not cond: yield env.timeout(1)`) | `yield sim.wait_until(lambda: cond)` | Checked after every event, no polling delay. |
| your own lists of waits, `numpy.mean`, ... | `sim.metrics.observe/record/increment` | Plus everything resources, queues and containers record. |
| your own loop over seeds | `Experiment(model, scenarios, replications=n)` | Confidence intervals, common random numbers, parallel workers, checkpoints. |

## The same model, both ways

A bank with two tellers. In SimPy:

```py
import random
import simpy

def customer(env, tellers, waits):
    arrive = env.now
    with tellers.request() as req:
        yield req
        waits.append(env.now - arrive)
        yield env.timeout(random.expovariate(1.0))

def source(env, tellers, waits):
    while True:
        yield env.timeout(random.expovariate(1.8))
        env.process(customer(env, tellers, waits))

random.seed(1)
env = simpy.Environment()
tellers = simpy.Resource(env, capacity=2)
waits = []
env.process(source(env, tellers, waits))
env.run(until=1_000)
print(sum(waits) / len(waits))
```

In SimulSI, as a reusable model with parameters:

```python
from simulsi import Model, Parameter, Simulation


def bank(sim: Simulation, p) -> None:
    tellers = sim.resource("teller", p.tellers)
    arrivals, service = sim.stream("arrivals"), sim.stream("service")

    def customer(sim):
        yield from sim.use(tellers, service.exponential(1.0))

    def source(sim):
        while True:
            yield arrivals.exponential(1 / p.arrival_rate)
            sim.process(customer(sim))

    sim.process(source(sim))


model = Model(bank, name="bank", duration=1_000,
              parameters=[Parameter("tellers", 2, "int", low=1),
                          Parameter("arrival_rate", 1.8, "float", low=0)])
run = model.simulate(seed=1)
print(run.metrics["resource.teller.wait.mean"], run.metrics["resource.teller.utilization"])
```

and then, without more model code:

```python
from simulsi import Experiment, Scenario, compare

result = Experiment(model, [Scenario("baseline"), Scenario("three", {"tellers": 3})],
                    replications=10, seed=1).run()
print(result.format_summary(["resource.teller.wait.mean"]))
print(compare(result, "baseline", metrics=["resource.teller.wait.mean"]).format())
```

## Differences to watch for

* **Randomness.** Use `sim.stream("name")` (or a distribution's
  `.sample(stream)`) instead of the global `random` module. Named streams
  keep scenarios comparable (common random numbers) and runs reproducible.
* **Releasing.** There is no context manager on requests; use `sim.use(...)`
  for the common acquire-hold-release pattern, or release explicitly.
  Finishing a process while holding a unit produces a warning.
* **Waking up.** As in SimPy, waiters resume through the event queue, so a
  `release()` deep inside one process never runs another process's code
  synchronously.
* **Failing loudly.** An exception in a process that nobody waits for stops
  the run with that exception instead of being lost.
* **Speed.** On a like-for-like M/M/2 queue SimulSI takes about 1.6x as
  long as SimPy, mostly because it records statistics on every resource
  operation (see [the benchmark](https://github.com/Yashjindal11/simulsi/tree/main/benchmarks#comparison-with-simpy)).
  For most studies the experiment layer saves far more time than that.
