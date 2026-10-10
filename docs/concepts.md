# Core concepts

```text
Simulation
 ├── Clock            current time (float), optional calendar anchoring
 ├── Event queue      binary heap ordered by (time, priority, scheduling order)
 ├── Events           things that happen at an instant (callbacks or Event subclasses)
 ├── Processes        generator functions that wait for time, resources, items, signals
 ├── Entities         customers, jobs, patients, aircraft ... with state and history
 ├── Resources        pools of capacity (servers, machines, gates) with waiting lines
 ├── Queues           buffers with disciplines and optional capacity
 ├── Randomness       seeded, named random streams + distributions
 ├── Metrics          tallies, time-weighted gauges, counters, histograms
 └── Event log        optional structured trace (JSON / CSV / Parquet)
```

## Simulation and time

```python
from datetime import datetime, timedelta

from simulsi import Simulation

sim = Simulation(seed=1)                    # time is a float in abstract units
cal = Simulation(seed=1, epoch=datetime(2026, 3, 1, 6, 0), time_unit=timedelta(minutes=1))
cal.schedule(time=datetime(2026, 3, 1, 7, 30), event_type="shift_change")
cal.schedule(delay=timedelta(hours=2), event_type="inspection")
cal.run(until=timedelta(hours=3))
print(cal.now, cal.now_datetime)            # 180.0 2026-03-01 09:00:00
```

`run(until=T)` executes every event with timestamp `<= T` and leaves the
clock exactly at `T`, even if nothing happened near the end; `run()` without
`until` stops when no events remain; `sim.stop()` ends a run after the
current event. Runs can be resumed by calling `run` again with a later `until`.

## Events

```python
from simulsi import Event, Priority, Simulation

sim = Simulation(seed=1)
log = []
sim.schedule(time=5, event_type="ping", callback=lambda s, e: log.append(("ping", s.now)))


class Arrival(Event):
    def __init__(self, n):
        super().__init__("arrival", {"n": n})

    def execute(self, sim):
        log.append(("arrival", sim.now, self.payload["n"]))
        if self.payload["n"] < 2:
            sim.schedule(Arrival(self.payload["n"] + 1), delay=3)


first = sim.schedule(Arrival(0), time=1)
urgent = sim.schedule(time=5, event_type="urgent", priority=Priority.URGENT,
                      callback=lambda s, e: log.append(("urgent", s.now)))
late = sim.schedule(time=50, event_type="cancel-me")
sim.cancel(late)
sim.run()
print(log)   # urgent runs before ping at t=5
```

Every event has `event_id`, `timestamp`, `priority`, `event_type`, `payload`,
`callback` and `status` (`created`, `scheduled`, `cancelled`, `executed`).
`reschedule(event, time=...)` moves a pending (or cancelled) event. Ordering is
fully deterministic: time, then priority (lower first), then scheduling order.

## Processes

A process is a generator. What it yields decides what it waits for:

| yield | waits for | resumes with |
|---|---|---|
| a number / `timedelta` | that much time | `None` |
| `sim.timeout(d, value)` | `d` time units | `value` |
| `sim.request(resource)` | a unit of the resource | the `Request` |
| `queue.get()` / `queue.put(x)` | an item / free space | the item / `None` |
| `sim.signal()` | someone calling `signal.succeed(v)` | `v` |
| `sim.all_of(a, b)` / `sim.any_of(a, b)` | all / any of them | `{waitable: value}` |
| another `Process` | its completion | its return value |

```python
from simulsi import Interrupt, Simulation

sim = Simulation(seed=1)
machine = sim.resource("machine", capacity=1)


def job(sim):
    try:
        yield from sim.use(machine, 10)       # acquire, hold 10, release
    except Interrupt as stop:
        print(f"t={sim.now}: interrupted ({stop.cause}); unit released automatically")


p = sim.process(job(sim), name="job-1")
sim.call_at(lambda: p.interrupt("power cut"), time=4)
sim.run()
```

If a process raises and nobody waits on it, the exception propagates out of
`sim.run()` - model bugs are never swallowed.

## Entities

```python
from simulsi import Simulation

sim = Simulation(seed=1)
c = sim.entity("customer", priority=2, service_type="premium")   # id "customer-1"
c.set_state("waiting")
sim.run(until=3)
c.set_state("served", by="teller-2")
sim.run(until=5)
sim.dispose(c)                         # records time in system
print(c.id, c["priority"], c.time_in_states())
print(sim.run().metrics["entity.customer.time_in_system.mean"])
```

Entities have an `id` (reproducible per simulation), `created_at`, `state`,
free-form `attributes` and a `history` of state changes. Pass `entity=` to
`sim.process` or `sim.request` so logs and flow graphs know who did what.

## Resources

```python
from simulsi import Simulation

sim = Simulation(seed=1)
doctors = sim.resource("doctor", capacity=2, discipline="priority")


def patient(sim, acuity, arrive):
    yield arrive
    req = yield sim.request(doctors, priority=acuity, patience=30)
    if not req.granted:          # reneged after 30 time units
        return
    yield 20
    sim.release(req)


for i, acuity in enumerate([3, 3, 1, 2, 3]):
    sim.process(patient(sim, acuity, i))
m = sim.run(until=200).metrics
print(m["resource.doctor.utilization"], m["resource.doctor.wait.mean"])
```

* **Disciplines**: `"fifo"` (default), `"lifo"`, `"priority"` (lower value
  first, FIFO among equals) or any key function.
* **Reneging**: `patience=` withdraws an ungranted request after a while.
* **Capacity changes and downtime**: `set_capacity(n)`, `fail(units, interrupt=False)`,
  `repair(units)`. Units in use finish normally unless `interrupt=True`.
* **Statistics** (prefix `resource.<name>.`): `utilization` (busy / capacity,
  time-weighted), `availability`, `mean_busy`, `mean_queue_length`,
  `max_queue_length`, `wait.mean/max/p95/total`, `requests`, `grants`,
  `releases`, `reneges`, `failures`, `preemptions`, `throughput`, `busy_time`, `capacity_time`.
* **Invariants** (property-tested): `0 <= in_use <= capacity`;
  `requests = grants + waiting + reneged`; `grants = releases + preemptions + in_use`.

### Preemption

`sim.resource(name, capacity, discipline="priority", preemptive=True)` lets
a request that finds no free unit evict the least important user, but only if
the request is strictly more important (lower value). The evicted process gets
an `Interrupt` whose `cause` is a `Preempted(resource, by, usage_since)`. Its
unit is already released (calling `release` on it is a no-op), so it usually
re-requests the work that remains:

```python
from simulsi import Interrupt, Preempted, Simulation

sim = Simulation(seed=1)
or_room = sim.resource("theatre", 1, discipline="priority", preemptive=True)


def operation(sim, name, priority, arrive, work):
    yield arrive
    while work > 0:
        req = yield sim.request(or_room, priority=priority)
        start = sim.now
        try:
            yield work
            work = 0
            sim.release(req)
        except Interrupt as stop:
            assert isinstance(stop.cause, Preempted)
            work -= sim.now - start
            print(f"t={sim.now}: {name} bumped, {work} left")
    print(f"t={sim.now}: {name} done")


sim.process(operation(sim, "elective", 5, 0, 6))
sim.process(operation(sim, "emergency", 1, 2, 3))
sim.run()
```

## Queues

`Queue` is an ordered buffer between process steps, with optional capacity
(blocking `put`) and blocking `get`:

```python
from simulsi import Simulation

sim = Simulation(seed=1)
orders = sim.queue("orders", discipline="priority", capacity=10)


def producer(sim):
    for i in range(5):
        yield orders.put(f"order-{i}", priority=5 - i)
        yield 1


def consumer(sim):
    yield 10
    while True:
        item = yield orders.get()
        print(sim.now, item)
        yield 2


sim.process(producer(sim))
sim.process(consumer(sim))
print(sim.run(until=30).metrics["queue.orders.wait.mean"])
```

Metrics (prefix `queue.<name>.`): `mean_length`, `max_length`, `wait.mean/max`,
`puts`, `gets`, `removed`, `throughput`, `length_final`. Items never vanish:
`puts == gets + removed + length`.

`orders.get(filter=lambda o: o.startswith("rush"))` takes the first matching
item in discipline order (waiting until one arrives); a filtered get that is
waiting does not hold up gets for other items. `try_get(filter)` is the
non-blocking form.

## Containers

A `Container` holds a *level* (fuel, stock, cash, parts in a bin) rather than
distinct items. `get(amount)` waits until enough is there; `put(amount)` waits
until there is room. Waiters are served strictly in arrival order.

```python
from simulsi import Simulation

sim = Simulation(seed=2)
stock = sim.container("stock", capacity=200, init=120)


def demand(sim):
    rng = sim.stream("demand")
    while True:
        yield 1.0
        yield stock.get(rng.integers(5, 15))


def replenish(sim):
    while True:
        yield sim.wait_until(lambda: stock.level < 40)   # reorder point
        yield 3.0                                         # lead time
        yield stock.put(150)


sim.process(demand(sim))
sim.process(replenish(sim))
m = sim.run(until=500).metrics
print(m["container.stock.level.mean"], m["container.stock.stockouts"])
```

Metrics (prefix `container.<name>.`): `level.mean/min/max`, `level_final`,
`put_amount`, `get_amount`, `stockouts` (gets that had to wait),
`get_wait.mean`, `put_wait.mean`.

`sim.wait_until(predicate)` works with any model state: the predicate is
checked after every event until it is true, and the yield returns the time
it became true. Keep predicates cheap.

When the predicate only depends on some resources, containers or queues,
list them: `sim.wait_until(lambda: stock.level < 40, on=[stock])`. The
predicate is then re-checked only when one of them changes, which makes
many pending conditions cheap (about 8x faster in a model with hundreds of
waiting processes).

## Continuous levels

A `Level` is a quantity that changes at a rate between events: a fuel
tank, a battery, a reservoir. Processes change the rate or add to the
value, and wait for thresholds; crossing times are computed exactly, not
by stepping.

```python
from simulsi import Level, Simulation

sim = Simulation()
battery = Level(sim, "battery", init=80, rate=-2.0, low=0, high=100)  # drains 2 per minute


def charger(sim):
    while True:
        yield battery.when(20, "down")
        battery.set_rate(10.0)          # plug in
        yield battery.when(100, "up")
        battery.set_rate(-2.0)          # back to work


sim.process(charger(sim))
m = sim.run(until=600).metrics
print(m["level.battery.mean"], m["level.battery.min"])
```

* `when(threshold, "up" | "down" | "any")` triggers with the time the
  level gets there, or at once if it already is.
* The level stays within `[low, high]`: at a bound it stops (`rate` reads
  0) until the rate turns.
* `add(amount)` jumps (a delivery) and returns what fitted.
* Metrics `level.<name>.mean` (exact time average), `final`, `min`, `max`.

## Movement on networks

`simulsi.spatial` gives travel times on a graph of places: roads between
depots, aisles in a warehouse, routes between airports.

```python
from simulsi import Simulation
from simulsi.spatial import Network, grid_network

roads = Network([("depot", "A", 12), ("A", "B", 7), ("depot", "B", 25)], name="roads")
print(roads.travel_time("depot", "B"), roads.path("depot", "B"))   # 19.0 ['depot', 'A', 'B']

sim = Simulation()


def van(sim):
    here = "depot"
    for stop in ["A", "B", "depot"]:
        yield from roads.trip(sim, here, stop)
        here = stop


sim.process(van(sim))
print(sim.run().metrics["network.roads.trip_time.mean"])
```

* `Network(links, directed=False)`, `add_link`, `remove_link` (a closed
  road), `travel_time`, `path`, `nearest(origin, candidates)` and
  `matrix()`. Shortest paths use Dijkstra and are cached.
* `grid_network(width, height, diagonal=False, blocked=[...])` builds a
  grid of `(x, y)` cells with walls.
* `Network.from_coordinates(points, speed=..., metric="haversine")` links
  places by straight-line or great-circle distance.
* `yield from net.trip(sim, a, b, on_leg=...)` moves edge by edge, so the
  clock advances by the travel time; `on_leg(from, to, time)` can log or
  animate positions.

## Schedules and time-varying arrivals

```python
from simulsi import Simulation
from simulsi.processes import PiecewiseRate, arrivals, capacity_schedule

sim = Simulation(seed=4)
agents = sim.resource("agents", 2)
# shifts in minutes: 2 agents at night, 6 from 08:00, 4 from 17:00, every day
capacity_schedule(sim, agents, [(0, 2), (480, 6), (1020, 4)], period=1440)
# calls per minute by time of day
calls = PiecewiseRate([(0, 0.05), (480, 0.6), (720, 0.9), (1020, 0.3)], period=1440)


def call(sim, i):
    yield from sim.use(agents, sim.stream("talk").exponential(6.0))


arrivals(sim, calls, call, stream="calls")
m = sim.run(until=7 * 1440).metrics
print(m["resource.agents.utilization"], m["resource.agents.wait.mean"])
```

* `capacity_schedule(sim, resource, [(time, capacity), ...], period=None)`
  changes capacity at the given offsets. Lowering capacity does not
  interrupt work in progress; the new limit applies to later grants.
* `arrivals(sim, rate, spawn, stream=..., limit=None, until=None)` calls
  `spawn(sim, i)` at every arrival and starts it as a process if it returns a
  generator. `rate` can be a number (Poisson process), a `PiecewiseRate`
  (exact, one random draw per arrival) or any function `rate(t)` together with
  `max_rate=` (generated by thinning).

## Batching, split/join and routing

```python
from simulsi import Simulation
from simulsi.processes import Router, batch, join, shortest_queue, split

sim = Simulation(seed=5)
parts = sim.queue("parts")
lines = [sim.resource("line-a", 1), sim.resource("line-b", 1)]
inspection = Router(sim, "inspection", {"pass": 0.9, "rework": 0.1})


def supplier(sim):
    for i in range(12):
        yield 0.5
        yield parts.put(i)


def finish(sim, line):
    yield from sim.use(line, 2.0)
    return inspection.choose()


def assembler(sim):
    while True:
        kit = yield from batch(parts, 4, timeout=5.0)     # up to 4 parts
        if not kit:
            return
        results = yield from join(split(sim, [finish(sim, shortest_queue(lines))
                                              for _ in range(2)]))
        print(sim.now, kit, results)


sim.process(supplier(sim))
sim.process(assembler(sim))
m = sim.run().metrics
print(m["route.inspection.pass"], m["route.inspection.rework"])
```

* `batch(queue, size, timeout=None)` collects items (`yield from`); with a
  timeout it returns whatever has arrived by then.
* `split(sim, generators)` starts parallel processes and `join(processes)`
  waits for all of them and returns their return values in order (a failure
  in any of them is raised in the joiner).
* `Router(sim, name, {route: probability})` draws from its own stream
  (`route:<name>`) and counts `route.<name>.<route>`.
* `shortest_queue(resources)` picks the resource with the fewest waiting.

## Randomness

```python
from simulsi.randomness import (Categorical, Empirical, Exponential, LogNormal, Normal,
                                RandomStream, Triangular, from_spec)

rng = RandomStream(seed=42)
service = LogNormal.from_moments(mean=5.0, std=2.0)
print(service.sample(rng), service.mean, service.variance)
print(Categorical({"economy": 0.8, "business": 0.2}).sample(rng))
print(Empirical([3.1, 4.7, 5.2, 9.9]).sample_n(rng, 3))
print(from_spec({"distribution": "triangular", "low": 1, "mode": 2, "high": 6}))
```

Available: `Constant`, `Uniform`, `Normal`, `Exponential` (rate or mean),
`Poisson`, `Binomial`, `Gamma`, `LogNormal` (+ `from_moments`), `Triangular`,
`Empirical` (optionally weighted), `Categorical`, and `Custom(fn)`.
Parameters are validated eagerly (e.g. probabilities must be in [0, 1] and
sum to 1). `RandomStream` also has `uniform`, `normal`, `exponential`,
`poisson`, `binomial`, `gamma`, `lognormal`, `triangular`, `bernoulli`,
`choice`, `integers` and `shuffle`.

In a simulation use `sim.stream("name")`: streams are keyed by name, so the
arrival stream is the same whatever else the model draws. This is what makes
common-random-number comparisons effective.

## Input data: fitting and traces

Input distributions should come from data. `fit_distribution` fits the
exponential, gamma, lognormal, normal, uniform, triangular and Poisson
families by maximum likelihood and ranks them:

```python
import numpy as np

from simulsi.randomness import fit_distribution

observed = np.random.default_rng(7).gamma(2.5, 1.6, size=400)   # e.g. service times
report = fit_distribution(observed)
print(report.format())                    # AIC, BIC, KS statistic and p-value per family
service = report.best.distribution        # a simulsi Gamma, ready to sample
print(service, report.best.distribution.to_spec())
```

* Families whose support does not fit the data (gamma or lognormal with
  zeros, Poisson with fractions) are listed in `report.skipped`.
* KS p-values use parameters estimated from the same data, so they are
  optimistic. With fewer than about 30 observations, or when the best family
  is still rejected, use `Empirical(data)` instead.
* `report.plot()` overlays the top fits on a histogram.
* `simulsi fit data.csv --column service` does the same from the command
  line and prints a spec for `experiment.yaml`.

**Trace-driven runs** replay history instead of sampling. With recorded
arrivals and service times, a valid model should reproduce the recorded
waits, which makes this a strong validation check:

```py
from simulsi.processes import TraceReplay, load_trace, trace_arrivals

history = load_trace("arrivals.csv")          # rows like {"time": 3.5, "service": 4.1}

def job(sim, record):
    yield from sim.use(server, record["service"])

trace_arrivals(sim, history, job)               # one arrival per row, at row["time"]
repair_times = TraceReplay([12.0, 30.5, 8.2], cycle=True)
repair_times.sample()                           # next recorded value
```

## Metrics

```python
from simulsi import Simulation

sim = Simulation(seed=1)
m = sim.metrics
m.observe("wait_time", 4.2)          # one observation (tally: mean, std, quantiles)
m.record("work_in_progress", 7)      # a level that holds until changed (time-weighted)
m.increment("orders_shipped")        # a count (also reported as a rate per time unit)
m.histogram("order_size", 3, bins=[0, 2, 5, 10])
sim.on_finish(lambda s: s.metrics.set("cost", 1234.5))   # a final value
r = sim.run(until=10)
print(r.metrics["wait_time.mean"], r.metrics["orders_shipped.rate"], r.metrics["cost"])
print(r.details["metrics"]["tallies"]["order_size"]["histogram"])
```

`sim.warmup(t)` (or `Model(warmup=...)`) discards statistics collected before
`t` so steady-state estimates are not biased by an empty-and-idle start.

## Failures and disruptions

```python
from simulsi import Simulation
from simulsi.processes import FailureProcess, RecoveryProcess, capacity_reduction
from simulsi.randomness import Exponential, Triangular

sim = Simulation(seed=3)
press = sim.resource("press", capacity=2)
crew = sim.resource("crew", capacity=1)
FailureProcess(sim, press, time_to_failure=Exponential(mean=300),
               recovery=RecoveryProcess(Triangular(10, 20, 60), crew=crew), units=1)
capacity_reduction(sim, press, at=500, duration=60, capacity=1)   # planned maintenance
m = sim.run(until=2000).metrics
print(m["failure.press.count"], m["resource.press.availability"])
```

`ScheduledDisruption(sim, at=..., duration=..., apply=..., revert=...)` covers
anything else - a demand spike, a weather window, a closed route.

## State snapshots, event logs and graphs

```python
from simulsi.introspection import model_graph
from simulsi.models import mmc

sim = mmc.create({"servers": 2}, seed=1, trace=True)
sim.run(until=50)
snap = sim.snapshot()          # time, pending events, entities, resources, queues, metrics
print(snap["now"], snap["resources"]["server"])
sim.log.export("trace.csv")    # or .json / .parquet

graphs = model_graph(mmc, seed=1, duration=50)
print(graphs["inventory"])
```

The *flow graph* is built from the event log of a traced run (entities moving
between arrival, waiting lines, resources, queues and exit) and renders to
Mermaid or Graphviz DOT; the *state graph* shows each entity type's state
machine. Both are observed from a run - SimulSI models are ordinary Python,
so the structure is not inferred from source code.

## Model validation

```python
from simulsi.models import mmc
from simulsi.validation import validate_model

report = validate_model(mmc, {"servers": 0})
print(report.format())          # parameter error: below minimum 1
print(validate_model(mmc, {"servers": 2}).ok)
```

Checks: parameter types, bounds and required values; build errors; zero
capacities; exceptions during a smoke run (e.g. events scheduled in the past);
models that schedule nothing; processes that finish while holding a resource;
processes blocked with no pending events (possible deadlock); and, given
`expected_states`, states never reached or never declared. A clean report is
evidence, not proof.
