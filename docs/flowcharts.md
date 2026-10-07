# Flowchart models (YAML, no code)

Many process models are a flowchart: entities arrive, queue for resources,
are served, are routed on, and leave. SimulSI can build those from a YAML
file, which works everywhere a model is expected - `simulsi run`,
`whatif`, `experiment` configs, the dashboard (`simulsi ui --model
clinic.yaml`) and the optimiser.

```bash
simulsi run examples/flowcharts/clinic.yaml
simulsi whatif examples/flowcharts/clinic.yaml --presets
simulsi whatif examples/flowcharts/clinic.yaml --vary doctors=2,3,4 --vary arrival_mean=3,5
```

## Example

```yaml
flow:
  name: clinic
  duration: 600          # run length (time units of your choice)
  warmup: 60
  parameters:            # tunable values; refer to them as $name
    doctors: {default: 3, low: 1, description: doctors on shift}
    arrival_mean: 5.0
    lab_share: {default: 0.3, kind: probability}
  resources:
    nurse: 2
    doctor: $doctors
  sources:
    - name: patient
      interarrival: {distribution: exponential, mean: $arrival_mean}
      next: triage
  stations:
    triage: {resource: nurse, service: {distribution: triangular, low: 2, mode: 4, high: 8},
             next: consult}
    consult:
      resource: doctor
      service: {distribution: lognormal, mu: 2.5, sigma: 0.4}
      patience: {distribution: exponential, mean: 90}   # leaves if not seen in time
      next: {lab: $lab_share, exit: rest}             # random routing
    lab: {delay: 10, next: exit}
  presets:
    extra_doctor: {doctors: 4}
```

From Python:

```python
from simulsi.flowchart import load_flowchart

model = load_flowchart("""
flow:
  name: shop
  duration: 480
  parameters: {clerks: 2}
  resources: {clerk: $clerks}
  sources: [{name: customer, rate: 0.8, next: till}]
  stations: {till: {resource: clerk, service: {distribution: exponential, mean: 2}, next: exit}}
""")
print(model.simulate(seed=1).metrics["resource.clerk.utilization"])
print(model.evaluate({"clerks": 3}, metric="entity.customer.time_in_system.mean"))
```

## Reference

**Top level** (under `flow:`): `name`, `description`, `version`, `duration`,
`warmup`, `parameters`, `resources`, `sources`, `stations`, `sinks`,
`presets`, `outputs`. Unknown keys are errors, so typos are caught.

**parameters**: `name: value` (kind inferred) or `name: {default, kind, low,
high, unit, description}`. Any value elsewhere written as `$name` is replaced
by the parameter's value for each run.

**resources**: `name: capacity` or `name: {capacity, discipline, schedule,
period}`. `discipline` is `fifo`, `lifo` or `priority`; `schedule` is a
list of `[time, capacity]` rows (shifts), repeating every `period`.

**sources** (a list): `name`, `entity` (type name, default: the source name),
exactly one of `interarrival` (a distribution or number), `rate` (Poisson
arrivals per time unit) or `rate_table` (`[[time, rate], ...]` with an
optional `period`), optional `limit`, and `next`.

**stations**: either `resource` + `service` (with optional `units` - how
many units to hold - `priority` and `patience`), or `delay` (time passes
with no resource). Durations are numbers or
[distribution specs](concepts.md#randomness).

**next**: a station or sink name, or a mapping `{name: probability}`; one
entry may be `rest` for whatever probability is left. Sinks are `exit` plus
any names listed in `sinks`; entities that run out of patience go to the
`abandoned` sink.

**Metrics** (besides the usual resource metrics):
`entity.<type>.time_in_system.*`, `station.<name>.visits`,
`station.<name>.abandoned`, `sink.<name>` and `route.<station>.<destination>`.
Default outputs are each entity type's mean time in system and each
resource's utilization and mean wait.

Flowchart files are data: they can only create the building blocks above and
registered distributions, never run code. For anything more (state-dependent
logic, custom metrics), write a Python model - the YAML keys map one-to-one
onto `sim.resource`, `sim.request`, `Router` and `arrivals`.
