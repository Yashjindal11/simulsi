"""Files written by ``simulsi init``."""

MODEL_PY = '''"""A starter SimulSI model: customers queue for a pool of servers.

Run it:            simulsi run model.py --param servers=3
Run an experiment: simulsi experiment experiment.yaml
"""

from simulsi import Parameter, Simulation, model
from simulsi.randomness import Exponential


@model(
    duration=480.0,  # one 8-hour day in minutes
    parameters=[
        Parameter("arrival_rate", 0.9, "float", low=0.0, unit="customers/min"),
        Parameter("mean_service", 2.0, "float", low=0.0, unit="min"),
        Parameter("servers", 2, "int", low=1),
    ],
)
def service_desk(sim: Simulation, p):
    servers = sim.resource("server", capacity=p.servers)
    arrivals = sim.stream("arrivals")
    service = sim.stream("service")
    inter = Exponential(rate=p.arrival_rate)
    duration = Exponential(mean=p.mean_service)

    def customer(sim, c):
        c.set_state("waiting")
        req = yield sim.request(servers)
        c.set_state("in_service")
        yield duration.sample(service)
        sim.release(req)
        sim.dispose(c)

    def source(sim):
        while True:
            yield inter.sample(arrivals)
            c = sim.entity("customer")
            sim.process(customer(sim, c), entity=c)

    sim.process(source(sim), name="arrivals")


model = service_desk
'''

EXPERIMENT_YAML = """# SimulSI experiment configuration. Data only: nothing here is executed.
model: model.py:service_desk
name: service-desk-staffing

simulation:
  seed: 42

experiment:
  replications: 30
  workers: 1

parameters:          # base values (override model defaults)
  arrival_rate: 0.9

scenarios:
  - name: high_demand
    parameters: {arrival_rate: 1.3}
  - name: extra_server
    parameters: {servers: 3}
  - name: high_demand_extra_server
    parameters: {arrival_rate: 1.3, servers: 3}

metrics:
  - resource.server.wait.mean
  - resource.server.utilization
  - entity.customer.time_in_system.mean

cost:
  terms:
    - {name: staff, kind: cost, metric: resource.server.capacity_time, rate: 0.5}
    - {name: waiting, kind: cost, metric: resource.server.wait.total, rate: 0.2}

output:
  directory: results/service-desk
  formats: [json, csv]
"""

README_MD = """# {name}

A SimulSI project.

```bash
simulsi validate experiment.yaml
simulsi run model.py --param servers=3
simulsi experiment experiment.yaml
simulsi analyze results/service-desk
simulsi visualize results/service-desk --out plots
```
"""
