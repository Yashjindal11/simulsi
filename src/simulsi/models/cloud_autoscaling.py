"""Cloud service autoscaling under bursty traffic.

Requests arrive at a base rate with random traffic bursts. Each server
handles one request at a time; a request that waits longer than
``timeout`` fails. Every ``check_interval`` seconds an autoscaler compares
load (busy + queued requests per server) with its thresholds: above
``scale_up_at`` it launches a server, which only starts serving after
``spin_up`` seconds (a cold start); below ``scale_down_at`` it removes one.
Servers cost money while they are provisioned.

Try: ``spin_up`` 10 vs 120, ``min_servers`` 2 vs 8, ``scale_step`` 1 vs 4,
``burst_multiplier`` 1 vs 6, ``check_interval`` 5 vs 60.

Time unit: seconds. Synthetic data only.
"""

from __future__ import annotations

from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.schedules import arrivals
from simulsi.randomness.distributions import LogNormal


def _build(sim: Simulation, p: Params) -> None:
    servers = sim.resource("server", p.min_servers)
    rs, rt = sim.stream("requests"), sim.stream("traffic")
    service = LogNormal.from_moments(p.service_time, p.service_time)
    state = {"burst": False, "pending": 0}
    m = sim.metrics
    for name in ("requests", "timeouts", "slo_met", "scale_ups", "scale_downs"):
        m.counter(name)
    m.record("servers", p.min_servers)

    def traffic() -> Any:
        while True:
            yield rt.exponential(p.mean_time_between_bursts)
            state["burst"] = True
            m.record("burst", 1)
            yield rt.exponential(p.mean_burst_length)
            state["burst"] = False
            m.record("burst", 0)

    def rate(t: float) -> float:
        return float(p.request_rate) * (float(p.burst_multiplier) if state["burst"] else 1.0)

    def request(sim: Simulation, i: int) -> Any:
        m.increment("requests")
        t0 = sim.now
        req = yield sim.request(servers, patience=p.timeout)
        if not req.granted:
            m.increment("timeouts")
            return
        yield service.sample(rs)
        servers.release(req)
        latency = sim.now - t0
        m.observe("latency", latency)
        if latency <= p.slo_latency:
            m.increment("slo_met")

    def launch() -> Any:
        state["pending"] += 1
        m.increment("scale_ups")
        yield p.spin_up
        state["pending"] -= 1
        servers.set_capacity(servers.capacity + 1)
        m.record("servers", servers.capacity)

    def autoscaler() -> Any:
        while True:
            yield p.check_interval
            load = (servers.in_use + servers.queue_size) / max(1, servers.capacity)
            total = servers.capacity + state["pending"]
            if load > p.scale_up_at and total < p.max_servers:
                for _ in range(min(p.scale_step, p.max_servers - total)):
                    sim.process(launch(), name="launch")
            elif (
                load < p.scale_down_at
                and state["pending"] == 0
                and servers.capacity > p.min_servers
            ):
                servers.set_capacity(servers.capacity - 1)
                m.increment("scale_downs")
                m.record("servers", servers.capacity)

    sim.process(traffic(), name="traffic")
    arrivals(
        sim,
        rate,
        request,
        stream="arrivals",
        max_rate=p.request_rate * max(1.0, p.burst_multiplier),
    )
    sim.process(autoscaler(), name="autoscaler")

    def finish(sim: Simulation) -> None:
        c = sim.metrics.counters
        n = c["requests"].value
        sim.metrics.set("error_rate", c["timeouts"].value / n if n else 0.0)
        sim.metrics.set("slo_attainment", c["slo_met"].value / n if n else float("nan"))
        hours = servers.capacity_level.area() / 3600
        sim.metrics.set("cost", hours * p.server_cost_per_hour)

    sim.on_finish(finish)


cloud_autoscaling = Model(
    _build,
    name="cloud_autoscaling",
    duration=3600.0,
    warmup=300.0,
    version="1",
    description="Web service with bursty traffic, cold starts, timeouts and threshold autoscaling.",
    parameters=[
        Parameter("request_rate", 20.0, "float", low=0, unit="req/s"),
        Parameter("service_time", 0.2, "float", low=0.001, unit="s"),
        Parameter("burst_multiplier", 3.0, "float", low=1),
        Parameter("mean_time_between_bursts", 600.0, "float", low=1, unit="s"),
        Parameter("mean_burst_length", 120.0, "float", low=1, unit="s"),
        Parameter("min_servers", 4, "int", low=1),
        Parameter("max_servers", 30, "int", low=1),
        Parameter("spin_up", 45.0, "float", low=0, unit="s", description="cold-start time"),
        Parameter("check_interval", 15.0, "float", low=0.1, unit="s"),
        Parameter("scale_up_at", 0.8, "float", low=0, description="load per server that adds one"),
        Parameter(
            "scale_down_at", 0.3, "float", low=0, description="load per server that removes one"
        ),
        Parameter(
            "scale_step", 1, "int", low=1, description="servers launched per scale-up decision"
        ),
        Parameter("timeout", 2.0, "float", low=0, unit="s"),
        Parameter("slo_latency", 0.5, "float", low=0, unit="s"),
        Parameter("server_cost_per_hour", 0.10, "float", low=0),
    ],
    outputs=["slo_attainment", "error_rate", "latency.mean", "latency.p95", "servers.mean", "cost"],
    presets={
        "slow_cold_start": {"spin_up": 120.0},
        "instant_start": {"spin_up": 5.0},
        "overprovisioned": {"min_servers": 10},
        "lazy_autoscaler": {"check_interval": 60.0, "scale_up_at": 1.2},
        "flash_crowd": {"burst_multiplier": 6.0},
        "flash_crowd_ready": {
            "burst_multiplier": 6.0,
            "spin_up": 10.0,
            "check_interval": 5.0,
            "scale_step": 4,
        },
    },
)
