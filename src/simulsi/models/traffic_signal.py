"""Signalised intersection: cycle length, green split and actuated control.

Vehicles arrive on a north-south and an east-west approach and queue at a
signal. During green, queued vehicles leave one per ``saturation_headway``
seconds; every phase change loses ``lost_time / 2`` seconds (amber and
all-red). With fixed timing the cycle repeats with greens split by
``green_split``; delay is U-shaped in the cycle length: short cycles waste
time switching, long ones make vehicles wait through long reds. Webster's
classic formula for fixed-time delay is reported alongside for comparison.
With ``control = "actuated"`` the green stays while vehicles keep arriving
(between ``min_green`` and ``max_green``) and switches when the queue
clears.

Try: ``cycle`` 30 vs 60 vs 150, ``green_split`` 0.3 vs 0.6, ``flow_ns``
up to 900, ``control`` fixed vs actuated.

Time unit: seconds. Flows in vehicles per hour. Synthetic data only.
"""

from __future__ import annotations

import math
from typing import Any

from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation

APPROACHES = ("ns", "ew")
EPS = 1e-9


def webster_delay(flow_vph: float, green: float, cycle: float, headway: float) -> float:
    """Webster (1958) average delay per vehicle at a fixed-time signal (NaN if oversaturated)."""
    q = flow_vph / 3600
    s = 1 / headway
    lam = green / cycle
    x = q / (lam * s) if lam > 0 else math.inf
    if q <= 0:
        return 0.0
    if x >= 1:
        return math.nan
    return float(
        cycle * (1 - lam) ** 2 / (2 * (1 - lam * x))
        + x**2 / (2 * q * (1 - x))
        - 0.65 * (cycle / q**2) ** (1 / 3) * x ** (2 + 5 * lam)
    )


def _build(sim: Simulation, p: Params) -> None:
    queues = {a: sim.queue(a) for a in APPROACHES}
    flows = {"ns": p.flow_ns, "ew": p.flow_ew}
    rs = sim.stream("vehicles")
    effective = p.cycle - p.lost_time
    if p.control == "fixed" and effective <= 0:
        raise ValueError("cycle must be longer than the lost time")
    greens = {"ns": effective * p.green_split, "ew": effective * (1 - p.green_split)}
    m = sim.metrics
    m.counter("phase_changes")
    for a in APPROACHES:
        m.counter(f"throughput.{a}")

    def source(a: str) -> Any:
        if flows[a] <= 0:
            return
        mean_gap = 3600 / flows[a]
        while True:
            yield rs.exponential(mean_gap)
            yield queues[a].put(sim.now)

    def wait_for_vehicle(q: Any, until: float) -> Any:
        """The next vehicle's arrival time, or ``None`` if none comes before ``until``."""
        if until - sim.now <= EPS:
            return None
        get = q.get()
        done = yield sim.any_of(get, sim.timeout(until - sim.now))
        if get in done:
            return done[get]
        q.cancel(get)
        return None

    def discharge(a: str, green: float, actuated: bool) -> Any:
        """Serve approach ``a`` for one green phase."""
        q = queues[a]
        start = sim.now
        end = start + (p.max_green if actuated else green)
        min_end = start + p.min_green
        while sim.now < end - EPS:
            if actuated and sim.now >= min_end - EPS:
                # gap-out: stay green only if a vehicle arrives within one headway
                arrived = yield from wait_for_vehicle(q, min(end, sim.now + p.saturation_headway))
            elif actuated and len(q) == 0:
                arrived = yield from wait_for_vehicle(q, min_end)
                if arrived is None:
                    continue
            else:
                arrived = yield from wait_for_vehicle(q, end)
            if arrived is None:
                return
            yield p.saturation_headway
            m.observe(f"delay.{a}", sim.now - arrived)
            m.observe("delay", sim.now - arrived)
            m.increment(f"throughput.{a}")

    def controller() -> Any:
        actuated = p.control == "actuated"
        while True:
            for a in APPROACHES:
                m.record("green_ns", 1 if a == "ns" else 0)
                yield from discharge(a, greens[a], actuated)
                m.record("green_ns", 0)
                m.increment("phase_changes")
                yield p.lost_time / 2

    for a in APPROACHES:
        sim.process(source(a), name=f"arrivals-{a}")
    sim.process(controller(), name="signal")

    if p.control == "fixed":

        def finish(sim: Simulation) -> None:
            for a in APPROACHES:
                sim.metrics.set(
                    f"webster.delay_{a}",
                    webster_delay(flows[a], greens[a], p.cycle, p.saturation_headway),
                )

        sim.on_finish(finish)


traffic_signal = Model(
    _build,
    name="traffic_signal",
    duration=3600.0,
    warmup=600.0,
    version="1",
    description="Two-phase signalised intersection: fixed vs actuated timing, compared with Webster.",
    parameters=[
        Parameter("flow_ns", 600.0, "float", low=0, unit="veh/h"),
        Parameter("flow_ew", 400.0, "float", low=0, unit="veh/h"),
        Parameter("control", "fixed", "str", choices=("fixed", "actuated")),
        Parameter("cycle", 60.0, "float", low=1, unit="s", description="fixed-time cycle length"),
        Parameter("green_split", 0.55, "probability", description="share of green for north-south"),
        Parameter(
            "lost_time", 8.0, "float", low=0, unit="s", description="lost per cycle (2 phases)"
        ),
        Parameter("saturation_headway", 2.0, "float", low=0.5, unit="s"),
        Parameter("min_green", 7.0, "float", low=0, unit="s"),
        Parameter("max_green", 50.0, "float", low=1, unit="s"),
    ],
    outputs=[
        "delay.mean",
        "delay.ns.mean",
        "delay.ew.mean",
        "webster.delay_ns",
        "queue.ns.max_length",
        "queue.ew.max_length",
    ],
    presets={
        "short_cycle": {"cycle": 30.0},
        "long_cycle": {"cycle": 120.0},
        "wrong_split": {"green_split": 0.35},
        "rush_hour": {"flow_ns": 850.0},
        "actuated": {"control": "actuated"},
        "rush_hour_actuated": {"flow_ns": 850.0, "control": "actuated"},
    },
)
