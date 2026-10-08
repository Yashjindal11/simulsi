"""Simulate one day of an airline network: aircraft rotations, crews, runways, gates and recovery.

Each aircraft (tail) works through its rotation as a simulsi process. A
departure waits for the aircraft (inbound arrival plus minimum turn), the
crew (its previous flight plus a connection time) and any known estimate,
then picks up a sampled *primary* delay and queues for a departure slot
(only at airports with a runway rate or weather; arrivals there queue for
landing slots too, and wait out closures). About an hour before each
departure, operations control looks at the projected delay and may

* swap in a spare aircraft at the airport (the late aircraft becomes the spare),
* call a standby crew (late inbound crew, or a crew that would break its duty limit),
* cancel a round trip when the projected delay passes ``cancel_threshold``,
* cancel the rest of a rotation that would break a curfew.

Arrivals may wait for a gate at airports with a gate limit. Passenger
connections are checked after the day: a connection is missed when either
flight is cancelled or the outbound leaves less than ``mct`` minutes after
the inbound arrives.

The engine is deterministic for a seed, and every flight draws its random
primary delay, block time and turn time from its own position in a fixed
order, so two plans (say, with and without an action) see the same
disturbances - common random numbers, which makes comparisons sharp.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

from simulsi.aviation.config import OpsConfig, WeatherEvent
from simulsi.aviation.passengers import eu261_compensation
from simulsi.aviation.schedule import Flight, Schedule
from simulsi.aviation.state import Action, OpsState, apply_actions
from simulsi.core.model import Model, Parameter, Params
from simulsi.core.simulation import Simulation
from simulsi.processes.process import Signal

LOOKAHEAD = 60.0
SPARE_SETUP = 20.0


@dataclass
class FlightOutcome:
    """What happened to one flight in one simulated day."""

    id: str
    tail: str
    dep: float = math.nan
    arr: float = math.nan
    cancelled: bool = False
    reason: str = ""
    dep_delay: float = math.nan
    arr_delay: float = math.nan
    primary: float = 0.0
    reactionary_aircraft: float = 0.0
    reactionary_crew: float = 0.0
    runway: float = 0.0
    gate_wait: float = 0.0
    spare: bool = False
    standby: bool = False
    crew_risk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DayOutcome:
    """One simulated day: per-flight outcomes, missed connections and network metrics."""

    flights: dict[str, FlightOutcome]
    misconnected: dict[tuple[str, str], int]
    metrics: dict[str, float]
    weather: list[str] = field(default_factory=list)


def _hub(schedule: Schedule) -> str:
    counts: dict[str, int] = {}
    for f in schedule.flights:
        counts[f.origin] = counts.get(f.origin, 0) + 1
    return max(counts, key=lambda a: (counts[a], a)) if counts else ""


def build_day(
    sim: Simulation,
    schedule: Schedule,
    config: OpsConfig,
    *,
    weather: Sequence[WeatherEvent] = (),
    state: OpsState | None = None,
    cancelled: Iterable[str] = (),
    out: dict[str, Any] | None = None,
) -> None:
    """Add one day of operations to ``sim``; outcomes land in ``out`` when the run finishes."""
    cfg = config
    result: dict[str, Any] = out if out is not None else {}
    forced = set(cancelled)
    status = state.flights if state else {}
    now0 = state.now if state else -math.inf
    rs_wx = sim.stream("weather")
    active = [w for w in weather if w.probability >= 1 or rs_wx.random() < w.probability]
    result["weather"] = [w.name or f"{w.airport} weather" for w in active]

    # Pre-draw randomness per flight in id order: plans that differ only by actions share it.
    rs_p, rs_b, rs_t = sim.stream("primary"), sim.stream("block"), sim.stream("turn")
    ordered = sorted(schedule.flights, key=lambda f: f.id)
    primary = {f.id: cfg.delays.primary(f, rs_p) for f in ordered}
    block = {f.id: cfg.delays.block(f, rs_b) for f in ordered}
    turn_factor = {f.id: rs_t.triangular(0.9, 1.0, 1.4) for f in ordered}

    wx_airports = {w.airport for w in active}
    base_rate = {a: float(r) for a, r in cfg.runway_rate.items()}
    for a in wx_airports:
        base_rate.setdefault(a, cfg.weather_base_rate)
    runways = {a: sim.resource(f"runway.{a}", 1) for a in sorted(base_rate)}
    runways_in = {a: sim.resource(f"arrivals.{a}", 1) for a in sorted(base_rate)}

    tail_process: dict[str, Any] = {}

    def _move(req: Any, proc: Any) -> None:
        """Hand a granted request to another process, which will release it."""
        if req.owner is not None and req in req.owner.held:
            req.owner.held.remove(req)
        req.owner = proc
        proc.held.append(req)

    def _wait_release(req: Any, minutes: float) -> Any:
        yield minutes
        sim.release(req)

    def _release_later(req: Any, minutes: float) -> None:
        _move(req, sim.process(_wait_release(req, minutes), name="release"))

    gates = {a: sim.resource(f"gate.{a}", int(n)) for a, n in sorted(cfg.gates.items())}
    spares: dict[str, list[float]] = {a: [0.0] * int(n) for a, n in cfg.spares.items() if n > 0}
    standby = {a: int(n) for a, n in cfg.standby_crews.items()}

    outcomes: dict[str, FlightOutcome] = {
        f.id: FlightOutcome(f.id, f.tail) for f in schedule.flights
    }
    arrived: dict[str, Signal] = {f.id: sim.signal(f"arr.{f.id}") for f in schedule.flights}
    projected_arr: dict[str, float] = {f.id: f.sta for f in schedule.flights}
    crew_prev: dict[str, Flight] = {}
    duty_start: dict[str, float] = {}
    pairing_end: dict[str, float] = {}
    for crew, legs in schedule.pairings.items():
        duty_start[crew] = legs[0].std - cfg.report_time
        pairing_end[crew] = legs[-1].sta
        for before, after in pairwise(legs):
            if before.std < after.std:
                crew_prev[after.id] = before

    def capacity(airport: str, t: float) -> tuple[float, float]:
        """(capacity factor, time it changes) at ``t``."""
        factor, until = 1.0, math.inf
        for w in active:
            if w.airport == airport and w.start <= t < w.end and w.capacity < factor:
                factor, until = w.capacity, w.end
        return factor, until

    def finish_flight(f: Flight, arr: float) -> None:
        outcomes[f.id].arr = arr
        outcomes[f.id].arr_delay = arr - f.sta
        projected_arr[f.id] = arr
        arrived[f.id].succeed(arr)

    def cancel(f: Flight, reason: str) -> None:
        o = outcomes[f.id]
        o.cancelled, o.reason = True, reason
        projected_arr[f.id] = f.sta
        arrived[f.id].succeed(f.sta)

    def crew_ready_projection(f: Flight) -> float:
        prev = crew_prev.get(f.id)
        if prev is None:
            return -math.inf
        if arrived[prev.id].triggered:
            return float(arrived[prev.id].value) + cfg.min_crew_connect
        o = outcomes[prev.id]
        if not math.isnan(o.dep):
            start = o.dep
        else:
            st = status.get(prev.id)
            known = st.etd if st is not None and st.etd is not None else -math.inf
            start = max(prev.std, sim.now, known)
        return start + prev.block + cfg.min_crew_connect

    swap_signal: dict[str, Signal] = {}
    released: dict[str, int] = {}

    def controller(prev: Flight, nxt: Flight) -> Any:
        """Operations control looks at ``nxt`` an hour ahead while its aircraft is still inbound."""
        t = max(sim.now, nxt.std - LOOKAHEAD)
        if t > sim.now:
            yield t - sim.now
        if arrived[prev.id].triggered or nxt.id in forced or nxt.id in status:
            return
        pool = spares[nxt.origin]
        proj_ready = max(projected_arr[prev.id], sim.now) + cfg.turn(nxt.origin)
        if proj_ready - nxt.std <= cfg.swap_threshold:
            return
        k = min(range(len(pool)), key=pool.__getitem__)
        spare_ready = max(pool[k], sim.now + SPARE_SETUP, nxt.std)
        if spare_ready < proj_ready - 0.5 * cfg.swap_threshold:
            pool[k] = math.inf  # reserved until the late aircraft lands and takes its place
            swap_signal[nxt.id].succeed((k, spare_ready))

    def arrive(f: Flight, o: FlightOutcome, touchdown: float, queue: bool) -> Any:
        """Land, take a gate and turn; returns (gate request, ready time) unless swapped out."""
        if touchdown > sim.now:
            yield touchdown - sim.now
        landing = runways_in.get(f.dest) if queue else None
        if landing is not None:
            held = sim.now
            req = yield sim.request(landing)
            while True:
                factor, until = capacity(f.dest, sim.now)
                if factor > 0:
                    break
                yield until - sim.now
            o.runway += sim.now - held
            _release_later(req, 60.0 / (base_rate[f.dest] * factor))
        finish_flight(f, sim.now)
        gate_req = None
        gate_pool = gates.get(f.dest)
        if gate_pool is not None:
            arr_t = sim.now
            gate_req = yield sim.request(gate_pool)
            o.gate_wait = sim.now - arr_t
        ready_t = sim.now + cfg.turn(f.dest) * turn_factor[f.id]
        k = released.pop(f.id, None)
        if k is not None:  # swapped out: this aircraft becomes the spare
            spares[f.dest][k] = ready_t
            if gate_req is not None:
                yield ready_t - sim.now
                sim.release(gate_req)
            return None
        if gate_req is not None:
            _move(gate_req, tail_process[f.tail])
        return gate_req, ready_t

    def rest_to_cancel(legs: Sequence[Flight], i: int) -> int:
        """Legs to cancel from ``i``: a round trip when possible, else the rest of the rotation."""
        f = legs[i]
        if i + 1 < len(legs) and legs[i + 1].origin == f.dest and legs[i + 1].dest == f.origin:
            return 2
        return len(legs) - i

    def tail(name: str, legs: list[Flight]) -> Any:
        ready = state.aircraft_unavailable.get(name, 0.0) if state else 0.0
        gate_req = None
        spare_next = False
        i = 0
        while i < len(legs):
            f = legs[i]
            o = outcomes[f.id]
            st = status.get(f.id)
            if f.id in forced or (st is not None and st.cancelled):
                cancel(f, "action" if f.id in forced else "status")
                i += 1
                continue
            if st is not None and st.atd is not None:
                # already airborne or landed: replay the known times
                if sim.now < st.atd:
                    yield st.atd - sim.now
                if gate_req is not None:
                    sim.release(gate_req)
                    gate_req = None
                o.dep, o.dep_delay = st.atd, st.atd - f.std
                o.primary = max(0.0, o.dep_delay)
                touchdown = st.ata if st.ata is not None else max(now0, st.atd + block[f.id])
                projected_arr[f.id] = touchdown
                arrival = sim.process(arrive(f, o, touchdown, queue=False), name=f"arr-{f.id}")
            else:
                decide_at = max(sim.now, f.std - LOOKAHEAD, now0)
                if decide_at > sim.now:
                    yield decide_at - sim.now
                earliest = max(f.std, now0)
                if st is not None and st.etd is not None:
                    earliest = max(earliest, st.etd)
                crew = f.crew
                # spare aircraft: swapped in while the late aircraft was inbound, or now
                pool = spares.get(f.origin)
                if spare_next:
                    o.spare, spare_next = True, False
                elif ready - max(earliest, f.std) > cfg.swap_threshold and pool:
                    k = min(range(len(pool)), key=pool.__getitem__)
                    spare_ready = max(pool[k], sim.now + SPARE_SETUP)
                    if spare_ready < ready - 0.5 * cfg.swap_threshold:
                        pool[k] = ready
                        ready = spare_ready
                        o.spare = True
                # standby crew for a late inbound crew
                crew_proj = crew_ready_projection(f)
                plan_dep = max(earliest, ready)
                use_standby = bool(
                    crew
                    and crew_proj - plan_dep > cfg.swap_threshold
                    and standby.get(f.origin, 0) > 0
                )
                # duty limit check for whoever is operating
                if crew and not use_standby:
                    proj_dep = max(plan_dep, crew_proj)
                    if proj_dep + f.block + cfg.release_time - duty_start[crew] > cfg.duty_limit:
                        if standby.get(f.origin, 0) > 0:
                            use_standby = True
                        else:
                            n = rest_to_cancel(legs, i)
                            for g in legs[i : i + n]:
                                cancel(g, "crew")
                            i += n
                            continue
                if use_standby:
                    standby[f.origin] -= 1
                    o.standby = True
                    crew_proj = max(plan_dep, sim.now + cfg.standby_callout)
                    duty_start[crew] = max(plan_dep, crew_proj) - cfg.report_time
                proj_dep = max(plan_dep, crew_proj)
                curfew_o = cfg.curfew.get(f.origin, math.inf)
                curfew_d = cfg.curfew.get(f.dest, math.inf)
                if proj_dep > curfew_o or proj_dep + f.block > curfew_d:
                    for g in legs[i:]:
                        cancel(g, "curfew")
                    break
                if proj_dep - f.std > cfg.cancel_threshold:
                    n = rest_to_cancel(legs, i)
                    for g in legs[i : i + n]:
                        cancel(g, "delay")
                    i += n
                    continue
                # crew: wait for the real inbound crew unless a standby took over
                crew_ready = crew_proj if use_standby else -math.inf
                prev = crew_prev.get(f.id)
                if not use_standby and prev is not None:
                    sig = arrived[prev.id]
                    if not sig.triggered:
                        yield sig
                    crew_ready = float(sig.value) + cfg.min_crew_connect
                start = max(earliest, ready, crew_ready)
                if start > sim.now:
                    yield start - sim.now
                o.reactionary_aircraft = max(0.0, ready - max(earliest, f.std))
                o.reactionary_crew = max(0.0, crew_ready - max(earliest, ready, f.std))
                p = primary[f.id] if st is None or st.etd is None else 0.0
                o.primary = p + max(0.0, earliest - f.std)
                if p > 0:
                    yield p
                if crew:
                    end = pairing_end[crew] + (sim.now - f.std) + cfg.release_time
                    o.crew_risk = end - duty_start[crew] > cfg.duty_limit
                if gate_req is not None:
                    sim.release(gate_req)
                    gate_req = None
                runway = runways.get(f.origin)
                queued = sim.now
                flown = 0.0
                if runway is not None:
                    req = yield sim.request(runway)
                    while True:
                        factor, until = capacity(f.origin, sim.now)
                        if factor > 0:
                            break
                        yield until - sim.now
                    o.runway = sim.now - queued
                    flown = 60.0 / (base_rate[f.origin] * factor)
                    _release_later(req, flown)
                o.dep = sim.now
                o.dep_delay = o.dep - f.std
                projected_arr[f.id] = o.dep + f.block
                arrival = sim.process(
                    arrive(f, o, o.dep + max(flown, block[f.id]), queue=True), name=f"arr-{f.id}"
                )
            # wait for the aircraft - unless operations control swaps a spare in for the next leg
            nxt = legs[i + 1] if i + 1 < len(legs) else None
            swap = None
            if nxt is not None and spares.get(nxt.origin) and o.dep >= now0 - 1e-9:
                swap = swap_signal[nxt.id] = sim.signal(f"swap.{nxt.id}")
                sim.process(controller(f, nxt), name=f"ctl-{nxt.id}")
                if not arrival.triggered:
                    yield sim.any_of(arrival, swap)
            if swap is not None and swap.triggered:
                k, spare_ready = swap.value
                if arrival.triggered:
                    landed = arrival.value
                    spares[f.dest][k] = landed[1]
                    if landed[0] is not None:
                        _release_later(landed[0], max(0.0, landed[1] - sim.now))
                else:
                    released[f.id] = k
                ready, gate_req, spare_next = spare_ready, None, True
            else:
                landed = yield arrival
                gate_req, ready = landed
            i += 1
        if gate_req is not None:
            yield max(0.0, ready - sim.now)
            sim.release(gate_req)

    for name, legs in schedule.rotations.items():
        tail_process[name] = sim.process(tail(name, legs), name=f"tail-{name}")

    def finish(sim: Simulation) -> None:
        metrics, misconnected = summarize(schedule, cfg, outcomes)
        for k, v in metrics.items():
            sim.metrics.set(k, v)
        result["flights"] = outcomes
        result["misconnected"] = misconnected
        result["metrics"] = metrics

    sim.on_finish(finish)


def summarize(
    schedule: Schedule, cfg: OpsConfig, outcomes: Mapping[str, FlightOutcome]
) -> tuple[dict[str, float], dict[tuple[str, str], int]]:
    operated = [o for o in outcomes.values() if not o.cancelled]
    n_op = len(operated)
    arr_delay = sum(max(0.0, o.arr_delay) for o in operated)
    on_time_arr = sum(1 for o in operated if o.arr_delay <= cfg.on_time)
    on_time_dep = sum(1 for o in operated if o.dep_delay <= cfg.on_time)
    prim = sum(o.primary for o in operated)
    react = sum(o.reactionary_aircraft + o.reactionary_crew for o in operated)
    rwy = sum(o.runway for o in operated)
    total_cause = prim + react + rwy
    misconnected: dict[tuple[str, str], int] = {}
    connecting = 0
    for c in schedule.connections:
        connecting += c.pax
        i, o = outcomes.get(c.inbound), outcomes.get(c.outbound)
        if i is None or o is None:
            continue
        if i.cancelled or o.cancelled or o.dep - i.arr < cfg.mct:
            misconnected[(c.inbound, c.outbound)] = c.pax
    missed = sum(misconnected.values())
    comp = 0.0
    if cfg.eu261:
        for o in outcomes.values():
            f = schedule.by_id[o.id]
            comp += eu261_compensation(f, o.arr_delay, o.cancelled)
    cancelled = len(outcomes) - n_op
    by_reason: dict[str, int] = {}
    for o in outcomes.values():
        if o.cancelled:
            by_reason[o.reason] = by_reason.get(o.reason, 0) + 1
    gate_waits = [o.gate_wait for o in operated if o.gate_wait > 0]
    last = max((o.arr for o in operated), default=math.nan)
    cost = (
        cfg.delay_cost_per_minute * arr_delay
        + cfg.cancel_cost * cancelled
        + cfg.misconnect_cost_per_pax * missed
        + comp
    )
    metrics = {
        "flights": float(len(outcomes)),
        "operated": float(n_op),
        "cancelled": float(cancelled),
        "cancelled.delay": float(by_reason.get("delay", 0)),
        "cancelled.crew": float(by_reason.get("crew", 0)),
        "cancelled.curfew": float(by_reason.get("curfew", 0)),
        "otp": on_time_arr / n_op if n_op else math.nan,
        "otp_departure": on_time_dep / n_op if n_op else math.nan,
        "arrival_delay.mean": arr_delay / n_op if n_op else math.nan,
        "delay_minutes": arr_delay,
        "reactionary_share": react / total_cause if total_cause > 0 else 0.0,
        "runway_share": rwy / total_cause if total_cause > 0 else 0.0,
        "connecting_pax": float(connecting),
        "misconnected_pax": float(missed),
        "misconnect_rate": missed / connecting if connecting else 0.0,
        "spare_swaps": float(sum(o.spare for o in outcomes.values())),
        "standby_used": float(sum(o.standby for o in outcomes.values())),
        "crew_risk_flights": float(sum(o.crew_risk for o in operated)),
        "gate_waits": float(len(gate_waits)),
        "gate_wait.mean": sum(gate_waits) / len(gate_waits) if gate_waits else 0.0,
        "eu261_compensation": comp,
        "cost": cost,
        "last_arrival": last,
    }
    return metrics, misconnected


def simulate_day(
    schedule: Schedule,
    config: OpsConfig | None = None,
    *,
    seed: int | None = 0,
    weather: Sequence[WeatherEvent] = (),
    state: OpsState | None = None,
    actions: Sequence[Action] = (),
) -> DayOutcome:
    """Simulate one day and return per-flight outcomes and network metrics.

    >>> from simulsi.aviation import Schedule
    >>> day = simulate_day(Schedule.synthetic(tails=4), seed=1)
    >>> 0 <= day.metrics["otp"] <= 1
    True
    """
    cfg = config or OpsConfig()
    plan, cancelled = apply_actions(schedule, actions) if actions else (schedule, set())
    if state is not None:
        state.check(plan)
    out: dict[str, Any] = {}
    sim = Simulation(seed=seed, name="airline-day", record_series=False, keep_values=False)
    build_day(sim, plan, cfg, weather=weather, state=state, cancelled=cancelled, out=out)
    result = sim.run()
    if result.warnings:
        out.setdefault("metrics", {})["warnings"] = float(len(result.warnings))
    return DayOutcome(out["flights"], out["misconnected"], out["metrics"], out["weather"])


def network_model(
    schedule: Schedule,
    config: OpsConfig | None = None,
    *,
    weather: Sequence[WeatherEvent] = (),
    hub: str | None = None,
) -> Model:
    """The day as a simulsi :class:`~simulsi.Model`, for experiments, optimisation and Pareto search.

    Parameters: ``spares`` and ``standby_crews`` (at the hub),
    ``swap_threshold``, ``cancel_threshold``, ``min_turn``, ``delay_scale``
    (multiplies the chance of a primary delay) and ``weather`` (0/1).
    """
    cfg0 = config or OpsConfig()
    schedule.check()
    hub = hub or _hub(schedule)

    def build(sim: Simulation, p: Params) -> None:
        delays = cfg0.delays
        scaled = type(delays)(
            prob=min(1.0, delays.prob * p.delay_scale),
            mean=delays.mean,
            block_cv=delays.block_cv,
            table={k: (min(1.0, v[0] * p.delay_scale), v[1]) for k, v in delays.table.items()},
            block_bias=delays.block_bias,
            predictor=delays.predictor,
        )
        cfg = cfg0.replace(
            spares={**cfg0.spares, hub: p.spares},
            standby_crews={**cfg0.standby_crews, hub: p.standby_crews},
            swap_threshold=p.swap_threshold,
            cancel_threshold=p.cancel_threshold,
            min_turn=p.min_turn,
            delays=scaled,
        )
        build_day(sim, schedule, cfg, weather=weather if p.weather else ())

    return Model(
        build,
        name="airline_network",
        version="1",
        description=f"{len(schedule)} flights, {len(schedule.rotations)} tails, hub {hub}",
        parameters=[
            Parameter(
                "spares",
                cfg0.spares.get(hub, 0),
                "int",
                low=0,
                description=f"spare aircraft at {hub}",
            ),
            Parameter(
                "standby_crews",
                cfg0.standby_crews.get(hub, 0),
                "int",
                low=0,
                description=f"standby crews at {hub}",
            ),
            Parameter("swap_threshold", cfg0.swap_threshold, "float", low=0, unit="min"),
            Parameter("cancel_threshold", cfg0.cancel_threshold, "float", low=0, unit="min"),
            Parameter("min_turn", cfg0.min_turn, "float", low=1, unit="min"),
            Parameter(
                "delay_scale",
                1.0,
                "float",
                low=0,
                description="multiplies the primary-delay chance",
            ),
            Parameter("weather", 1 if weather else 0, "int", low=0, high=1),
        ],
        outputs=["otp", "cancelled", "misconnected_pax", "delay_minutes", "cost"],
    )
