"""Passenger-side decisions: compensation exposure, overbooking and check-in staffing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from simulsi.aviation.schedule import Flight, format_time
from simulsi.core.simulation import Simulation
from simulsi.errors import ConfigError
from simulsi.models.queueing import erlang_c


def eu261_amount(block_minutes: float) -> float:
    """EU261 compensation per passenger (EUR), with block time standing in for distance.

    Short-haul (up to ~1 500 km, about 2.5 h) EUR 250, medium (up to
    ~3 500 km, about 4.5 h) EUR 400, long-haul EUR 600.
    """
    if block_minutes <= 150:
        return 250.0
    if block_minutes <= 270:
        return 400.0
    return 600.0


def eu261_compensation(flight: Flight, arrival_delay: float, cancelled: bool) -> float:
    """Compensation owed for one flight: arrivals 3 h+ late, or short-notice cancellations.

    Extraordinary circumstances (weather, ATC) are not modelled, so this is
    an upper bound on exposure.
    """
    if cancelled or (not math.isnan(arrival_delay) and arrival_delay >= 180):
        return flight.pax * eu261_amount(flight.block)
    return 0.0


@dataclass
class OverbookingResult:
    capacity: int
    show_rate: float
    best_limit: int
    table: list[dict[str, float]]

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        return format_table(self.table)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "show_rate": self.show_rate,
            "best_limit": self.best_limit,
            "table": self.table,
        }


def _binom_pmf(n: int, p: float) -> list[float]:
    if p <= 0:
        return [1.0] + [0.0] * n
    if p >= 1:
        return [0.0] * n + [1.0]
    lp, lq = math.log(p), math.log1p(-p)
    return [
        math.exp(
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq
        )
        for k in range(n + 1)
    ]


def overbooking(
    capacity: int,
    show_rate: float,
    *,
    fare: float = 150.0,
    denied_boarding_cost: float = 600.0,
    max_extra: int | None = None,
) -> OverbookingResult:
    """The booking limit that maximises expected revenue minus denied-boarding cost.

    Passengers show up independently with probability ``show_rate``. For
    each limit from ``capacity`` upwards the table gives expected empty
    seats, expected denied boardings, the chance of denying anyone and the
    expected net revenue.

    >>> overbooking(180, 0.92).best_limit > 180
    True
    """
    if capacity < 1:
        raise ConfigError("capacity must be >= 1")
    if not 0 < show_rate <= 1:
        raise ConfigError("show_rate must be in (0, 1]")
    extra = max_extra if max_extra is not None else max(5, int(capacity * (1 - show_rate) * 2.5))
    rows = []
    for limit in range(capacity, capacity + extra + 1):
        pmf = _binom_pmf(limit, show_rate)
        flown = sum(min(k, capacity) * pk for k, pk in enumerate(pmf))
        denied = sum((k - capacity) * pk for k, pk in enumerate(pmf) if k > capacity)
        p_deny = sum(pk for k, pk in enumerate(pmf) if k > capacity)
        rows.append(
            {
                "limit": float(limit),
                "empty_seats": capacity - flown,
                "denied": denied,
                "p_any_denied": p_deny,
                "net_revenue": fare * flown - denied_boarding_cost * denied,
            }
        )
    best = max(rows, key=lambda r: r["net_revenue"])
    return OverbookingResult(capacity, show_rate, int(best["limit"]), rows)


@dataclass
class StaffingPlan:
    """Counters per time slot for a check-in or security area."""

    slot_minutes: float
    start: float
    arrivals: list[float]
    staff: list[int]
    expected_wait: list[float]
    simulated_wait: list[float]

    def table(self) -> list[dict[str, Any]]:
        return [
            {
                "slot": format_time(self.start + k * self.slot_minutes),
                "arrivals": round(a, 1),
                "staff": s,
                "expected_wait": round(w, 1),
                "simulated_wait": round(sw, 1),
            }
            for k, (a, s, w, sw) in enumerate(
                zip(self.arrivals, self.staff, self.expected_wait, self.simulated_wait, strict=True)
            )
        ]

    @property
    def staff_hours(self) -> float:
        return sum(self.staff) * self.slot_minutes / 60

    def format(self) -> str:
        from simulsi.analysis.report import format_table

        return format_table(self.table())


def passenger_arrivals(
    flights: Sequence[Flight],
    *,
    slot_minutes: float = 15.0,
    earliest: float = 180.0,
    latest: float = 45.0,
    peak: float = 100.0,
) -> tuple[float, list[float]]:
    """Expected check-in arrivals per slot: each passenger turns up between ``earliest``
    and ``latest`` minutes before departure (triangular, most likely ``peak``)."""
    if not flights:
        raise ConfigError("no departing flights")
    lo, mode, hi = latest, peak, earliest
    start = math.floor((min(f.std for f in flights) - hi) / slot_minutes) * slot_minutes
    end = max(f.std for f in flights) - lo
    n = max(1, math.ceil((end - start) / slot_minutes))
    out = [0.0] * n

    def cdf(x: float) -> float:  # triangular CDF of "minutes before departure"
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        if x <= mode:
            return (x - lo) ** 2 / ((hi - lo) * (mode - lo))
        return 1 - (hi - x) ** 2 / ((hi - lo) * (hi - mode))

    for f in flights:
        for k in range(n):
            a, b = start + k * slot_minutes, start + (k + 1) * slot_minutes
            out[k] += f.pax * (cdf(f.std - a) - cdf(f.std - b))
    return start, out


def checkin_staffing(
    flights: Sequence[Flight],
    *,
    service_minutes: float = 2.5,
    target_wait: float = 10.0,
    slot_minutes: float = 15.0,
    max_staff: int = 60,
    seed: int = 0,
    replications: int = 5,
    corrections: int = 8,
) -> StaffingPlan:
    """Counters needed per slot to keep the mean wait under ``target_wait`` minutes.

    Staffing per slot starts from the Erlang C formula on that slot's
    arrival rate (the usual "stationary independent period" method). Queues
    carry over from busy slots into quiet ones, which that method misses, so
    a simulation with the time-varying staffing then checks every slot and
    adds counters where (and just before where) the simulated wait is too
    long, up to ``corrections`` rounds. ``expected_wait`` is the Erlang C
    figure for the first guess, ``simulated_wait`` the final check.
    """
    start, arrivals = passenger_arrivals(flights, slot_minutes=slot_minutes)
    mu = 1.0 / service_minutes
    staff: list[int] = []
    expected: list[float] = []
    for a in arrivals:
        lam = a / slot_minutes
        if lam <= 0:
            staff.append(1)
            expected.append(0.0)
            continue
        c = max(1, math.ceil(lam / mu))
        while c <= max_staff:
            if lam / (c * mu) < 1 and erlang_c(lam, mu, c)["mean_wait"] <= target_wait:
                break
            c += 1
        staff.append(min(c, max_staff))
        expected.append(
            erlang_c(lam, mu, staff[-1])["mean_wait"] if lam < staff[-1] * mu else math.inf
        )
    sim_wait = _simulate_staffing(
        start, arrivals, staff, slot_minutes, service_minutes, seed, replications
    )
    for _ in range(corrections):
        short = [k for k, w in enumerate(sim_wait) if w > target_wait and staff[k] < max_staff]
        if not short:
            break
        for k in short:
            staff[k] += 1
            if k > 0 and staff[k - 1] < max_staff:
                staff[k - 1] += 1  # the queue was built up in the slot before
        sim_wait = _simulate_staffing(
            start, arrivals, staff, slot_minutes, service_minutes, seed, replications
        )
    return StaffingPlan(slot_minutes, start, arrivals, staff, expected, sim_wait)


def _simulate_staffing(
    start: float,
    arrivals: Sequence[float],
    staff: Sequence[int],
    slot: float,
    service: float,
    seed: int,
    replications: int,
) -> list[float]:
    totals = [0.0] * len(arrivals)
    counts = [0] * len(arrivals)
    for r in range(replications):
        _staffing_run(start, arrivals, staff, slot, service, seed + r, totals, counts)
    return [t / c if c else 0.0 for t, c in zip(totals, counts, strict=True)]


def _staffing_run(
    start: float,
    arrivals: Sequence[float],
    staff: Sequence[int],
    slot: float,
    service: float,
    seed: int,
    totals: list[float],
    counts: list[int],
) -> None:
    """One simulated day at the desks; adds each slot's waits to ``totals``/``counts``."""
    sim = Simulation(seed=seed, start=start, record_series=False, keep_values=False)
    desk = sim.resource("desk", staff[0])
    rs_a, rs_s = sim.stream("arrivals"), sim.stream("service")

    def shifts() -> Any:
        for s in staff:
            desk.set_capacity(s)
            yield slot

    def pax(k: int) -> Any:
        t0 = sim.now
        req = yield sim.request(desk)
        totals[k] += sim.now - t0
        counts[k] += 1
        yield rs_s.exponential(service)
        sim.release(req)

    def source() -> Any:
        for k, a in enumerate(arrivals):
            end = start + (k + 1) * slot
            rate = a / slot
            if rate <= 0:
                yield max(0.0, end - sim.now)
                continue
            while True:
                gap = rs_a.exponential(1.0 / rate)
                if sim.now + gap >= end:
                    yield end - sim.now
                    break
                yield gap
                sim.process(pax(k))

    sim.process(shifts())
    sim.process(source())
    sim.run()
