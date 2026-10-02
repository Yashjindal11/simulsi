"""Failure and disruption primitives.

* :class:`FailureProcess` - a resource alternates between *up* (for a random
  time-to-failure) and *down* (until recovered). Recovery is described by a
  :class:`RecoveryProcess`: a random repair time, optionally needing a unit of
  a repair-crew resource first (so repairs can queue for technicians).
* :class:`ScheduledDisruption` - a deterministic window during which
  something is changed and later restored (capacity cut, demand spike, ...).
* :func:`capacity_reduction` - the most common scheduled disruption.

All randomness comes from named streams, so disruptions are reproducible and
do not disturb other streams (common random numbers stay intact).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simulsi.randomness.distributions import DistributionLike, as_distribution

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation
    from simulsi.processes.process import Process
    from simulsi.resources.resource import Resource


@dataclass
class RecoveryProcess:
    """How a failed resource comes back: repair time, optionally needing a crew unit."""

    time_to_repair: DistributionLike
    crew: Resource | None = None

    def run(self, sim: Simulation, stream_name: str) -> Any:
        dist = as_distribution(self.time_to_repair)
        stream = sim.stream(stream_name)
        if self.crew is None:
            yield max(0.0, float(dist.sample(stream)))
            return
        req = yield sim.request(self.crew)
        yield max(0.0, float(dist.sample(stream)))
        self.crew.release(req)


class FailureProcess:
    """Random failures of a resource.

    Each cycle: wait ``time_to_failure`` (measured from the end of the last
    repair), take ``units`` units down (default: all), recover via
    ``recovery``, bring them back. With ``interrupt=True`` jobs holding the
    failed units are interrupted (see :meth:`Resource.fail`).

    Metrics: ``failure.<name>.count`` (counter) and
    ``failure.<name>.downtime`` (tally of outage durations).
    """

    def __init__(
        self,
        sim: Simulation,
        resource: Resource,
        time_to_failure: DistributionLike,
        recovery: RecoveryProcess | DistributionLike,
        *,
        units: int | None = None,
        interrupt: bool = False,
        name: str | None = None,
        start: bool = True,
    ) -> None:
        self.sim = sim
        self.resource = resource
        self.ttf = as_distribution(time_to_failure)
        self.recovery = (
            recovery if isinstance(recovery, RecoveryProcess) else RecoveryProcess(recovery)
        )
        self.units = units
        self.interrupt = interrupt
        self.name = name or resource.name
        self.process: Process | None = None
        if resource.sim is not sim:
            sim.add_resource(resource)
        if start:
            self.start()

    def start(self) -> Process:
        if self.process is not None:
            raise RuntimeError(f"failure process {self.name!r} already started")
        self.process = self.sim.process(self._run(), name=f"failure:{self.name}")
        return self.process

    def _run(self) -> Any:
        sim, res = self.sim, self.resource
        stream = sim.stream(f"failure:{self.name}")
        while True:
            yield max(0.0, float(self.ttf.sample(stream)))
            before = res.down
            res.fail(self.units, interrupt=self.interrupt, cause=f"{self.name} failed")
            taken = res.down - before
            sim.metrics.increment(f"failure.{self.name}.count")
            t0 = sim.now
            yield from self.recovery.run(sim, f"repair:{self.name}")
            res.repair(taken)
            sim.metrics.observe(f"failure.{self.name}.downtime", sim.now - t0)


class ScheduledDisruption:
    """Apply a change at ``at`` and revert it ``duration`` later.

    >>> # demand spike: double the arrival rate between t=100 and t=160
    >>> # ScheduledDisruption(sim, at=100, duration=60,
    >>> #     apply=lambda: state.update(rate=2 * base), revert=lambda: state.update(rate=base))
    """

    def __init__(
        self,
        sim: Simulation,
        *,
        at: float,
        duration: float | None,
        apply: Callable[[], Any],
        revert: Callable[[], Any] | None = None,
        name: str = "disruption",
    ) -> None:
        if duration is not None and duration < 0:
            raise ValueError("duration must be >= 0")
        self.name = name
        sim.call_at(lambda: self._fire(sim, apply), time=at, label=f"disruption.start:{name}")
        if duration is not None and revert is not None:
            sim.call_at(revert, time=at + duration, label=f"disruption.end:{name}")

    def _fire(self, sim: Simulation, apply: Callable[[], Any]) -> None:
        sim.metrics.increment(f"disruption.{self.name}.count")
        apply()


def capacity_reduction(
    sim: Simulation,
    resource: Resource,
    *,
    at: float,
    duration: float,
    capacity: int,
    name: str | None = None,
) -> ScheduledDisruption:
    """Temporarily set ``resource`` capacity to ``capacity`` during ``[at, at + duration)``."""
    original: list[int] = []

    def apply() -> None:
        original.append(resource.capacity)
        resource.set_capacity(capacity)

    def revert() -> None:
        resource.set_capacity(original.pop())

    return ScheduledDisruption(
        sim,
        at=at,
        duration=duration,
        apply=apply,
        revert=revert,
        name=name or f"{resource.name}.capacity",
    )
