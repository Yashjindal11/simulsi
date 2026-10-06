"""Shared resources with finite capacity (servers, machines, gates, staff ...)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simulsi.core.trace import LogRecord
from simulsi.errors import CapacityError, ResourceUsageError
from simulsi.metrics.collectors import Counter, Tally, TimeWeighted
from simulsi.processes.process import Process, Waitable
from simulsi.queues.discipline import Discipline, OrderedBuffer

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation
    from simulsi.entities.entity import Entity


@dataclass(frozen=True)
class Preempted:
    """``Interrupt.cause`` received by a process whose unit was taken by a higher-priority request."""

    resource: str
    by: Request
    usage_since: float


class Request(Waitable):
    """A claim on one unit of a :class:`Resource`.

    Yielding the request waits until it is granted; the yield returns the
    request itself, so ``req = yield sim.request(server)`` works. With a
    ``patience`` the request is withdrawn (reneged) if not granted in time and
    the yield returns with ``req.granted == False``.
    """

    __slots__ = (
        "_patience_event",
        "entity",
        "granted_at",
        "owner",
        "preempted",
        "priority",
        "released_at",
        "reneged",
        "requested_at",
        "resource",
    )

    def __init__(
        self,
        resource: Resource,
        sim: Simulation,
        *,
        priority: float = 0,
        entity: Entity | None = None,
        owner: Process | None = None,
    ) -> None:
        super().__init__(sim)
        self.resource = resource
        self.priority = priority
        self.entity = entity
        self.owner = owner
        self.requested_at = sim.now
        self.granted_at: float | None = None
        self.released_at: float | None = None
        self.reneged = False
        self.preempted = False
        self._patience_event: Any = None

    @property
    def granted(self) -> bool:
        return self.granted_at is not None

    @property
    def waiting_time(self) -> float | None:
        return None if self.granted_at is None else self.granted_at - self.requested_at

    def cancel(self) -> None:
        """Withdraw a request that has not been granted yet."""
        self.resource._withdraw(self, reneged=False)

    def release(self) -> None:
        self.resource.release(self)

    def __repr__(self) -> str:
        state = "granted" if self.granted else ("reneged" if self.reneged else "waiting")
        if self.preempted:
            state = "preempted"
        return f"<Request {self.resource.name!r} {state}>"


class Resource:
    """A pool of ``capacity`` identical units that entities acquire and release.

    Statistics gathered automatically: utilization (time-average busy units /
    capacity), availability (fraction of capacity not down), queue length
    (time-weighted), waiting time per granted request, and counts of
    requests, grants, releases and reneges.

    Capacity can change during a run (:meth:`set_capacity`) and units can be
    taken down and repaired (:meth:`fail` / :meth:`repair`). Units already in
    use when capacity drops finish normally unless ``interrupt=True``.

    With ``preemptive=True`` (requires the ``"priority"`` discipline) a request
    that finds no free unit evicts the lowest-priority user if it is strictly
    more important (lower value). The evicted process receives
    :class:`~simulsi.errors.Interrupt` with a :class:`Preempted` cause; its unit
    is already released, so it typically re-requests the remaining work.
    """

    def __init__(
        self,
        name: str,
        capacity: int = 1,
        *,
        discipline: Discipline = "fifo",
        preemptive: bool = False,
        sim: Simulation | None = None,
    ) -> None:
        self.name = name
        self._check_capacity(capacity)
        if preemptive and discipline != "priority":
            raise ValueError("preemptive resources need discipline='priority'")
        self.capacity = int(capacity)
        self.discipline = discipline
        self.preemptive = preemptive
        self._waiting: OrderedBuffer[Request] = OrderedBuffer(discipline)
        self.users: list[Request] = []
        self.down = 0
        self.sim: Simulation | None = None
        if sim is not None:
            sim.add_resource(self)

    @staticmethod
    def _check_capacity(capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise CapacityError(f"capacity must be a non-negative integer, got {capacity!r}")

    # -- binding -------------------------------------------------------------

    def _bind(self, sim: Simulation) -> None:
        if self.sim is not None:
            if self.sim is not sim:
                raise ResourceUsageError(
                    f"resource {self.name!r} belongs to another simulation; "
                    "create resources inside the model function for each run"
                )
            return
        self.sim = sim
        clock = sim.clock
        series = sim.metrics.record_series
        self.busy = TimeWeighted(f"{self.name}.busy", clock, 0, record_series=series)
        self.queue_length = TimeWeighted(f"{self.name}.queue", clock, 0, record_series=series)
        self.capacity_level = TimeWeighted(
            f"{self.name}.capacity", clock, self.capacity, record_series=series
        )
        self.effective_level = TimeWeighted(
            f"{self.name}.effective", clock, self.effective_capacity, record_series=series
        )
        self.wait_times = Tally(f"{self.name}.wait", keep_values=sim.metrics.keep_values)
        self.requests = Counter(f"{self.name}.requests", clock)
        self.grants = Counter(f"{self.name}.grants", clock)
        self.releases = Counter(f"{self.name}.releases", clock)
        self.reneges = Counter(f"{self.name}.reneges", clock)
        self.failures = Counter(f"{self.name}.failures", clock)
        self.preemptions = Counter(f"{self.name}.preemptions", clock)

    def _require_sim(self) -> Simulation:
        if self.sim is None:
            raise ResourceUsageError(
                f"resource {self.name!r} is not attached to a simulation; use sim.add_resource()"
            )
        return self.sim

    # -- state ---------------------------------------------------------------

    @property
    def effective_capacity(self) -> int:
        return max(0, self.capacity - self.down)

    @property
    def in_use(self) -> int:
        return len(self.users)

    @property
    def available(self) -> int:
        return max(0, self.effective_capacity - self.in_use)

    @property
    def waiting(self) -> list[Request]:
        return list(self._waiting)

    @property
    def queue_size(self) -> int:
        return len(self._waiting)

    # -- acquire / release -----------------------------------------------------

    def request(
        self,
        *,
        priority: float = 0,
        entity: Entity | None = None,
        patience: float | None = None,
    ) -> Request:
        sim = self._require_sim()
        owner = sim._active_process
        if entity is None and owner is not None:
            entity = owner.entity
        req = Request(self, sim, priority=priority, entity=entity, owner=owner)
        self.requests.increment()
        if sim.log is not None:
            sim.log.append(
                LogRecord(
                    sim.now,
                    "resource.request",
                    entity=_eid(entity),
                    resource=self.name,
                    metadata={"priority": priority},
                )
            )
        if not self._waiting and len(self.users) < self.capacity - self.down:
            self._grant(req)
            return req
        self._waiting.push(req, priority, req)
        self.queue_length.record(len(self._waiting))
        if patience is not None:
            if patience < 0 or math.isnan(patience):
                raise ValueError("patience must be >= 0")
            req._patience_event = sim._schedule_internal(
                lambda: self._withdraw(req, reneged=True), "_renege", delay=patience
            )
        if self.preemptive and self.users:
            victim = max(self.users, key=lambda r: (r.priority, r.granted_at or 0.0))
            if victim.priority > priority:
                self._preempt(victim, req)
                self._dispatch()
        return req

    def _preempt(self, victim: Request, by: Request) -> None:
        sim = self._require_sim()
        self.users.remove(victim)
        victim.released_at = sim.now
        victim.preempted = True
        owner = victim.owner
        if owner is not None and victim in owner.held:
            owner.held.remove(victim)
        self.busy.record(len(self.users))
        self.preemptions.increment()
        if sim.log is not None:
            sim.log.append(
                LogRecord(
                    sim.now,
                    "resource.preempt",
                    entity=_eid(victim.entity),
                    resource=self.name,
                    metadata={
                        "by": _eid(by.entity),
                        "held": sim.now
                        - (sim.now if victim.granted_at is None else victim.granted_at),
                    },
                )
            )
        if owner is not None and owner.is_alive and owner is not sim._active_process:
            owner.interrupt(
                Preempted(
                    self.name, by, sim.now if victim.granted_at is None else victim.granted_at
                )
            )

    def _grant(self, req: Request) -> None:
        sim = self.sim
        assert sim is not None  # requests are only issued by attached resources
        req.granted_at = sim.clock._now
        self.users.append(req)
        self.busy.record(len(self.users))
        self.grants.increment()
        self.wait_times.observe(req.granted_at - req.requested_at)
        if req._patience_event is not None:
            sim.cancel(req._patience_event)
            req._patience_event = None
        if req.owner is not None:
            req.owner.held.append(req)
        if sim.log is not None:
            sim.log.append(
                LogRecord(
                    sim.now,
                    "resource.acquire",
                    entity=_eid(req.entity),
                    resource=self.name,
                    metadata={"wait": req.granted_at - req.requested_at, "in_use": self.in_use},
                )
            )
        req.succeed(req)

    def _withdraw(self, req: Request, *, reneged: bool) -> None:
        if req.triggered or not self._waiting.remove(req):
            return
        sim = self._require_sim()
        self.queue_length.record(len(self._waiting))
        if req._patience_event is not None:
            sim.cancel(req._patience_event)
            req._patience_event = None
        if reneged:
            req.reneged = True
            self.reneges.increment()
            if sim.log is not None:
                sim.log.append(
                    LogRecord(
                        sim.now, "resource.renege", entity=_eid(req.entity), resource=self.name
                    )
                )
            req.succeed(req)
        else:
            req._value = req  # mark finished without waking anyone
        # Removing a request can unblock others only under custom disciplines, but it is cheap.
        self._dispatch()

    def release(self, req: Request) -> None:
        if req.resource is not self:
            raise ResourceUsageError(f"{req!r} was not issued by resource {self.name!r}")
        if req.preempted:
            return  # the unit was already taken away; releasing is a harmless no-op
        if req.released_at is not None:
            raise ResourceUsageError(f"{req!r} was already released")
        if not req.granted:
            raise ResourceUsageError(f"{req!r} was never granted; cancel() it instead")
        sim = self.sim
        assert sim is not None  # granted, so attached
        self.users.remove(req)
        req.released_at = sim.clock._now
        owner = req.owner
        if owner is not None and req in owner.held:
            owner.held.remove(req)
        self.busy.record(len(self.users))
        self.releases.increment()
        if sim.log is not None:
            sim.log.append(
                LogRecord(
                    sim.now,
                    "resource.release",
                    entity=_eid(req.entity),
                    resource=self.name,
                    metadata={
                        "held": sim.now - (sim.now if req.granted_at is None else req.granted_at)
                    },
                )
            )
        self._dispatch()

    def _dispatch(self) -> None:
        waiting = self._waiting
        while waiting and len(self.users) < self.capacity - self.down:
            req = waiting.pop()
            self.queue_length.record(len(waiting))
            self._grant(req)

    # -- capacity and downtime ---------------------------------------------------

    def set_capacity(self, capacity: int) -> None:
        self._check_capacity(capacity)
        sim = self._require_sim()
        old = self.capacity
        self.capacity = int(capacity)
        self.capacity_level.record(self.capacity)
        self.effective_level.record(self.effective_capacity)
        if sim.log is not None:
            sim.log.append(
                LogRecord(
                    sim.now,
                    "resource.capacity",
                    resource=self.name,
                    old_state=str(old),
                    new_state=str(self.capacity),
                )
            )
        self._dispatch()

    def fail(self, units: int | None = None, *, interrupt: bool = False, cause: Any = None) -> None:
        """Take ``units`` (default: all) out of service.

        With ``interrupt=True``, the most recently granted users beyond the
        remaining capacity are interrupted (their processes receive
        :class:`~simulsi.errors.Interrupt` with ``cause``).
        """
        sim = self._require_sim()
        n = self.capacity - self.down if units is None else int(units)
        if n < 0:
            raise CapacityError("units must be >= 0")
        self.down = min(self.capacity, self.down + n)
        self.failures.increment()
        self.effective_level.record(self.effective_capacity)
        if sim.log is not None:
            sim.log.append(
                LogRecord(
                    sim.now, "resource.down", resource=self.name, metadata={"down": self.down}
                )
            )
        if interrupt:
            excess = self.in_use - self.effective_capacity
            for req in list(reversed(self.users))[: max(0, excess)]:
                if req.owner is not None and req.owner.is_alive:
                    req.owner.interrupt(cause if cause is not None else f"{self.name} failed")

    def repair(self, units: int | None = None) -> None:
        """Return ``units`` (default: all) to service."""
        sim = self._require_sim()
        n = self.down if units is None else int(units)
        if n < 0:
            raise CapacityError("units must be >= 0")
        self.down = max(0, self.down - n)
        self.effective_level.record(self.effective_capacity)
        if sim.log is not None:
            sim.log.append(
                LogRecord(sim.now, "resource.up", resource=self.name, metadata={"down": self.down})
            )
        self._dispatch()

    # -- statistics --------------------------------------------------------------

    def reset_stats(self) -> None:
        for c in (self.busy, self.queue_length, self.capacity_level, self.effective_level):
            c.reset()
        self.wait_times.reset()
        for k in (
            self.requests,
            self.grants,
            self.releases,
            self.reneges,
            self.failures,
            self.preemptions,
        ):
            k.reset()

    @property
    def utilization(self) -> float:
        """Time-average busy units divided by time-average nominal capacity."""
        cap = self.capacity_level.area()
        return self.busy.area() / cap if cap > 0 else math.nan

    @property
    def availability(self) -> float:
        cap = self.capacity_level.area()
        return self.effective_level.area() / cap if cap > 0 else math.nan

    def summary(self) -> dict[str, Any]:
        if self.sim is None:
            return {"capacity": self.capacity, "bound": False}
        eff = self.effective_level.area()
        return {
            "capacity": self.capacity,
            "in_use": self.in_use,
            "queue_length_now": self.queue_size,
            "utilization": self.utilization,
            "utilization_of_available": self.busy.area() / eff if eff > 0 else math.nan,
            "availability": self.availability,
            "mean_busy": self.busy.mean,
            "mean_queue_length": self.queue_length.mean,
            "max_queue_length": self.queue_length.max,
            "wait": self.wait_times.summary(),
            "requests": self.requests.value,
            "grants": self.grants.value,
            "releases": self.releases.value,
            "reneges": self.reneges.value,
            "failures": self.failures.value,
            "preemptions": self.preemptions.value,
            "throughput": self.releases.rate,
        }

    def flat_metrics(self) -> dict[str, float]:
        s = self.summary()
        if not s.get("bound", True):
            return {}
        p = f"resource.{self.name}."
        out = {
            p + k: float(s[k])
            for k in (
                "utilization",
                "availability",
                "mean_busy",
                "mean_queue_length",
                "max_queue_length",
                "requests",
                "grants",
                "releases",
                "reneges",
                "failures",
                "preemptions",
                "throughput",
            )
        }
        out[p + "wait.mean"] = float(s["wait"]["mean"])
        out[p + "wait.max"] = float(s["wait"]["max"])
        out[p + "wait.total"] = float(s["wait"]["sum"])
        out[p + "busy_time"] = self.busy.area()
        out[p + "capacity_time"] = self.capacity_level.area()
        if "p95" in s["wait"]:
            out[p + "wait.p95"] = float(s["wait"]["p95"])
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "down": self.down,
            "in_use": self.in_use,
            "queue_length": self.queue_size,
            "discipline": self.discipline if isinstance(self.discipline, str) else "custom",
            "users": [_eid(r.entity) or (r.owner.name if r.owner else None) for r in self.users],
            "waiting": [
                _eid(r.entity) or (r.owner.name if r.owner else None) for r in self._waiting
            ],
        }

    def __repr__(self) -> str:
        return f"Resource({self.name!r}, capacity={self.capacity}, in_use={self.in_use})"


def _eid(entity: Entity | None) -> str | None:
    return None if entity is None else entity.id
