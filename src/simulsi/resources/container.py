"""Containers: a level of a bulk or countable quantity (fuel, stock, cash, parts)."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

from simulsi.core.trace import LogRecord
from simulsi.errors import CapacityError, ResourceUsageError
from simulsi.metrics.collectors import Counter, Tally, TimeWeighted
from simulsi.processes.process import Waitable

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation


class ContainerPut(Waitable):
    __slots__ = ("amount", "requested_at")

    def __init__(self, sim: Simulation, amount: float) -> None:
        super().__init__(sim)
        self.amount = amount
        self.requested_at = sim.now


class ContainerGet(Waitable):
    __slots__ = ("amount", "requested_at")

    def __init__(self, sim: Simulation, amount: float) -> None:
        super().__init__(sim)
        self.amount = amount
        self.requested_at = sim.now


class Container:
    """A level between 0 and ``capacity`` that processes add to and take from.

    ``yield tank.get(30)`` waits until 30 units are available and removes
    them; ``yield tank.put(50)`` waits until there is room. Waiting gets (and
    waiting puts) are served strictly in arrival order, so a large request is
    not starved by a stream of small ones.

    Metrics (``container.<name>.``): ``level.mean``, ``level.min``,
    ``level.max``, ``level_final``, ``put_amount``, ``get_amount``,
    ``get_wait.mean``, ``put_wait.mean`` and ``stockouts`` (gets that had to
    wait).
    """

    def __init__(
        self,
        name: str = "container",
        capacity: float = math.inf,
        *,
        init: float = 0.0,
        sim: Simulation | None = None,
    ) -> None:
        if not (capacity > 0):
            raise CapacityError(f"container capacity must be > 0, got {capacity!r}")
        if not (0 <= init <= capacity):
            raise CapacityError(f"initial level must be in [0, {capacity}], got {init!r}")
        self.name = name
        self.capacity = float(capacity)
        self.level = float(init)
        self._getters: deque[ContainerGet] = deque()
        self._putters: deque[ContainerPut] = deque()
        self.sim: Simulation | None = None
        if sim is not None:
            sim.add_container(self)

    def _bind(self, sim: Simulation) -> None:
        if self.sim is not None:
            if self.sim is not sim:
                raise ResourceUsageError(f"container {self.name!r} belongs to another simulation")
            return
        self.sim = sim
        series = sim.metrics.record_series
        self.level_stat = TimeWeighted(
            f"{self.name}.level", sim.clock, self.level, record_series=series
        )
        self.get_waits = Tally(f"{self.name}.get_wait", keep_values=sim.metrics.keep_values)
        self.put_waits = Tally(f"{self.name}.put_wait", keep_values=sim.metrics.keep_values)
        self.put_amount = Counter(f"{self.name}.put_amount", sim.clock)
        self.get_amount = Counter(f"{self.name}.get_amount", sim.clock)
        self.stockouts = Counter(f"{self.name}.stockouts", sim.clock)

    def _require_sim(self) -> Simulation:
        if self.sim is None:
            raise ResourceUsageError(
                f"container {self.name!r} is not attached; use sim.add_container()"
            )
        return self.sim

    @staticmethod
    def _check_amount(amount: float) -> float:
        x = float(amount)
        if not (x > 0) or math.isinf(x):
            raise ValueError(f"amount must be a positive finite number, got {amount!r}")
        return x

    # -- operations ------------------------------------------------------------

    def put(self, amount: float) -> ContainerPut:
        """Add ``amount``; the returned waitable triggers once it fits."""
        sim = self._require_sim()
        x = self._check_amount(amount)
        if x > self.capacity:
            raise CapacityError(f"cannot put {x} into {self.name!r} with capacity {self.capacity}")
        req = ContainerPut(sim, x)
        self._putters.append(req)
        self._settle()
        return req

    def get(self, amount: float) -> ContainerGet:
        """Take ``amount``; the returned waitable triggers once enough is available."""
        sim = self._require_sim()
        x = self._check_amount(amount)
        if x > self.capacity:
            raise CapacityError(f"cannot get {x} from {self.name!r} with capacity {self.capacity}")
        req = ContainerGet(sim, x)
        if self._getters or x > self.level:
            self.stockouts.increment()
        self._getters.append(req)
        self._settle()
        return req

    def cancel(self, req: ContainerGet | ContainerPut) -> bool:
        """Withdraw a pending get or put (e.g. after a timeout)."""
        pool: deque[Any] = self._getters if isinstance(req, ContainerGet) else self._putters
        if req.triggered or req not in pool:
            return False
        pool.remove(req)
        req._value = None
        self._settle()
        return True

    def _settle(self) -> None:
        sim = self._require_sim()
        progress = True
        while progress:
            progress = False
            if self._putters and self.level + self._putters[0].amount <= self.capacity:
                req = self._putters.popleft()
                self._change(sim, req.amount, "container.put")
                self.put_amount.increment(req.amount)
                self.put_waits.observe(sim.now - req.requested_at)
                req.succeed(req.amount)
                progress = True
            if self._getters and self._getters[0].amount <= self.level:
                greq = self._getters.popleft()
                self._change(sim, -greq.amount, "container.get")
                self.get_amount.increment(greq.amount)
                self.get_waits.observe(sim.now - greq.requested_at)
                greq.succeed(greq.amount)
                progress = True

    def _change(self, sim: Simulation, delta: float, kind: str) -> None:
        old = self.level
        # Snap tiny floating-point residue so the level never drifts outside [0, capacity].
        self.level = min(self.capacity, max(0.0, old + delta))
        self.level_stat.record(self.level)
        if sim.log is not None:
            sim.log.append(
                LogRecord(
                    sim.now,
                    kind,
                    resource=self.name,
                    old_state=f"{old:g}",
                    new_state=f"{self.level:g}",
                    metadata={"amount": abs(delta)},
                )
            )

    # -- statistics --------------------------------------------------------------

    def reset_stats(self) -> None:
        self.level_stat.reset()
        self.get_waits.reset()
        self.put_waits.reset()
        for c in (self.put_amount, self.get_amount, self.stockouts):
            c.reset()

    def summary(self) -> dict[str, Any]:
        if self.sim is None:
            return {"bound": False}
        return {
            "level_now": self.level,
            "capacity": None if math.isinf(self.capacity) else self.capacity,
            "mean_level": self.level_stat.mean,
            "min_level": self.level_stat.min,
            "max_level": self.level_stat.max,
            "put_amount": self.put_amount.value,
            "get_amount": self.get_amount.value,
            "stockouts": self.stockouts.value,
            "get_wait": self.get_waits.summary(),
            "put_wait": self.put_waits.summary(),
            "waiting_getters": len(self._getters),
            "waiting_putters": len(self._putters),
        }

    def flat_metrics(self) -> dict[str, float]:
        if self.sim is None:
            return {}
        s = self.summary()
        p = f"container.{self.name}."
        return {
            p + "level.mean": float(s["mean_level"]),
            p + "level.min": float(s["min_level"]),
            p + "level.max": float(s["max_level"]),
            p + "level_final": float(s["level_now"]),
            p + "put_amount": float(s["put_amount"]),
            p + "get_amount": float(s["get_amount"]),
            p + "stockouts": float(s["stockouts"]),
            p + "get_wait.mean": float(s["get_wait"]["mean"]),
            p + "put_wait.mean": float(s["put_wait"]["mean"]),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "capacity": None if math.isinf(self.capacity) else self.capacity,
            "waiting_getters": [g.amount for g in list(self._getters)[:50]],
            "waiting_putters": [p.amount for p in list(self._putters)[:50]],
        }

    def __repr__(self) -> str:
        return f"Container({self.name!r}, level={self.level:g}, capacity={self.capacity:g})"
