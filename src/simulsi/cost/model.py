"""Cost and revenue modelling on top of simulation metrics.

A :class:`CostModel` is a list of terms, each turning one metric into money:

* fixed costs                 ``fixed("rent", 5000)``
* variable costs / revenue    ``variable("energy", "machine.busy_time", 0.4)``
* resource costs              ``resource("teller", per_capacity_time=30)``
* waiting costs               ``waiting("teller", per_time=2.5)``
* penalties                   ``penalty("sla", "resource.teller.wait.p95", above=10, amount=1000)``
* revenue                     ``revenue("sales", "entity.order.disposed", 12)``

Costs are computed per replication from that replication's metrics, so
experiments get a full distribution of cost/profit (with CIs) for free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from simulsi.errors import ConfigError

Kind = Literal["cost", "revenue"]


@dataclass(frozen=True)
class CostTerm:
    name: str
    kind: Kind
    metric: str | None = None
    rate: float = 0.0
    fixed: float = 0.0
    above: float | None = None
    below: float | None = None
    per_unit: bool = False

    def amount(self, metrics: Mapping[str, float]) -> float:
        if self.metric is None:
            return self.fixed
        if self.metric not in metrics:
            raise KeyError(f"cost term {self.name!r} needs metric {self.metric!r}")
        x = float(metrics[self.metric])
        if math.isnan(x):
            return math.nan
        if self.above is not None or self.below is not None:
            excess = 0.0
            if self.above is not None and x > self.above:
                excess = x - self.above
            elif self.below is not None and x < self.below:
                excess = self.below - x
            if excess <= 0:
                return 0.0
            return self.fixed + (self.rate * excess if self.per_unit else 0.0)
        return self.fixed + self.rate * x


@dataclass(frozen=True)
class CostBreakdown:
    items: dict[str, float]
    total_cost: float
    total_revenue: float

    @property
    def profit(self) -> float:
        return self.total_revenue - self.total_cost

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["profit"] = self.profit
        return d


@dataclass
class CostModel:
    terms: list[CostTerm] = field(default_factory=list)
    prefix: str = "cost"

    def _add(self, term: CostTerm) -> CostModel:
        if any(t.name == term.name for t in self.terms):
            raise ConfigError(f"duplicate cost term {term.name!r}")
        self.terms.append(term)
        return self

    def fixed(self, name: str, amount: float) -> CostModel:
        return self._add(CostTerm(name, "cost", fixed=amount))

    def variable(self, name: str, metric: str, rate: float) -> CostModel:
        return self._add(CostTerm(name, "cost", metric, rate))

    def revenue(self, name: str, metric: str, rate: float) -> CostModel:
        return self._add(CostTerm(name, "revenue", metric, rate))

    def resource(
        self, resource: str, *, per_capacity_time: float = 0.0, per_busy_time: float = 0.0
    ) -> CostModel:
        """Staffing/ownership cost (per unit of capacity per time) and usage cost (per busy unit-time)."""
        if per_capacity_time:
            self._add(
                CostTerm(
                    f"{resource}.capacity",
                    "cost",
                    f"resource.{resource}.capacity_time",
                    per_capacity_time,
                )
            )
        if per_busy_time:
            self._add(
                CostTerm(
                    f"{resource}.usage", "cost", f"resource.{resource}.busy_time", per_busy_time
                )
            )
        return self

    def waiting(self, resource: str, *, per_time: float) -> CostModel:
        """Cost of customer waiting: total wait time at ``resource`` times ``per_time``."""
        return self._add(
            CostTerm(f"{resource}.waiting", "cost", f"resource.{resource}.wait.total", per_time)
        )

    def penalty(
        self,
        name: str,
        metric: str,
        *,
        amount: float = 0.0,
        above: float | None = None,
        below: float | None = None,
        per_unit: float | None = None,
    ) -> CostModel:
        """A charge when ``metric`` breaches a threshold: flat ``amount`` plus ``per_unit`` x excess."""
        if (above is None) == (below is None):
            raise ConfigError("penalty needs exactly one of above= or below=")
        return self._add(
            CostTerm(
                name,
                "cost",
                metric,
                rate=per_unit or 0.0,
                fixed=amount,
                above=above,
                below=below,
                per_unit=per_unit is not None,
            )
        )

    def calculate(self, result: Any) -> CostBreakdown:
        """Cost breakdown for a :class:`SimulationResult` or a metrics mapping."""
        metrics: Mapping[str, float] = result.metrics if hasattr(result, "metrics") else result
        items: dict[str, float] = {}
        cost = revenue = 0.0
        for t in self.terms:
            v = t.amount(metrics)
            items[t.name] = v
            if t.kind == "cost":
                cost += v
            else:
                revenue += v
        return CostBreakdown(items, cost, revenue)

    def metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Derived metrics (``cost.total``, ``cost.revenue``, ``cost.profit``, ``cost.<term>``)."""
        b = self.calculate(metrics)
        p = self.prefix
        out = {f"{p}.{k}": v for k, v in b.items.items()}
        out[f"{p}.total"] = b.total_cost
        out[f"{p}.revenue"] = b.total_revenue
        out[f"{p}.profit"] = b.profit
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> CostModel:
        """Build from configuration: ``{"terms": [{"name":..., "kind":..., "metric":..., ...}]}``."""
        raw = data.get("terms", []) if isinstance(data, Mapping) else data
        allowed = set(CostTerm.__dataclass_fields__)
        cm = cls()
        for t in raw:
            unknown = set(t) - allowed
            if unknown:
                raise ConfigError(f"unknown cost term fields: {sorted(unknown)}")
            if t.get("kind", "cost") not in ("cost", "revenue"):
                raise ConfigError(
                    f"cost term kind must be 'cost' or 'revenue', got {t.get('kind')!r}"
                )
            try:
                cm._add(CostTerm(**{"kind": "cost", **t}))
            except TypeError as exc:
                raise ConfigError(f"invalid cost term {t!r}: {exc}") from exc
        return cm

    def to_dict(self) -> dict[str, Any]:
        return {"terms": [asdict(t) for t in self.terms]}
