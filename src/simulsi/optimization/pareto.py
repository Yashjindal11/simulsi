"""Multi-objective optimisation: the Pareto front of trade-offs.

Real decisions trade objectives against each other - cost against service
level, on-time performance against passenger delay. :func:`pareto_search`
evaluates candidate designs (a grid, a Latin hypercube over ranges, or named
scenarios) with common random numbers and returns the *non-dominated* ones:
designs for which no other design is at least as good on every objective
and strictly better on one.

>>> from simulsi.models import disruption_recovery
>>> res = pareto_search(
...     disruption_recovery,
...     {"cost": "min", "otp": "max"},
...     grid={"policy": ["delay", "cancel", "spares"], "spares": [1, 3]},
...     replications=2,
... )
>>> len(res.front) >= 1
True
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from simulsi.analysis.report import format_table, html_table
from simulsi.core.model import Model
from simulsi.optimization.surrogate import _int_params, _scale, latin_hypercube
from simulsi.randomness.stream import derive_seed
from simulsi.scenarios.scenario import Scenario, grid

Sense = Literal["min", "max"]


def dominates(a: Sequence[float], b: Sequence[float], senses: Sequence[Sense]) -> bool:
    """True if ``a`` is at least as good as ``b`` everywhere and strictly better somewhere."""
    better = False
    for x, y, s in zip(a, b, senses, strict=True):
        if s == "max":
            x, y = -x, -y
        if x > y:
            return False
        if x < y:
            better = True
    return better


def pareto_front(points: Sequence[Sequence[float]], senses: Sequence[Sense]) -> list[int]:
    """Indices of the non-dominated points (NaN points are never on the front)."""
    valid = [i for i, p in enumerate(points) if all(math.isfinite(v) for v in p)]
    return [
        i
        for i in valid
        if not any(dominates(points[j], points[i], senses) for j in valid if j != i)
    ]


@dataclass
class Design:
    name: str
    parameters: dict[str, Any]
    objectives: dict[str, float]
    half_widths: dict[str, float]
    on_front: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "objectives": dict(self.objectives),
            "half_widths": dict(self.half_widths),
            "on_front": self.on_front,
        }


@dataclass
class ParetoResult:
    objectives: dict[str, Sense]
    designs: list[Design]
    replications: int
    notes: list[str] = field(default_factory=list)

    @property
    def front(self) -> list[Design]:
        """Non-dominated designs, sorted by the first objective."""
        first = next(iter(self.objectives))
        return sorted((d for d in self.designs if d.on_front), key=lambda d: d.objectives[first])

    def _rows(self, designs: Sequence[Design]) -> list[dict[str, Any]]:
        rows = []
        for d in designs:
            row: dict[str, Any] = {"design": d.name}
            for k in self.objectives:
                v, hw = d.objectives[k], d.half_widths.get(k, math.nan)
                row[k] = v if math.isnan(hw) else f"{v:.4g} ±{hw:.2g}"
            row["pareto"] = "*" if d.on_front else ""
            rows.append(row)
        return rows

    def format(self, all_designs: bool = False) -> str:
        senses = ", ".join(f"{k} ({s})" for k, s in self.objectives.items())
        shown = self.designs if all_designs else self.front
        head = (
            f"Pareto front: {len(self.front)} of {len(self.designs)} designs "
            f"non-dominated on {senses}; {self.replications} replications each"
        )
        return "\n".join(
            [head, format_table(self._rows(shown)), *(f"note: {n}" for n in self.notes)]
        )

    def _repr_html_(self) -> str:
        return html_table(
            self._rows(self.designs), caption=f"Pareto search ({len(self.front)} on front)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": dict(self.objectives),
            "replications": self.replications,
            "designs": [d.to_dict() for d in self.designs],
            "front": [d.name for d in self.front],
            "notes": list(self.notes),
        }

    def plot(self, x: str | None = None, y: str | None = None, ax: Any = None) -> Any:
        """Scatter of two objectives with the front highlighted (needs matplotlib)."""
        import matplotlib.pyplot as plt

        keys = list(self.objectives)
        if len(keys) < 2 and (x is None or y is None):
            raise ValueError("plotting needs two objectives")
        x = x or keys[0]
        y = y or keys[1]
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))
        for d in self.designs:
            ax.errorbar(
                d.objectives[x],
                d.objectives[y],
                xerr=d.half_widths.get(x),
                yerr=d.half_widths.get(y),
                fmt="o",
                color="#4f46e5" if d.on_front else "#cbd5e1",
                ecolor="#94a3b8",
                capsize=2,
            )
        front = sorted(self.front, key=lambda d: d.objectives[x])
        ax.plot(
            [d.objectives[x] for d in front], [d.objectives[y] for d in front], "-", color="#4f46e5"
        )
        for d in front:
            ax.annotate(
                d.name,
                (d.objectives[x], d.objectives[y]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set_xlabel(f"{x} ({self.objectives[x]})")
        ax.set_ylabel(f"{y} ({self.objectives[y]})")
        ax.set_title("Pareto front")
        ax.margins(x=0.2, y=0.1)
        return ax


def pareto_search(
    model: Model,
    objectives: Mapping[str, Sense],
    *,
    grid: Mapping[str, Sequence[Any]] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    scenarios: Mapping[str, Mapping[str, Any]] | None = None,
    budget: int = 30,
    replications: int = 5,
    seed: int = 0,
    fixed: Mapping[str, Any] | None = None,
    workers: int = 1,
) -> ParetoResult:
    """Evaluate candidate designs and mark the Pareto-optimal ones.

    Candidates come from exactly one of ``grid`` (full factorial), ``ranges``
    (a Latin hypercube of ``budget`` points) or ``scenarios`` (named
    parameter sets). ``objectives`` maps metric names to ``"min"`` or
    ``"max"``. Means over ``replications`` (common random numbers) decide
    dominance; designs whose intervals overlap may swap places with more
    replications, which the notes point out.
    """
    from simulsi.experiments.experiment import Experiment

    if not objectives:
        raise ValueError("give at least one objective")
    for k, s in objectives.items():
        if s not in ("min", "max"):
            raise ValueError(f"objective {k!r}: sense must be 'min' or 'max', got {s!r}")
    if sum(x is not None for x in (grid, ranges, scenarios)) != 1:
        raise ValueError("give exactly one of grid=, ranges= or scenarios=")
    base = dict(fixed or {})
    cands: list[Scenario]
    if grid is not None:
        cands = _grid_scenarios(grid, base)
    elif ranges is not None:
        names = list(ranges)
        unit = latin_hypercube(budget, len(names), derive_seed(seed, "pareto"))
        points = _scale(unit, ranges, _int_params(model, names))
        cands = [Scenario(f"design-{i + 1}", {**base, **pt}) for i, pt in enumerate(points)]
    else:
        assert scenarios is not None
        cands = [Scenario(k, {**base, **v}) for k, v in scenarios.items()]
    if not cands:
        raise ValueError("no candidate designs")
    exp = Experiment(model, cands, replications=replications, seed=seed, workers=workers)
    issues = exp.validate()
    if issues:
        raise ValueError("; ".join(issues))
    res = exp.run()
    names = list(objectives)
    summary = {(r["scenario"], r["metric"]): r for r in res.summary(names)}
    designs = []
    for sc in cands:
        obj = {k: float(summary[(sc.name, k)]["mean"]) for k in names}
        hw = {k: float(summary[(sc.name, k)]["half_width"]) for k in names}
        designs.append(
            Design(sc.name, {k: v for k, v in sc.parameters.items() if k not in base}, obj, hw)
        )
    senses = [objectives[k] for k in names]
    front = set(pareto_front([[d.objectives[k] for k in names] for d in designs], senses))
    for i in front:
        designs[i].on_front = True
    notes = []
    missing = [d.name for d in designs if any(math.isnan(v) for v in d.objectives.values())]
    if missing:
        notes.append(
            f"{len(missing)} design(s) produced no value for an objective and were excluded"
        )
    close = _overlaps(designs, names)
    if close:
        notes.append(
            f"{close} dominated design(s) overlap a front design within the CIs; "
            "more replications may change the front"
        )
    return ParetoResult(dict(objectives), designs, replications, notes)


def _grid_scenarios(axes: Mapping[str, Sequence[Any]], base: Mapping[str, Any]) -> list[Scenario]:
    return [Scenario(s.name, dict(s.parameters)) for s in grid(axes, dict(base))]


def _overlaps(designs: Sequence[Design], names: Sequence[str]) -> int:
    front = [d for d in designs if d.on_front]
    count = 0
    for d in designs:
        if d.on_front or any(math.isnan(v) for v in d.objectives.values()):
            continue
        for f in front:
            if all(
                abs(d.objectives[k] - f.objectives[k])
                <= np.nan_to_num(d.half_widths[k]) + np.nan_to_num(f.half_widths[k])
                for k in names
            ):
                count += 1
                break
    return count
