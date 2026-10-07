"""Ranking and selection: pick the best scenario with a statistical guarantee.

:func:`select_best` implements the fully sequential procedure KN of Kim and
Nelson (2001). It runs every scenario ``n0`` times, then adds one
replication at a time to the scenarios still in contention and eliminates
any that is clearly worse than another. With probability at least
``confidence`` the scenario selected is the true best, or within
``indifference`` of it (the smallest difference you care about).

Replications use common random numbers, which KN exploits: it works with the
variance of *paired* differences, which CRN makes small. Compared with a
fixed number of replications per scenario, clearly inferior scenarios stop
early and the budget goes to the close contenders.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulsi.analysis.report import format_table
from simulsi.core.model import Model
from simulsi.randomness.stream import derive_seed
from simulsi.scenarios.scenario import Scenario, check_unique_names


@dataclass
class SelectionResult:
    best: str
    metric: str
    minimize: bool
    confidence: float
    indifference: float
    converged: bool
    means: dict[str, float]
    replications: dict[str, int]
    eliminated: dict[str, tuple[int, str]] = field(default_factory=dict)
    note: str = ""

    @property
    def total_replications(self) -> int:
        return sum(self.replications.values())

    def format(self) -> str:
        rows = []
        for name in sorted(self.means, key=lambda n: (n != self.best, n in self.eliminated)):
            out = self.eliminated.get(name)
            rows.append(
                {
                    "scenario": name,
                    "mean": self.means[name],
                    "replications": self.replications[name],
                    "status": "selected"
                    if name == self.best
                    else (f"eliminated at r={out[0]} by {out[1]}" if out else "in contention"),
                }
            )
        head = (
            f"best {self.metric} ({'min' if self.minimize else 'max'}imised): {self.best}  "
            f"[KN, confidence {self.confidence:g}, indifference {self.indifference:g}, "
            f"{self.total_replications} replications]"
        )
        text = head + "\n" + format_table(rows)
        return text + (f"\nnote: {self.note}" if self.note else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "metric": self.metric,
            "minimize": self.minimize,
            "confidence": self.confidence,
            "indifference": self.indifference,
            "converged": self.converged,
            "means": dict(self.means),
            "replications": dict(self.replications),
            "eliminated": {k: {"at": v[0], "by": v[1]} for k, v in self.eliminated.items()},
            "total_replications": self.total_replications,
            "note": self.note,
        }


def select_best(
    model: Model,
    scenarios: Sequence[Scenario] | Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    indifference: float,
    minimize: bool = True,
    confidence: float = 0.95,
    n0: int = 10,
    max_replications: int = 500,
    seed: int = 0,
) -> SelectionResult:
    """Select the best of ``scenarios`` on ``metric`` with procedure KN.

    ``indifference`` (delta) is in the metric's units: differences smaller
    than this are treated as ties, and any scenario within delta of the best
    counts as a correct selection. Smaller delta means more replications.
    If ``max_replications`` per scenario is reached first, the scenario with
    the best mean among those remaining is returned with ``converged=False``.
    """
    if isinstance(scenarios, Mapping):
        scenarios = [Scenario(k, dict(v)) for k, v in scenarios.items()]
    scenarios = list(scenarios)
    check_unique_names(scenarios)
    k = len(scenarios)
    if k < 2:
        raise ValueError("need at least two scenarios to select from")
    if not (indifference > 0):
        raise ValueError("indifference must be > 0")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if n0 < 2 or max_replications < n0:
        raise ValueError("need 2 <= n0 <= max_replications")
    for s in scenarios:
        model.resolve(s.parameters)  # fail fast on bad parameters

    names = [s.name for s in scenarios]
    params = {s.name: s.parameters for s in scenarios}
    sign = -1.0 if minimize else 1.0  # internally maximise
    data: dict[str, list[float]] = {n: [] for n in names}

    def run(name: str, r: int) -> None:
        m = model.simulate(params[name], seed=derive_seed(seed, "replication", r)).metrics
        if metric not in m:
            raise KeyError(f"model produced no metric {metric!r}")
        value = float(m[metric])
        if math.isnan(value):
            raise ValueError(f"metric {metric!r} is NaN in {name!r} replication {r}")
        data[name].append(sign * value)

    for r in range(n0):
        for n in names:
            run(n, r)

    alpha = 1 - confidence
    eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    h2 = 2 * eta * (n0 - 1)
    first = {n: np.asarray(data[n]) for n in names}
    s2 = {
        (i, j): float(np.var(first[i] - first[j], ddof=1)) for i in names for j in names if i != j
    }
    alive = list(names)
    eliminated: dict[str, tuple[int, str]] = {}
    r = n0
    delta = indifference
    while True:
        means = {n: float(np.mean(data[n])) for n in alive}
        out = []
        for i in alive:
            for j in alive:
                if i == j:
                    continue
                w = max(0.0, delta / (2 * r) * (h2 * s2[(i, j)] / delta**2 - r))
                if means[i] < means[j] - w:
                    out.append((i, j))
                    break
        for i, j in out:
            eliminated[i] = (r, j)
        alive = [n for n in alive if n not in eliminated]
        if len(alive) == 1 or r >= max_replications:
            break
        for n in alive:
            run(n, r)
        r += 1

    means_all = {n: sign * float(np.mean(data[n])) for n in names}
    converged = len(alive) == 1
    best = max(alive, key=lambda n: sign * means_all[n])
    note = (
        ""
        if converged
        else f"max_replications={max_replications} reached with {len(alive)} scenarios "
        "still in contention; the best mean among them was returned"
    )
    return SelectionResult(
        best=best,
        metric=metric,
        minimize=minimize,
        confidence=confidence,
        indifference=indifference,
        converged=converged,
        means=means_all,
        replications={n: len(data[n]) for n in names},
        eliminated=eliminated,
        note=note,
    )
