"""Simulation optimisation: search a parameter box, then confirm the winner.

:func:`optimize` searches by grid, Latin hypercube random search or Bayesian
optimisation (a Gaussian-process surrogate with expected improvement), all
evaluated with common random numbers through :class:`Objective`. Because
every estimate is noisy, the best few candidates are then re-run with fresh
seeds and compared with ranking and selection
(:func:`~simulsi.analysis.selection.select_best`), so the reported winner is
not just the luckiest draw.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats

from simulsi.analysis.report import format_table
from simulsi.analysis.selection import SelectionResult, select_best
from simulsi.core.model import Model
from simulsi.optimization.objective import Evaluation, Objective
from simulsi.optimization.surrogate import (
    GaussianProcess,
    _int_params,
    _scale,
    latin_hypercube,
)
from simulsi.randomness.stream import derive_seed

Method = Literal["grid", "random", "bayes", "cem"]


@dataclass
class OptimizationResult:
    method: str
    metric: str | None
    minimize: bool
    best_parameters: dict[str, Any]
    best_value: float
    history: list[Evaluation]
    selection: SelectionResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def evaluations(self) -> int:
        return len(self.history)

    def top(self, n: int = 5) -> list[Evaluation]:
        valid = [e for e in self.history if not math.isnan(e.value)]
        return sorted(valid, key=lambda e: e.value if self.minimize else -e.value)[:n]

    def format(self) -> str:
        lines = [
            f"{self.method} search, {self.evaluations} evaluations; best "
            f"{self.metric or 'objective'} = {self.best_value:.6g} at {self.best_parameters}"
        ]
        lines.append(format_table([{**e.parameters, "value": e.value} for e in self.top()]))
        if self.selection is not None:
            lines.append(self.selection.format())
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        from simulsi.analysis.report import html_table

        caption = (
            f"{self.method} search, {self.evaluations} evaluations; best "
            f"{self.metric or 'objective'} = {self.best_value:.6g} at {self.best_parameters}"
        )
        return html_table([{**e.parameters, "value": e.value} for e in self.top()], caption=caption)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "metric": self.metric,
            "minimize": self.minimize,
            "best_parameters": dict(self.best_parameters),
            "best_value": self.best_value,
            "evaluations": self.evaluations,
            "history": [{"parameters": e.parameters, "value": e.value} for e in self.history],
            "selection": None if self.selection is None else self.selection.to_dict(),
            "notes": list(self.notes),
        }


def _grid(
    ranges: Mapping[str, tuple[float, float]], ints: list[bool], levels: int
) -> list[dict[str, Any]]:
    axes: list[list[Any]] = []
    for (lo, hi), is_int in zip(ranges.values(), ints, strict=True):
        if is_int:
            axes.append(list(range(math.ceil(lo), math.floor(hi) + 1)))
        else:
            axes.append([float(v) for v in np.linspace(lo, hi, levels)])
    return [dict(zip(ranges, combo, strict=True)) for combo in itertools.product(*axes)]


def optimize(
    model: Model,
    metric: str | None,
    ranges: Mapping[str, tuple[float, float]],
    *,
    method: Method = "bayes",
    budget: int = 30,
    replications: int = 5,
    minimize: bool = True,
    transform: Callable[[Mapping[str, float]], float] | None = None,
    fixed: Mapping[str, Any] | None = None,
    seed: int = 0,
    grid_levels: int = 5,
    n_initial: int | None = None,
    confirm_top: int = 3,
    indifference: float | None = None,
) -> OptimizationResult:
    """Search ``ranges`` (``{parameter: (low, high)}``) for the best ``metric``.

    * ``method="grid"`` evaluates every combination (integers: every value;
      floats: ``grid_levels`` points); it fails if that exceeds ``budget``.
    * ``"random"`` evaluates a Latin hypercube of ``budget`` points.
    * ``"bayes"`` starts from a Latin hypercube of ``n_initial`` points
      (default ``max(5, 2d + 1)``) and then picks each next point by
      expected improvement on a Gaussian-process surrogate.
    * ``"cem"`` is the cross-entropy method: sample a population from a
      Gaussian over the box, keep the best fifth, refit the Gaussian to them
      and repeat. It suits policy parameters (thresholds, trigger levels)
      and copes with noisy, non-smooth responses.

    Each candidate is the mean of ``replications`` runs with shared seeds.
    With ``indifference`` set (and a ``metric``), the best ``confirm_top``
    distinct candidates are compared afresh by :func:`select_best` (new
    seeds), and its choice becomes ``best_parameters``.
    """
    names = list(ranges)
    if not names:
        raise ValueError("need at least one parameter range")
    for k, (lo, hi) in ranges.items():
        if not hi > lo:
            raise ValueError(f"range for {k!r} needs low < high")
    if budget < 1:
        raise ValueError("budget must be >= 1")
    obj = Objective(
        model,
        metric,
        names,
        fixed=fixed,
        replications=replications,
        seed=seed,
        minimize=minimize,
        transform=transform,
    )
    ints = _int_params(model, names)
    notes: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    def run(p: Mapping[str, Any]) -> None:
        key = tuple(p[k] for k in names)
        if key not in seen:
            seen.add(key)
            obj(p)

    if method == "grid":
        points = _grid(ranges, ints, grid_levels)
        if len(points) > budget:
            raise ValueError(
                f"the grid has {len(points)} points but budget={budget}; "
                "raise the budget or use method='random' or 'bayes'"
            )
        for p in points:
            run(p)
    elif method == "random":
        for p in _scale(
            latin_hypercube(budget, len(names), derive_seed(seed, "opt")), ranges, ints
        ):
            run(p)
    elif method == "bayes":
        _bayes(obj, run, seen, names, ranges, ints, budget, n_initial, seed)
    elif method == "cem":
        _cross_entropy(obj, run, seen, names, ranges, ints, budget, seed)
    else:
        raise ValueError(f"unknown method {method!r}; use 'grid', 'random', 'bayes' or 'cem'")

    best = obj.best()
    if best is None:
        raise RuntimeError("no valid evaluation; check parameter ranges against the model")
    result = OptimizationResult(
        method=method,
        metric=metric,
        minimize=minimize,
        best_parameters={k: best.parameters[k] for k in names},
        best_value=best.value,
        history=list(obj.history),
        notes=notes,
    )
    if indifference is not None:
        if metric is None:
            notes.append("confirmation skipped: select_best needs a metric, not a transform")
        else:
            top = result.top(confirm_top)
            if len(top) >= 2:
                cands = {f"candidate-{i + 1}": dict(e.parameters) for i, e in enumerate(top)}
                sel = select_best(
                    model,
                    cands,
                    metric,
                    indifference=indifference,
                    minimize=minimize,
                    seed=derive_seed(seed, "confirm"),
                )
                result.selection = sel
                chosen = cands[sel.best]
                result.best_parameters = {k: chosen[k] for k in names}
                result.best_value = sel.means[sel.best]
                if sel.best != "candidate-1":
                    notes.append(
                        f"confirmation runs preferred {sel.best} over the search's best "
                        "(the search estimate was optimistic)"
                    )
    return result


def _bayes(
    obj: Objective,
    run: Callable[[Mapping[str, Any]], None],
    seen: set[tuple[Any, ...]],
    names: list[str],
    ranges: Mapping[str, tuple[float, float]],
    ints: list[bool],
    budget: int,
    n_initial: int | None,
    seed: int,
) -> None:
    d = len(names)
    n0 = min(budget, n_initial if n_initial is not None else max(5, 2 * d + 1))
    for p in _scale(latin_hypercube(n0, d, derive_seed(seed, "opt")), ranges, ints):
        run(p)
    rng = np.random.default_rng(derive_seed(seed, "opt-ei"))
    lo = np.array([ranges[k][0] for k in names], dtype=float)
    hi = np.array([ranges[k][1] for k in names], dtype=float)
    while len(seen) < budget:
        valid = [e for e in obj.history if not math.isnan(e.value)]
        X = np.array([[float(e.parameters[k]) for k in names] for e in valid])
        y = np.array([obj.sign * e.value for e in valid])  # minimise internally
        gp = GaussianProcess(seed=seed + len(seen)).fit(X, y)
        # candidates: uniform over the box plus perturbations of the incumbent
        best_x = X[int(np.argmin(y))]
        cand = lo + rng.random((2000, d)) * (hi - lo)
        local = best_x + rng.normal(0, 0.05, (500, d)) * (hi - lo)
        cand = np.clip(np.vstack([cand, local]), lo, hi)
        for j, is_int in enumerate(ints):
            if is_int:
                cand[:, j] = np.round(cand[:, j])
        mu, sd = gp.predict(cand, return_std=True)
        f_best = float(np.min(gp.predict(X)))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (f_best - mu) / sd
            ei = np.where(sd > 0, (f_best - mu) * stats.norm.cdf(z) + sd * stats.norm.pdf(z), 0.0)
        picked = False
        for i in np.argsort(-ei):
            p = {
                k: (int(cand[i, j]) if ints[j] else float(cand[i, j])) for j, k in enumerate(names)
            }
            if tuple(p[k] for k in names) not in seen:
                run(p)
                picked = True
                break
        if not picked:
            break  # every candidate was already evaluated (small integer space)


def _cross_entropy(
    obj: Objective,
    run: Callable[[Mapping[str, Any]], None],
    seen: set[tuple[Any, ...]],
    names: list[str],
    ranges: Mapping[str, tuple[float, float]],
    ints: list[bool],
    budget: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(derive_seed(seed, "opt-cem"))
    lo = np.array([ranges[k][0] for k in names], dtype=float)
    hi = np.array([ranges[k][1] for k in names], dtype=float)
    mean, std = (lo + hi) / 2, (hi - lo) / 3
    population = max(6, min(20, budget // 4))
    stalls = 0
    while len(seen) < budget and stalls < 5:
        before = len(seen)
        batch: list[dict[str, Any]] = []
        for _ in range(min(population, budget - len(seen))):
            x = np.clip(rng.normal(mean, std), lo, hi)
            p = {k: (round(float(x[j])) if ints[j] else float(x[j])) for j, k in enumerate(names)}
            run(p)
            batch.append(p)
        stalls = stalls + 1 if len(seen) == before else 0
        scored = [(obj(p), p) for p in batch]  # cached: no extra simulation
        scored.sort(key=lambda t: t[0])
        elite = np.array(
            [[float(p[k]) for k in names] for _, p in scored[: max(2, len(scored) // 5)]]
        )
        mean = 0.7 * elite.mean(axis=0) + 0.3 * mean
        std = np.maximum(0.7 * elite.std(axis=0) + 0.3 * std, (hi - lo) * 0.01)
