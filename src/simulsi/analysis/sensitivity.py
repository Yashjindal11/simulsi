"""Sensitivity analysis: which inputs move which outputs, and by how much.

Methods implemented (all simple and well understood):

* :func:`one_at_a_time` - change one parameter at a time around a base
  point, with common random numbers, and report the change in each output
  plus a paired confidence interval and an elasticity.
* :func:`finite_difference` - local derivative ``d output / d parameter``
  by central (or forward) differences with common random numbers.
* :func:`correlation_sensitivity` - global screening from Monte Carlo
  samples: Pearson, Spearman rank correlation and standardised regression
  coefficients (SRC, with the regression R^2 so you can tell whether a
  linear summary is adequate).

These are screening tools. Correlation-based measures miss interactions and
strongly non-monotonic effects; variance-based methods (Sobol) would be a
natural extension behind the same result types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as _st

from simulsi.analysis.report import format_table
from simulsi.core.model import Model
from simulsi.experiments.montecarlo import MonteCarloResult
from simulsi.randomness.stream import derive_seed
from simulsi.statistics.core import paired_difference, t_half_width


@dataclass(frozen=True)
class SensitivityRow:
    parameter: str
    output: str
    method: str
    value: float
    ci_low: float = math.nan
    ci_high: float = math.nan
    setting: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SensitivityResult:
    def __init__(
        self, rows: Sequence[SensitivityRow], info: Mapping[str, Any] | None = None
    ) -> None:
        self.rows = list(rows)
        self.info = dict(info or {})

    def ranking(self, output: str | None = None, method: str | None = None) -> list[SensitivityRow]:
        """Rows sorted by absolute effect size, largest first."""
        rows = [
            r
            for r in self.rows
            if (output is None or r.output == output) and (method is None or r.method == method)
        ]
        return sorted(rows, key=lambda r: -abs(r.value) if not math.isnan(r.value) else 0.0)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rows]

    def format(self) -> str:
        return format_table(
            [
                {
                    "parameter": r.parameter,
                    "setting": r.setting,
                    "output": r.output,
                    "method": r.method,
                    "value": r.value,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                }
                for r in self.ranking()
            ]
        )

    def __repr__(self) -> str:
        return f"SensitivityResult({len(self.rows)} rows)"


def _run_metrics(
    model: Model, params: Mapping[str, Any], outputs: Sequence[str], reps: int, seed: int
) -> dict[str, np.ndarray[Any, Any]]:
    vals: dict[str, list[float]] = {o: [] for o in outputs}
    for r in range(reps):
        m = model.simulate(params, seed=derive_seed(seed, "replication", r)).metrics
        for o in outputs:
            if o not in m:
                raise KeyError(f"model produced no metric {o!r}")
            vals[o].append(m[o])
    return {o: np.asarray(v, dtype=float) for o, v in vals.items()}


def one_at_a_time(
    model: Model,
    changes: Mapping[str, Sequence[Any]],
    outputs: Sequence[str],
    *,
    base: Mapping[str, Any] | None = None,
    replications: int = 10,
    seed: int = 0,
    confidence: float = 0.95,
) -> SensitivityResult:
    """Vary each parameter in ``changes`` alone; report output change vs the base point.

    ``value`` is the mean change in the output; ``detail`` includes the
    elasticity ``(dy/y) / (dx/x)`` when both base values are non-zero numbers.
    """
    base_params = dict(model.resolve(base))
    base_out = _run_metrics(model, base_params, outputs, replications, seed)
    rows = []
    for param, values in changes.items():
        if param not in base_params:
            raise KeyError(f"unknown parameter {param!r}")
        x0 = base_params[param]
        for v in values:
            out = _run_metrics(model, {**(base or {}), param: v}, outputs, replications, seed)
            for o in outputs:
                d = paired_difference(base_out[o], out[o], confidence)
                elasticity = math.nan
                y0 = float(np.nanmean(base_out[o]))
                if (
                    isinstance(x0, int | float)
                    and isinstance(v, int | float)
                    and x0
                    and y0
                    and v != x0
                ):
                    elasticity = (d.difference / y0) / ((v - x0) / x0)
                rows.append(
                    SensitivityRow(
                        param,
                        o,
                        "oat",
                        d.difference,
                        d.ci_low,
                        d.ci_high,
                        f"{param}={v}",
                        f"base {param}={x0}; base output={y0:.4g}; elasticity={elasticity:.3g}",
                    )
                )
    return SensitivityResult(
        rows, {"base": base_params, "replications": replications, "seed": seed}
    )


def finite_difference(
    model: Model,
    parameters: Sequence[str],
    outputs: Sequence[str],
    *,
    base: Mapping[str, Any] | None = None,
    relative_step: float = 0.05,
    replications: int = 10,
    seed: int = 0,
    scheme: Literal["central", "forward"] = "central",
    confidence: float = 0.95,
) -> SensitivityResult:
    """Local derivatives by finite differences with common random numbers.

    Integer parameters use a step of at least 1. The CI comes from the
    per-replication paired differences, so it reflects simulation noise only;
    a step that is too small amplifies noise, too large adds bias.
    """
    base_params = dict(model.resolve(base))
    rows = []
    base_out = (
        _run_metrics(model, base_params, outputs, replications, seed)
        if scheme == "forward"
        else None
    )
    for p in parameters:
        x0 = base_params[p]
        if isinstance(x0, bool) or not isinstance(x0, int | float):
            raise TypeError(f"parameter {p!r} is not numeric")
        h: float = abs(x0) * relative_step or relative_step
        if isinstance(x0, int):
            h = max(1, round(h))
        hi_params = {**(base or {}), p: x0 + h}
        hi = _run_metrics(model, hi_params, outputs, replications, seed)
        if scheme == "central":
            lo = _run_metrics(model, {**(base or {}), p: x0 - h}, outputs, replications, seed)
            span = 2 * h
        else:
            assert base_out is not None
            lo, span = base_out, h
        for o in outputs:
            per_rep = (hi[o] - lo[o]) / span
            mean = float(np.nanmean(per_rep))
            hw = t_half_width(per_rep, confidence)
            rows.append(
                SensitivityRow(p, o, f"fd-{scheme}", mean, mean - hw, mean + hw, f"step={h:g}")
            )
    return SensitivityResult(rows, {"base": base_params, "relative_step": relative_step})


def correlation_sensitivity(
    mc: MonteCarloResult,
    outputs: Sequence[str] | None = None,
    *,
    methods: Sequence[Literal["pearson", "spearman", "src"]] = ("pearson", "spearman", "src"),
) -> SensitivityResult:
    """Global screening from Monte Carlo samples of uncertain inputs."""
    names = [k for k, v in mc.inputs.items() if np.issubdtype(np.asarray(v).dtype, np.number)]
    if not names:
        raise ValueError("no numeric sampled inputs to analyse")
    X = np.column_stack([np.asarray(mc.inputs[k], dtype=float) for k in names])
    rows = []
    info: dict[str, Any] = {"r2": {}}
    for o in outputs if outputs is not None else mc.output_names:
        y = mc.outputs[o]
        ok = ~np.isnan(y)
        Xo, yo = X[ok], y[ok]
        if len(yo) < 3 or np.std(yo) == 0:
            continue
        for j, name in enumerate(names):
            x = Xo[:, j]
            if np.std(x) == 0:
                continue
            if "pearson" in methods:
                r, pv = _st.pearsonr(x, yo)
                rows.append(
                    SensitivityRow(name, o, "pearson", float(r), detail=f"p={float(pv):.3g}")
                )
            if "spearman" in methods:
                rho, pv = _st.spearmanr(x, yo)
                rows.append(
                    SensitivityRow(name, o, "spearman", float(rho), detail=f"p={float(pv):.3g}")
                )
        if "src" in methods:
            sd = Xo.std(axis=0)
            keep = sd > 0
            Z = (Xo[:, keep] - Xo[:, keep].mean(axis=0)) / sd[keep]
            zy = (yo - yo.mean()) / yo.std()
            A = np.column_stack([np.ones(len(zy)), Z])
            coef, *_ = np.linalg.lstsq(A, zy, rcond=None)
            pred = A @ coef
            r2 = 1 - float(((zy - pred) ** 2).sum() / (zy**2).sum())
            info["r2"][o] = r2
            for name, c in zip(
                [n for n, k in zip(names, keep, strict=True) if k], coef[1:], strict=True
            ):
                rows.append(SensitivityRow(name, o, "src", float(c), detail=f"R2={r2:.3f}"))
    return SensitivityResult(rows, info)
