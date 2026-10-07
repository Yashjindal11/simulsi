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

* :func:`sobol_indices` - variance-based global sensitivity: first-order
  and total-effect Sobol indices (Saltelli 2010 / Jansen estimators) with
  bootstrap confidence intervals. Captures interactions and non-linear
  effects at a cost of ``N * (d + 2)`` model evaluations.
* :func:`morris_screening` - Morris elementary effects: a cheap global
  screen (``r * (d + 1)`` evaluations) that separates negligible inputs from
  important ones (``mu_star``) and flags non-linearity or interactions
  (``sigma``). Use it to prune many inputs before a Sobol analysis.

Correlation measures are cheap screening tools that miss interactions and
non-monotonic effects; use Sobol indices when that matters.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as _st

from simulsi.analysis.report import format_table
from simulsi.core.model import Model
from simulsi.experiments.montecarlo import MonteCarloResult, sample_inputs
from simulsi.randomness.distributions import DistributionLike
from simulsi.randomness.stream import RandomStream, derive_seed
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


def _evaluate_rows(
    model: Callable[..., Any] | Model,
    inputs: Mapping[str, np.ndarray[Any, Any]],
    rows: int,
    *,
    fixed: Mapping[str, Any],
    vectorized: bool,
    seed: int,
    crn_period: int,
    seed_index: Sequence[int] | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Evaluate ``model`` on every row of ``inputs``.

    Row ``j`` uses replication seed ``seed_index[j]`` if given, else ``j % crn_period``.
    """
    if vectorized:
        if isinstance(model, Model):
            raise ValueError("vectorized=True is not available for simulation models")
        res = model(**fixed, **inputs)
        arrays = res if isinstance(res, Mapping) else {"value": res}
        return {
            k: np.broadcast_to(np.asarray(v, dtype=float), (rows,)).copy()
            for k, v in arrays.items()
        }
    wants_rng = not isinstance(model, Model) and "rng" in inspect.signature(model).parameters
    out: dict[str, list[float]] = {}
    for j in range(rows):
        params = {
            **fixed,
            **{k: v[j].item() if hasattr(v[j], "item") else v[j] for k, v in inputs.items()},
        }
        rep_seed = derive_seed(
            seed, "replication", seed_index[j] if seed_index is not None else j % crn_period
        )
        if isinstance(model, Model):
            values: Any = model.simulate(params, seed=rep_seed).metrics
        elif wants_rng:
            values = model(**params, rng=RandomStream(rep_seed))
        else:
            values = model(**params)
        row = dict(values) if isinstance(values, Mapping) else {"value": float(values)}
        for k, v in row.items():
            out.setdefault(k, [math.nan] * j).append(float(v))
        for k in out:
            if len(out[k]) < j + 1:
                out[k].append(math.nan)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def sobol_indices(
    model: Callable[..., Any] | Model,
    parameters: Mapping[str, DistributionLike | Mapping[str, Any]],
    n: int = 1024,
    *,
    outputs: Sequence[str] | None = None,
    seed: int = 0,
    fixed: Mapping[str, Any] | None = None,
    vectorized: bool = False,
    confidence: float = 0.95,
    n_bootstrap: int = 500,
) -> SensitivityResult:
    """First-order and total-effect Sobol indices with bootstrap confidence intervals.

    Uses the Saltelli (2010) design: two independent sample matrices ``A`` and
    ``B`` of ``n`` rows plus, for each of the ``d`` parameters, ``A`` with that
    column taken from ``B``. First-order indices use the Saltelli (2010)
    estimator, total effects the Jansen (1999) estimator. Inputs are assumed
    independent. For simulation models every matrix row ``j`` uses the same
    replication seed (common random numbers), which keeps simulation noise
    from swamping the index estimates; noise still adds some bias, so use
    enough replication length or average several runs per row if outputs are
    very noisy.

    ``value`` is the index estimate (``method`` = ``sobol-first`` or
    ``sobol-total``). Estimates can fall slightly outside [0, 1] with small
    ``n``; the bootstrap interval shows how precise they are.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    names = list(parameters)
    d = len(names)
    if d == 0:
        raise ValueError("need at least one uncertain parameter")
    a = sample_inputs(parameters, n, derive_seed(seed, "sobol-A"))
    b = sample_inputs(parameters, n, derive_seed(seed, "sobol-B"))
    blocks = [a, b] + [{k: (b[k] if k == name else a[k]) for k in names} for name in names]
    design = {k: np.concatenate([blk[k] for blk in blocks]) for k in names}
    total = n * (d + 2)
    y_all = _evaluate_rows(
        model,
        design,
        total,
        fixed=dict(fixed or {}),
        vectorized=vectorized,
        seed=seed,
        crn_period=n,
    )
    rng = np.random.default_rng(derive_seed(seed, "sobol-bootstrap"))
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))
    alpha = (1 - confidence) / 2
    rows: list[SensitivityRow] = []
    info: dict[str, Any] = {"n": n, "evaluations": total, "variance": {}}
    for o in outputs if outputs is not None else list(y_all):
        if o not in y_all:
            raise KeyError(f"model produced no output {o!r}")
        y = y_all[o].reshape(d + 2, n)
        f_a, f_b, f_ab = y[0], y[1], y[2:]  # (n,), (n,), (d, n)
        var = np.var(np.concatenate([f_a, f_b]), ddof=1)
        s1 = np.mean(f_b * (f_ab - f_a), axis=1) / var
        st = 0.5 * np.mean((f_a - f_ab) ** 2, axis=1) / var
        # bootstrap over rows: shapes (B, n) and (d, B, n)
        fa_b, fb_b, fab_b = f_a[boot_idx], f_b[boot_idx], f_ab[:, boot_idx]
        var_b = np.var(np.concatenate([fa_b, fb_b], axis=1), axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = np.mean(fb_b * (fab_b - fa_b), axis=2) / var_b
            bt = 0.5 * np.mean((fa_b - fab_b) ** 2, axis=2) / var_b
        info["variance"][o] = float(var)
        for i, name in enumerate(names):
            lo1, hi1 = np.nanquantile(b1[i], [alpha, 1 - alpha])
            lot, hit = np.nanquantile(bt[i], [alpha, 1 - alpha])
            rows.append(
                SensitivityRow(
                    name, o, "sobol-first", float(s1[i]), float(lo1), float(hi1), f"N={n}"
                )
            )
            rows.append(
                SensitivityRow(
                    name, o, "sobol-total", float(st[i]), float(lot), float(hit), f"N={n}"
                )
            )
    return SensitivityResult(rows, info)


def morris_screening(
    model: Callable[..., Any] | Model,
    ranges: Mapping[str, tuple[float, float]],
    r: int = 20,
    *,
    levels: int = 4,
    outputs: Sequence[str] | None = None,
    seed: int = 0,
    fixed: Mapping[str, Any] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 500,
) -> SensitivityResult:
    """Morris elementary-effects screening over ``ranges`` (``{name: (low, high)}``).

    Builds ``r`` random one-at-a-time trajectories on a ``levels``-level grid
    (Morris 1991), each costing ``d + 1`` evaluations. Effects are measured
    on inputs scaled to [0, 1], so they are comparable across parameters and
    in output units. Per parameter and output it reports:

    * ``morris-mu_star`` - mean absolute effect (Campolongo et al. 2007), the
      importance measure, with a bootstrap CI over trajectories;
    * ``morris-mu`` - mean signed effect (direction; can cancel out);
    * ``morris-sigma`` - standard deviation of effects: large relative to
      ``mu_star`` means non-linear effects or interactions.

    Integer model parameters are rounded. For simulation models every point of
    a trajectory uses the same replication seed (common random numbers).
    """
    if r < 2:
        raise ValueError("r must be >= 2")
    if levels < 2 or levels % 2:
        raise ValueError("levels must be an even number >= 2")
    names = list(ranges)
    d = len(names)
    if d == 0:
        raise ValueError("need at least one parameter range")
    lows = np.array([float(ranges[k][0]) for k in names])
    highs = np.array([float(ranges[k][1]) for k in names])
    if np.any(highs <= lows):
        raise ValueError("every range needs low < high")
    is_int = [
        isinstance(model, Model) and k in model.parameters and model.parameters[k].kind == "int"
        for k in names
    ]
    rng = np.random.default_rng(derive_seed(seed, "morris"))
    delta = levels / (2 * (levels - 1))
    starts = np.arange(levels // 2) / (levels - 1)  # grid points x with x + delta <= 1
    points = np.empty((r, d + 1, d))
    moved = np.empty((r, d), dtype=int)
    steps = np.empty((r, d))
    for t in range(r):
        x = rng.choice(starts, size=d)
        down = rng.random(d) < 0.5
        x[down] += delta  # start at the top for factors that will step down
        points[t, 0] = x
        order = rng.permutation(d)
        for k, i in enumerate(order):
            step = -delta if down[i] else delta
            x = x.copy()
            x[i] += step
            points[t, k + 1] = x
            moved[t, k] = i
            steps[t, k] = step
    unit = points.reshape(r * (d + 1), d)
    real = lows + unit * (highs - lows)
    inputs: dict[str, np.ndarray[Any, Any]] = {}
    for j, pname in enumerate(names):
        col = real[:, j]
        inputs[pname] = np.round(col).astype(int) if is_int[j] else col
    y_all = _evaluate_rows(
        model,
        inputs,
        r * (d + 1),
        fixed=dict(fixed or {}),
        vectorized=False,
        seed=seed,
        crn_period=r * (d + 1),
        seed_index=[row // (d + 1) for row in range(r * (d + 1))],
    )
    boot = np.random.default_rng(derive_seed(seed, "morris-bootstrap")).integers(
        0, r, size=(n_bootstrap, r)
    )
    alpha = (1 - confidence) / 2
    rows: list[SensitivityRow] = []
    for o in outputs if outputs is not None else list(y_all):
        if o not in y_all:
            raise KeyError(f"model produced no output {o!r}")
        y = y_all[o].reshape(r, d + 1)
        ee = np.empty((r, d))
        for t in range(r):
            for k in range(d):
                ee[t, moved[t, k]] = (y[t, k + 1] - y[t, k]) / steps[t, k]
        mu = np.nanmean(ee, axis=0)
        mu_star = np.nanmean(np.abs(ee), axis=0)
        sigma = np.nanstd(ee, axis=0, ddof=1)
        boot_star = np.nanmean(np.abs(ee)[boot], axis=1)  # (B, d)
        for i, name in enumerate(names):
            lo, hi = np.nanquantile(boot_star[:, i], [alpha, 1 - alpha])
            setting = f"range=[{lows[i]:g}, {highs[i]:g}]"
            rows += [
                SensitivityRow(
                    name, o, "morris-mu_star", float(mu_star[i]), float(lo), float(hi), setting
                ),
                SensitivityRow(name, o, "morris-mu", float(mu[i]), setting=setting),
                SensitivityRow(
                    name,
                    o,
                    "morris-sigma",
                    float(sigma[i]),
                    setting=setting,
                    detail=f"sigma/mu_star={sigma[i] / mu_star[i]:.2f}" if mu_star[i] > 0 else "",
                ),
            ]
    return SensitivityResult(
        rows, {"r": r, "levels": levels, "evaluations": r * (d + 1), "delta": delta}
    )
