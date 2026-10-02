"""Optional plotting helpers (matplotlib or Plotly).

Install with ``pip install 'simulsi[viz]'`` (matplotlib) or
``'simulsi[plotly]'``. Every function returns the figure object and never
calls ``show()``, so it works in scripts, notebooks and tests alike.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from simulsi.statistics.core import convergence as _convergence
from simulsi.statistics.core import summarize

if TYPE_CHECKING:
    from simulsi.analysis.comparison import Comparison
    from simulsi.analysis.sensitivity import SensitivityResult
    from simulsi.core.simulation import SimulationResult
    from simulsi.core.trace import EventLog

Backend = Literal["matplotlib", "plotly"]


def _mpl() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install 'simulsi[viz]'") from exc
    return plt


def _go() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotly backend needs: pip install 'simulsi[plotly]'") from exc
    return go


class _Figure:
    """Minimal drawing surface over both backends."""

    def __init__(
        self, backend: Backend, title: str, xlabel: str, ylabel: str, ax: Any = None
    ) -> None:
        self.backend = backend
        if backend == "plotly":
            self.go = _go()
            self.fig = self.go.Figure()
            self.fig.update_layout(
                title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white"
            )
        elif backend == "matplotlib":
            plt = _mpl()
            if ax is None:
                self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
            else:
                self.ax, self.fig = ax, ax.figure
            self.ax.set_title(title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def step(self, x: Sequence[float], y: Sequence[float], label: str, dash: bool = False) -> None:
        if self.backend == "plotly":
            self.fig.add_trace(
                self.go.Scatter(
                    x=list(x),
                    y=list(y),
                    name=label,
                    mode="lines",
                    line={"shape": "hv", "dash": "dash" if dash else "solid"},
                )
            )
        else:
            self.ax.step(x, y, where="post", label=label, linestyle="--" if dash else "-")

    def line(
        self,
        x: Sequence[float],
        y: Sequence[float],
        label: str,
        band: tuple[Any, Any] | None = None,
    ) -> None:
        if self.backend == "plotly":
            if band is not None:
                self.fig.add_trace(
                    self.go.Scatter(
                        x=list(x) + list(x)[::-1],
                        y=list(band[1]) + list(band[0])[::-1],
                        fill="toself",
                        opacity=0.2,
                        line={"width": 0},
                        name=f"{label} CI",
                    )
                )
            self.fig.add_trace(self.go.Scatter(x=list(x), y=list(y), name=label, mode="lines"))
        else:
            self.ax.plot(x, y, label=label)
            if band is not None:
                self.ax.fill_between(x, band[0], band[1], alpha=0.2)

    def scatter(self, x: Sequence[float], y: Sequence[Any], label: str) -> None:
        if self.backend == "plotly":
            self.fig.add_trace(
                self.go.Scatter(
                    x=list(x), y=list(y), name=label, mode="markers", marker={"size": 4}
                )
            )
        else:
            self.ax.scatter(x, y, s=6, label=label)

    def hist(self, values: Sequence[float], bins: int, label: str) -> None:
        if self.backend == "plotly":
            self.fig.add_trace(
                self.go.Histogram(x=list(values), nbinsx=bins, name=label, opacity=0.75)
            )
        else:
            self.ax.hist(values, bins=bins, alpha=0.75, label=label)

    def vline(self, x: float, label: str, dash: bool = True) -> None:
        if self.backend == "plotly":
            self.fig.add_vline(x=x, line_dash="dash" if dash else "solid", annotation_text=label)
        else:
            self.ax.axvline(x, linestyle="--" if dash else "-", color="black", label=label)

    def barh(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        err: Sequence[tuple[float, float]] | None,
    ) -> None:
        if self.backend == "plotly":
            error_x = None
            if err is not None:
                error_x = {
                    "type": "data",
                    "symmetric": False,
                    "array": [hi for _, hi in err],
                    "arrayminus": [lo for lo, _ in err],
                }
            self.fig.add_trace(
                self.go.Bar(x=list(values), y=list(labels), orientation="h", error_x=error_x)
            )
        else:
            xerr = None if err is None else np.array(err).T
            self.ax.barh(list(labels), list(values), xerr=xerr, capsize=3)
            self.ax.axvline(0, color="black", linewidth=0.8)

    def segments(self, rows: Sequence[tuple[str, float, float, str]]) -> None:
        """Horizontal segments ``(row_label, start, end, category)`` - a Gantt chart."""
        cats = list(dict.fromkeys(r[3] for r in rows))
        labels = list(dict.fromkeys(r[0] for r in rows))
        if self.backend == "plotly":
            for c in cats:
                sel = [r for r in rows if r[3] == c]
                self.fig.add_trace(
                    self.go.Bar(
                        base=[r[1] for r in sel],
                        x=[r[2] - r[1] for r in sel],
                        y=[r[0] for r in sel],
                        orientation="h",
                        name=c,
                    )
                )
            self.fig.update_layout(barmode="overlay")
        else:
            cmap = _mpl().get_cmap("tab10")
            for i, c in enumerate(cats):
                sel = [r for r in rows if r[3] == c]
                self.ax.barh(
                    [labels.index(r[0]) for r in sel],
                    [r[2] - r[1] for r in sel],
                    left=[r[1] for r in sel],
                    color=cmap(i % 10),
                    label=c,
                    height=0.6,
                )
            self.ax.set_yticks(range(len(labels)), labels)

    def done(self) -> Any:
        if self.backend == "matplotlib":
            handles, _ = self.ax.get_legend_handles_labels()
            if handles:
                self.ax.legend(loc="best", fontsize="small")
            self.fig.tight_layout()
        return self.fig


def _close_series(
    points: Sequence[tuple[float, float]], end: float
) -> tuple[list[float], list[float]]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if xs and end > xs[-1]:
        xs.append(end)
        ys.append(ys[-1])
    return xs, ys


def plot_series(
    result: SimulationResult,
    keys: Sequence[str] | None = None,
    *,
    title: str = "Time series",
    ylabel: str = "value",
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Step plot of recorded time series (see ``SimulationResult.series``)."""
    keys = list(keys) if keys is not None else sorted(result.series)
    if not keys:
        raise ValueError("no series recorded; run the simulation with record_series=True")
    f = _Figure(backend, title, "time", ylabel, ax)
    for k in keys:
        if k not in result.series:
            raise KeyError(f"no series {k!r}; have {sorted(result.series)}")
        f.step(*_close_series(result.series[k], result.end_time), label=k)
    return f.done()


def plot_queue_length(
    result: SimulationResult, *, backend: Backend = "matplotlib", ax: Any = None
) -> Any:
    keys = [
        k
        for k in sorted(result.series)
        if k.endswith(".queue_length") or (k.startswith("queue.") and k.endswith(".length"))
    ]
    return plot_series(
        result, keys, title="Queue length over time", ylabel="items waiting", backend=backend, ax=ax
    )


def plot_utilization(
    result: SimulationResult,
    resource: str | None = None,
    *,
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Busy units (solid) against available capacity (dashed) for each resource."""
    names = sorted(
        {
            k.split(".")[1]
            for k in result.series
            if k.startswith("resource.") and k.endswith(".busy")
        }
    )
    if resource is not None:
        names = [resource]
    f = _Figure(backend, "Resource utilization", "time", "units", ax)
    for n in names:
        util = result.metrics.get(f"resource.{n}.utilization", math.nan)
        f.step(
            *_close_series(result.series[f"resource.{n}.busy"], result.end_time),
            label=f"{n} busy (util {util:.0%})",
        )
        cap = result.series.get(f"resource.{n}.capacity")
        if cap:
            f.step(*_close_series(cap, result.end_time), label=f"{n} capacity", dash=True)
    return f.done()


def plot_timeline(
    log: EventLog,
    *,
    event_types: Sequence[str] | None = None,
    max_points: int = 20_000,
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Each logged event as a dot: x = time, y = event type."""
    recs = [r for r in log.records if event_types is None or r.event_type in event_types][
        :max_points
    ]
    f = _Figure(backend, "Event timeline", "time", "event type", ax)
    for et in sorted({r.event_type for r in recs}):
        sel = [r for r in recs if r.event_type == et]
        f.scatter([r.timestamp for r in sel], [et] * len(sel), label=et)
    return f.done()


def plot_entity_trajectories(
    log: EventLog,
    *,
    max_entities: int = 30,
    end_time: float | None = None,
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Gantt chart of entity states from ``entity.created`` / ``entity.state`` log records."""
    current: dict[str, tuple[str, float]] = {}
    order: list[str] = []
    rows: list[tuple[str, float, float, str]] = []
    last_t = 0.0
    for r in log.records:
        last_t = max(last_t, r.timestamp)
        if r.entity is None:
            continue
        if r.event_type == "entity.created":
            if len(order) >= max_entities:
                continue
            order.append(r.entity)
            current[r.entity] = (r.new_state or "created", r.timestamp)
        elif r.event_type == "entity.state" and r.entity in current:
            state, start = current[r.entity]
            rows.append((r.entity, start, r.timestamp, state))
            current[r.entity] = (r.new_state or "?", r.timestamp)
    end = end_time if end_time is not None else last_t
    for e, (state, start) in current.items():
        if state != "disposed" and end > start:
            rows.append((e, start, end, state))
    f = _Figure(backend, "Entity trajectories", "time", "entity", ax)
    f.segments(rows)
    return f.done()


def plot_distribution(
    values: Sequence[float] | np.ndarray[Any, Any],
    *,
    title: str = "Distribution",
    bins: int = 30,
    label: str = "value",
    confidence: float = 0.95,
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Histogram with the mean and its confidence interval marked."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    s = summarize(arr, confidence)
    f = _Figure(backend, title, label, "count", ax)
    f.hist(arr.tolist(), bins, label)
    f.vline(s.mean, f"mean {s.mean:.4g}", dash=False)
    if not math.isnan(s.ci_low):
        f.vline(s.ci_low, f"{confidence:.0%} CI")
        f.vline(s.ci_high, "")
    return f.done()


def plot_convergence(
    values: Sequence[float],
    *,
    title: str = "Convergence of the mean",
    confidence: float = 0.95,
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    c = _convergence(values, confidence)
    f = _Figure(backend, title, "replications", "running mean", ax)
    f.line(
        c.n.tolist(),
        c.running_mean.tolist(),
        "mean",
        band=(
            (c.running_mean - c.running_half_width).tolist(),
            (c.running_mean + c.running_half_width).tolist(),
        ),
    )
    return f.done()


def plot_comparison(
    comparison: Comparison,
    metric: str | None = None,
    *,
    relative: bool = False,
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Difference from baseline per scenario with confidence-interval error bars."""
    rows = [r for r in comparison.rows if metric is None or r.metric == metric]
    labels = [f"{r.scenario} | {r.metric}" if metric is None else r.scenario for r in rows]
    scale = [100 / abs(r.baseline_mean) if relative and r.baseline_mean else 1.0 for r in rows]
    vals = [r.absolute_difference * s for r, s in zip(rows, scale, strict=True)]
    err = [
        ((r.absolute_difference - r.ci_low) * s, (r.ci_high - r.absolute_difference) * s)
        for r, s in zip(rows, scale, strict=True)
    ]
    f = _Figure(
        backend,
        f"Difference vs {rows[0].baseline if rows else 'baseline'}",
        "% change" if relative else "difference",
        "",
        ax,
    )
    f.barh(labels, vals, err)
    return f.done()


def plot_sensitivity(
    result: SensitivityResult,
    output: str | None = None,
    method: str | None = None,
    *,
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Tornado chart of effect sizes, largest at the top."""
    rows = result.ranking(output, method)[::-1]
    labels = [f"{r.parameter} ({r.method})" if method is None else r.parameter for r in rows]
    has_ci = all(not math.isnan(r.ci_low) for r in rows)
    err = [(r.value - r.ci_low, r.ci_high - r.value) for r in rows] if has_ci else None
    f = _Figure(backend, f"Sensitivity{': ' + output if output else ''}", "effect", "", ax)
    f.barh(labels, [r.value for r in rows], err)
    return f.done()


def save_figure(fig: Any, path: str, **kwargs: Any) -> None:
    """Save either backend's figure (``.png``/``.svg``/``.pdf`` or Plotly ``.html``)."""
    if hasattr(fig, "write_html"):
        if path.endswith(".html"):
            fig.write_html(path, include_plotlyjs="cdn", **kwargs)
        else:
            fig.write_image(path, **kwargs)
    else:
        fig.savefig(path, dpi=kwargs.pop("dpi", 130), **kwargs)
