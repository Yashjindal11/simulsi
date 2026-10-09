"""Animated recordings of a run: time series that grow as simulated time passes (GIF or MP4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from simulsi.core.simulation import SimulationResult


def _step_values(points: Sequence[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    """Value of a step series at each time."""
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    idx = np.searchsorted(xs, times, side="right") - 1
    out = np.where(idx >= 0, ys[np.clip(idx, 0, len(ys) - 1)], np.nan)
    return np.asarray(out, dtype=float)


def animate_series(
    result: SimulationResult,
    path: str | Path,
    keys: Sequence[str] | None = None,
    *,
    frames: int = 120,
    fps: int = 20,
    title: str = "Simulation replay",
) -> Path:
    """Record the run's time series as an animation that sweeps through simulated time.

    ``path`` ending in ``.gif`` uses Pillow (installed with matplotlib); other
    suffixes such as ``.mp4`` need ffmpeg. Each frame shows every series up to
    that moment plus a moving time marker. Defaults to queue lengths and busy
    units.

    >>> from simulsi.models import mmc
    >>> res = mmc.simulate(seed=1, duration=50, record_series=True)
    >>> import tempfile, os
    >>> out = animate_series(res, os.path.join(tempfile.mkdtemp(), "run.gif"), frames=10)
    >>> out.stat().st_size > 0
    True
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("animations need matplotlib: pip install 'simulsi[viz]'") from exc
    if keys is None:
        keys = [
            k
            for k in sorted(result.series)
            if k.endswith((".queue_length", ".busy"))
            or (k.startswith("queue.") and k.endswith(".length"))
        ] or sorted(result.series)[:6]
    keys = [k for k in keys if result.series.get(k)]
    if not keys:
        raise ValueError("no time series to animate; simulate with record_series=True")
    if frames < 2:
        raise ValueError("frames must be >= 2")
    start, end = result.start_time, result.end_time
    times = np.linspace(start, end, 400)
    values = {k: _step_values(result.series[k], times) for k in keys}
    top = max(float(np.nanmax(v)) for v in values.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(start, end)
    ax.set_ylim(0, max(1.0, top * 1.1))
    ax.set_xlabel("time")
    ax.set_title(title)
    lines = {k: ax.step([], [], where="post", label=k)[0] for k in keys}
    marker = ax.axvline(start, color="0.5", lw=1, ls="--")
    clock = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()

    def draw(i: int) -> list[Any]:
        t = start + (end - start) * i / (frames - 1)
        mask = times <= t
        for k, line in lines.items():
            line.set_data(times[mask], values[k][mask])
        marker.set_xdata([t, t])
        clock.set_text(f"t = {t:,.1f}")
        return [*lines.values(), marker, clock]

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out = Path(path)
    writer = PillowWriter(fps=fps) if out.suffix.lower() == ".gif" else FFMpegWriter(fps=fps)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    return out
