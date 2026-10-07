"""Benchmark tracking for CI: measure, compare with the previous run, report.

    python benchmarks/track.py --output bench.json [--previous prev.json]
                               [--summary "$GITHUB_STEP_SUMMARY"] [--threshold 0.25]

Measures engine throughput (events and process models, best of ``--repeat``
runs), a small serial experiment and, if SimPy is installed, the SimulSI /
SimPy time ratio. With ``--previous`` it compares against an earlier result
and prints a GitHub ``::warning::`` for any throughput that dropped by more
than ``--threshold``. Shared CI runners are noisy, so this warns rather than
fails; look for regressions that persist across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from simulsi.benchmarks import run_engine_benchmark, run_experiment_benchmark
from simulsi.experiments.provenance import environment


def measure(repeat: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, size in (("events", 200_000), ("processes", 50_000)):
        best = max(run_engine_benchmark(name, size).events_per_second for _ in range(repeat))
        out[f"{name}_per_second"] = best
    t0 = time.perf_counter()
    exp = run_experiment_benchmark(replications=8, workers=1, duration=5_000.0)
    out["experiment_events_per_second"] = exp.events_per_second
    out["experiment_seconds"] = time.perf_counter() - t0
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from compare_simpy import best as best_of
        from compare_simpy import simpy_mm2, simulsi_mm2

        s_t, _ = best_of(simulsi_mm2, 30_000, repeat)
        p_t, _ = best_of(simpy_mm2, 30_000, repeat)
        out["simpy_time_ratio"] = s_t / p_t
    except ImportError:
        pass
    return out


HIGHER_IS_BETTER = ("events_per_second", "processes_per_second", "experiment_events_per_second")


def compare(
    current: dict[str, Any], previous: dict[str, Any] | None, threshold: float
) -> tuple[str, list[str]]:
    rows = ["| metric | previous | current | change |", "|---|---:|---:|---:|"]
    warnings = []
    prev = (previous or {}).get("results", {})
    for key, value in current["results"].items():
        old = prev.get(key)
        if isinstance(old, int | float) and old:
            change = value / old - 1
            rows.append(f"| {key} | {old:,.3f} | {value:,.3f} | {change:+.1%} |")
            worse = -change if key in HIGHER_IS_BETTER else change
            if worse > threshold:
                warnings.append(
                    f"{key} is {abs(change):.0%} {'lower' if key in HIGHER_IS_BETTER else 'higher'} than the previous run ({old:,.3f} -> {value:,.3f})"
                )
        else:
            rows.append(f"| {key} | - | {value:,.3f} | - |")
    return "\n".join(rows), warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--previous", type=Path)
    ap.add_argument(
        "--summary", type=Path, help="append a Markdown report (e.g. $GITHUB_STEP_SUMMARY)"
    )
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    current = {"environment": environment(), "results": measure(args.repeat)}
    args.output.write_text(json.dumps(current, indent=2), encoding="utf-8")
    previous = None
    if args.previous and args.previous.is_file():
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
    table, warnings = compare(current, previous, args.threshold)
    report = "## SimulSI benchmarks\n\n" + table + "\n"
    if previous is None:
        report += "\nNo previous result to compare with (first run, or artifact expired).\n"
    for w in warnings:
        print(f"::warning title=Benchmark regression::{w}")
        report += f"\n> **Possible regression:** {w}\n"
    print(report)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
