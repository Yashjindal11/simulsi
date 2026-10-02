"""Run the benchmark suite and record results with the machine environment.

    python benchmarks/run_benchmarks.py                 # full suite (10k/100k/1M)
    python benchmarks/run_benchmarks.py --quick         # 10k/100k only

Writes benchmarks/results/latest.json and benchmarks/results/latest.md.
Numbers are only meaningful together with the recorded environment.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from simulsi.analysis.report import format_table
from simulsi.benchmarks import default_suite
from simulsi.experiments.provenance import environment
from simulsi.serialization import to_jsonable

OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    quick = "--quick" in sys.argv
    sizes = (10_000, 100_000) if quick else (10_000, 100_000, 1_000_000)
    results = default_suite(sizes, memory=True, workers=(1, 2, 4), replications=16)
    env = environment()
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    rows = [r.to_dict() for r in results]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(
        json.dumps(to_jsonable({"timestamp": stamp, "environment": env, "results": rows}), indent=2)
        + "\n"
    )
    table = format_table(
        rows, ["name", "size", "events", "seconds", "events_per_second", "peak_memory_mb"]
    )
    md = [
        "# Benchmark results",
        "",
        f"Measured {stamp} with `python benchmarks/run_benchmarks.py{' --quick' if quick else ''}`.",
        "",
        f"- Python {env['python']} ({env['implementation']}), NumPy {env['numpy']}",
        f"- {env['platform']}, {env['cpu_count']} logical CPUs",
        f"- SimulSI {env['simulsi']}",
        "",
        "```text",
        table,
        "```",
        "",
        "* `events`: pure engine overhead - callback events in 100 interleaved chains (size = events).",
        "* `processes`: M/M/2 queue with generator processes and a resource (size = customers).",
        "* `experiment(workers=N)`: 16 replications of the built-in M/M/1 model, 20,000 time units each (process start-up costs ~1 s on macOS, so parallelism pays off only for longer runs).",
        "* `peak_memory_mb`: tracemalloc peak during a separate run (only for sizes <= 100k).",
    ]
    (OUT / "latest.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
