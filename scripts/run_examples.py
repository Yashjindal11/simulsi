"""Run every example with a small number of replications (used in CI).

python scripts/run_examples.py            # quick mode
python scripts/run_examples.py --full     # the replication counts the examples ship with
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = [
    "queue",
    "bank_queue",
    "hospital",
    "warehouse",
    "manufacturing",
    "transportation",
    "aviation",
    "airline_ops",
]


def load(name: str):  # type: ignore[no-untyped-def]
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    full = "--full" in sys.argv
    for name in EXAMPLES:
        module = load(name)
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        if full:
            module.main()
        else:
            module.main(replications=3)
        print(f"--- {name}: {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
