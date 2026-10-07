"""Execute the example notebooks' code cells (no Jupyter needed).

Cells run in order in one namespace inside a temporary directory; lines
starting with ``%`` or ``!`` (IPython magics) are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "examples" / "notebooks").glob("*.ipynb"))

pytestmark = pytest.mark.integration


def test_notebooks_exist() -> None:
    assert len(NOTEBOOKS) >= 3


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path: Path, tmp_path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert not cell.get("outputs"), (
                f"{path.name} cell {i}: commit notebooks without outputs"
            )
            src = "".join(cell["source"])
            code = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith(("%", "!"))
            )
            try:
                exec(compile(code, f"{path.name}[cell {i}]", "exec"), namespace)
            except Exception as exc:  # pragma: no cover - reported with context
                pytest.fail(f"{path.name} cell {i} failed: {type(exc).__name__}: {exc}\n---\n{src}")
    finally:
        os.chdir(cwd)
        import matplotlib.pyplot as plt

        plt.close("all")
