"""Execute the ```python code blocks in the documentation.

Blocks in one file run in order in a shared namespace, inside a temporary
directory. Blocks fenced as ```py (or any other language) are not executed.
"""

from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "getting-started.md",
    ROOT / "docs" / "concepts.md",
    ROOT / "docs" / "experiments.md",
    ROOT / "docs" / "visualization.md",
    ROOT / "docs" / "extending.md",
    ROOT / "docs" / "research.md",
    ROOT / "docs" / "simpy-migration.md",
]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_doc_code_blocks_run(
    path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} not written yet")
    blocks = BLOCK.findall(path.read_text(encoding="utf-8"))
    assert blocks, f"no python blocks in {path.name}"
    module = types.ModuleType(f"doc_{path.stem.replace('-', '_')}")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for i, code in enumerate(blocks):
            try:
                exec(compile(code, f"{path.name}[block {i}]", "exec"), namespace)
            except Exception as exc:  # pragma: no cover - reported with context
                pytest.fail(
                    f"{path.name} block {i} failed: {type(exc).__name__}: {exc}\n---\n{code}"
                )
    finally:
        os.chdir(cwd)
        sys.modules.pop(module.__name__, None)
        import matplotlib.pyplot as plt

        plt.close("all")
    capsys.readouterr()
