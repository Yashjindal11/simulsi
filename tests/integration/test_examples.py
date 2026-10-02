"""Every shipped example must build, run, be reproducible and pass validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from simulsi import Model
from simulsi.validation import validate_model

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NAMES = [
    "queue",
    "bank_queue",
    "hospital",
    "warehouse",
    "manufacturing",
    "transportation",
    "aviation",
]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_example_runs_and_is_reproducible(name: str) -> None:
    m = load(name).model
    assert isinstance(m, Model)
    short = m.with_options(duration=m.duration / 4, warmup=0.0) if m.duration else m
    a = short.simulate(seed=1)
    b = short.simulate(seed=1)
    assert a.events_processed > 100
    assert a.metrics == b.metrics or all(
        (x == y) or (x != x and y != y)
        for x, y in zip(a.metrics.values(), b.metrics.values(), strict=True)
    )
    report = validate_model(m, smoke_duration=(m.duration or 100) / 10)
    assert report.ok, report.format()
    assert not [i for i in report.issues if i.code == "resource-not-released"], report.format()
