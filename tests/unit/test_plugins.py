from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any

import pytest

import simulsi.models as models


def test_entry_point_models_are_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    eps = [
        EntryPoint("plugin_queue", "simulsi.models.queueing:mmc", "simulsi.models"),
        EntryPoint("broken", "no_such_module_xyz:model", "simulsi.models"),
        EntryPoint("not_a_model", "simulsi.models.queueing:erlang_c", "simulsi.models"),
        EntryPoint("mmc", "simulsi.models.queueing:mmc", "simulsi.models"),
    ]

    def fake(*, group: str) -> list[EntryPoint]:
        assert group == "simulsi.models"
        return eps

    monkeypatch.setattr("importlib.metadata.entry_points", fake)
    monkeypatch.setattr(models, "BUILTIN_MODELS", dict(models.BUILTIN_MODELS))
    with pytest.warns(UserWarning, match="could not be loaded"):
        errors = models.load_plugin_models()
    assert set(errors) == {"broken", "not_a_model"}
    assert models.BUILTIN_MODELS["plugin_queue"] is models.mmc


def test_plugin_model_resolves_as_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    from simulsi.config.schema import resolve_model

    registry: dict[str, Any] = dict(models.BUILTIN_MODELS)
    registry["bakery"] = models.mmc
    monkeypatch.setattr(models, "BUILTIN_MODELS", registry)
    assert resolve_model("builtin:bakery") is models.mmc
