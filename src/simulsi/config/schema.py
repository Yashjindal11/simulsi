"""Experiment configuration files (YAML) and model references.

Configuration is *data only*: files are parsed with ``yaml.safe_load`` and
validated against a strict schema (unknown keys are errors). The one way a
config can lead to code running is the ``model:`` reference, which names a
Python model explicitly - the same trust model as pointing ``pytest`` at a
test file. Model files referenced by path must live inside the config file's
directory unless the caller opts out, which prevents ``../../`` tricks.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from simulsi.core.model import Model
from simulsi.errors import ConfigError
from simulsi.scenarios.scenario import BASELINE, Scenario, check_unique_names, grid


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


OutputFormat = Literal["json", "csv", "parquet"]
_DEFAULT_FORMATS: tuple[OutputFormat, ...] = ("json", "csv")


class SimulationSection(_Strict):
    duration: float | None = Field(default=None, gt=0)
    warmup: float | None = Field(default=None, ge=0)
    seed: int = Field(default=0, ge=0)


class ExperimentSection(_Strict):
    replications: int = Field(default=10, ge=1, le=1_000_000)
    workers: int = Field(default=1, ge=1, le=512)
    common_random_numbers: bool = True
    confidence: float = Field(default=0.95, gt=0, lt=1)
    on_error: Literal["raise", "record"] = "raise"


class ScenarioSection(_Strict):
    name: str = Field(min_length=1, max_length=200)
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class OutputSection(_Strict):
    directory: str | None = None
    formats: list[OutputFormat] = Field(default_factory=lambda: list(_DEFAULT_FORMATS))


class ExperimentConfig(_Strict):
    model: str = Field(min_length=1)
    name: str | None = None
    description: str = ""
    simulation: SimulationSection = Field(default_factory=SimulationSection)
    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    parameters: dict[str, Any] = Field(default_factory=dict)
    scenarios: list[ScenarioSection] = Field(default_factory=list)
    grid: dict[str, list[Any]] | None = None
    baseline: str = BASELINE
    metrics: list[str] | None = None
    cost: dict[str, Any] | None = None
    output: OutputSection = Field(default_factory=OutputSection)

    @field_validator("grid")
    @classmethod
    def _grid_nonempty(cls, v: dict[str, list[Any]] | None) -> dict[str, list[Any]] | None:
        if v is not None:
            for k, values in v.items():
                if not values:
                    raise ValueError(f"grid axis {k!r} is empty")
        return v

    def build_scenarios(self) -> list[Scenario]:
        base = Scenario(BASELINE, dict(self.parameters), "base parameters")
        out: list[Scenario] = []
        if self.grid:
            out = grid(self.grid, base)
        else:
            out.append(base)
        for s in self.scenarios:
            if s.name == BASELINE:
                out[0] = base.derive(BASELINE, s.description, **s.parameters)
            else:
                out.append(base.derive(s.name, s.description, **s.parameters))
        check_unique_names(out)
        return out


def _format_pydantic(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)


def parse_config(data: Mapping[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration: {_format_pydantic(exc)}") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    if p.stat().st_size > 5_000_000:
        raise ConfigError("configuration file is larger than 5 MB; refusing to parse")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def load_config(path: str | Path) -> ExperimentConfig:
    return parse_config(load_yaml(path))


# -- model references ------------------------------------------------------------


def _model_from_module(module: Any, attr: str | None, ref: str) -> Model:
    if attr:
        if not hasattr(module, attr):
            raise ConfigError(f"{ref}: no attribute {attr!r}")
        obj = getattr(module, attr)
    elif isinstance(getattr(module, "model", None), Model):
        obj = module.model
    else:
        found = list({id(v): v for v in vars(module).values() if isinstance(v, Model)}.values())
        if len(found) != 1:
            raise ConfigError(
                f"{ref}: found {len(found)} Model objects; name one explicitly as 'file.py:attribute'"
            )
        obj = found[0]
    if not isinstance(obj, Model):
        raise ConfigError(f"{ref}: {attr!r} is a {type(obj).__name__}, not a simulsi Model")
    return obj


_LOADED_FILES: dict[str, str] = {}


def loaded_model_files() -> dict[str, str]:
    """Module name -> path of model files loaded by path (re-loaded in worker processes)."""
    return dict(_LOADED_FILES)


def preload_model_files(files: Mapping[str, str]) -> None:
    """Worker-process initializer: import user model files under the same module names."""
    for path in files.values():
        _load_file(Path(path))


def _load_file(path: Path) -> Any:
    digest = hashlib.sha1(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    name = f"simulsi_user_model_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # lets pickle find the model by module name
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    _LOADED_FILES[name] = str(path)
    return module


def resolve_model(
    ref: str, *, base_dir: str | Path | None = None, allow_outside: bool = True
) -> Model:
    """Resolve ``builtin:<name>``, ``path/to/file.py[:attr]`` or ``package.module:attr``.

    With ``allow_outside=False`` file references must resolve inside ``base_dir``.
    """
    ref = ref.strip()
    if ref.startswith("builtin:"):
        from simulsi.models import BUILTIN_MODELS

        key = ref.split(":", 1)[1]
        if key not in BUILTIN_MODELS:
            raise ConfigError(f"unknown builtin model {key!r}; available: {sorted(BUILTIN_MODELS)}")
        return BUILTIN_MODELS[key]
    target, _, attr = ref.partition(":") if not _looks_like_windows_drive(ref) else (ref, "", "")
    if target.endswith(".py"):
        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        path = (base / target).resolve()
        if not allow_outside and base not in path.parents:
            raise ConfigError(
                f"model file {target!r} is outside the configuration directory {base}"
            )
        if not path.is_file():
            raise ConfigError(f"model file not found: {path}")
        return _model_from_module(_load_file(path), attr or None, ref)
    if not all(part.isidentifier() for part in target.split(".")):
        raise ConfigError(f"invalid model reference {ref!r}")
    try:
        module = importlib.import_module(target)
    except ImportError as exc:
        raise ConfigError(f"cannot import module {target!r}: {exc}") from exc
    return _model_from_module(module, attr or None, ref)


def _looks_like_windows_drive(ref: str) -> bool:
    return len(ref) > 2 and ref[1] == ":" and ref[0].isalpha() and ref[2] in "\\/"


def build_experiment(
    config: ExperimentConfig, *, base_dir: str | Path | None = None, allow_outside: bool = False
) -> Any:
    """Turn a validated config into a ready-to-run :class:`~simulsi.Experiment`."""
    from simulsi.experiments.experiment import Experiment

    m = resolve_model(config.model, base_dir=base_dir, allow_outside=allow_outside)
    sim = config.simulation
    if sim.duration is not None or sim.warmup is not None:
        m = m.with_options(duration=sim.duration, warmup=sim.warmup)
    exp = Experiment(
        m,
        config.build_scenarios(),
        replications=config.experiment.replications,
        seed=sim.seed,
        workers=config.experiment.workers,
        name=config.name or m.name,
        common_random_numbers=config.experiment.common_random_numbers,
        on_error=config.experiment.on_error,
    )
    issues = exp.validate()
    if config.baseline not in {s.name for s in exp.scenarios} and not config.grid:
        issues.append(f"baseline {config.baseline!r} is not one of the scenarios")
    if issues:
        raise ConfigError("invalid experiment configuration:\n  - " + "\n  - ".join(issues))
    return exp


def load_experiment(
    path: str | Path, *, allow_outside: bool = False
) -> tuple[Any, ExperimentConfig]:
    p = Path(path)
    cfg = load_config(p)
    return build_experiment(cfg, base_dir=p.parent, allow_outside=allow_outside), cfg
