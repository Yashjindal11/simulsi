"""Experiments: scenarios x replications, with provenance, checkpointing and parallelism."""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import pickle
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from simulsi.core.model import Model
from simulsi.errors import ConfigError
from simulsi.experiments.provenance import environment, git_commit, utc_now
from simulsi.randomness.stream import derive_seed
from simulsi.scenarios.scenario import BASELINE, Scenario, check_unique_names
from simulsi.scenarios.scenario import grid as _grid
from simulsi.serialization.io import read_json, to_jsonable, write_csv, write_json, write_parquet
from simulsi.statistics.core import ReplicationAdvice, required_replications, summarize

FORMAT_VERSION = 1


def new_experiment_id() -> str:
    return f"EXP-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


@dataclass
class ReplicationRecord:
    scenario: str
    replication: int
    seed: int
    parameters: dict[str, Any]
    metrics: dict[str, float]
    runtime: float
    events: int = 0
    warnings: list[str] = field(default_factory=list)
    status: Literal["ok", "error"] = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReplicationRecord:
        return cls(
            scenario=str(d["scenario"]),
            replication=int(d["replication"]),
            seed=int(d["seed"]),
            parameters=dict(d.get("parameters") or {}),
            metrics={
                k: (math.nan if v is None else float(v))
                for k, v in (d.get("metrics") or {}).items()
            },
            runtime=float(d.get("runtime") or 0.0),
            events=int(d.get("events") or 0),
            warnings=list(d.get("warnings") or []),
            status="error" if d.get("status") == "error" else "ok",
            error=d.get("error"),
        )


@dataclass
class ExperimentMetadata:
    experiment_id: str
    name: str
    model_name: str
    model_version: str
    seed: int
    replications: int
    common_random_numbers: bool
    scenarios: dict[str, dict[str, Any]]
    timestamp: str
    git: dict[str, Any] | None
    environment: dict[str, Any]
    runtime: float = 0.0
    workers: int = 1
    duration: float | None = None
    warmup: float = 0.0
    format_version: int = FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExperimentMetadata:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# -- worker entry point (module level so it pickles) ---------------------------


def _run_one(
    model: Model,
    scenario: str,
    params: dict[str, Any],
    replication: int,
    seed: int,
    raise_errors: bool,
) -> ReplicationRecord:
    t0 = time.perf_counter()
    try:
        result = model.simulate(params, seed=seed)
    except Exception as exc:
        if raise_errors:
            raise
        return ReplicationRecord(
            scenario,
            replication,
            seed,
            params,
            {},
            time.perf_counter() - t0,
            0,
            [],
            "error",
            "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
    return ReplicationRecord(
        scenario,
        replication,
        seed,
        params,
        result.metrics,
        time.perf_counter() - t0,
        result.events_processed,
        list(result.warnings),
    )


def _run_task(args: tuple[Any, ...]) -> ReplicationRecord:
    return _run_one(*args)


class Experiment:
    """Run a model across scenarios and independent replications.

    Seeds: replication ``r`` uses ``derive_seed(seed, "replication", r)``.
    With ``common_random_numbers=True`` (default) that seed is shared by all
    scenarios, so scenario comparisons are *paired* and much less noisy.
    Each replication is independent and fully determined by its seed, so
    results are identical whether run serially or with ``workers > 1``.
    """

    def __init__(
        self,
        model: Model,
        scenarios: Scenario | Sequence[Scenario] | None = None,
        *,
        replications: int = 10,
        seed: int = 0,
        workers: int = 1,
        name: str | None = None,
        common_random_numbers: bool = True,
        on_error: Literal["raise", "record"] = "raise",
    ) -> None:
        self.model = model
        if scenarios is None:
            self.scenarios = [Scenario(BASELINE, {}, "model defaults")]
        elif isinstance(scenarios, Scenario):
            self.scenarios = [scenarios]
        else:
            self.scenarios = list(scenarios)
        self.replications = replications
        self.seed = seed
        self.workers = workers
        self.name = name or model.name
        self.common_random_numbers = common_random_numbers
        self.on_error = on_error

    def grid(
        self, axes: Mapping[str, Sequence[Any]], base: Scenario | None = None
    ) -> list[Scenario]:
        """Replace the scenarios with a full-factorial grid and return them."""
        self.scenarios = _grid(axes, base)
        return self.scenarios

    def replication_seed(self, scenario: str, replication: int) -> int:
        if self.common_random_numbers:
            return derive_seed(self.seed, "replication", replication)
        return derive_seed(self.seed, "scenario", scenario, replication)

    def validate(self) -> list[str]:
        issues: list[str] = []
        try:
            check_unique_names(self.scenarios)
        except ConfigError as exc:
            issues.append(str(exc))
        for s in self.scenarios:
            issues += [
                f"scenario {s.name!r}: {i}" for i in self.model.check_parameters(s.parameters)
            ]
        if self.replications < 1:
            issues.append("replications must be >= 1")
        if self.workers < 1:
            issues.append("workers must be >= 1")
        if self.model.duration is None:
            issues.append("model has no duration; set Model(duration=...) so runs terminate")
        return issues

    def run(
        self,
        scenario: Scenario | Sequence[Scenario] | None = None,
        *,
        replications: int | None = None,
        seed: int | None = None,
        workers: int | None = None,
        checkpoint: str | Path | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ExperimentResult:
        """Run every (scenario, replication) pair and collect the results.

        ``checkpoint`` names a JSON-lines file: finished replications are
        appended as they complete, and a re-run with the same file skips them
        (after checking that the experiment definition matches).
        """
        if scenario is not None:
            self.scenarios = [scenario] if isinstance(scenario, Scenario) else list(scenario)
        if replications is not None:
            self.replications = replications
        if seed is not None:
            self.seed = seed
        if workers is not None:
            self.workers = workers
        issues = self.validate()
        if issues:
            raise ConfigError("invalid experiment:\n  - " + "\n  - ".join(issues))

        resolved = {s.name: self.model.resolve(s.parameters).to_jsonable() for s in self.scenarios}
        meta = ExperimentMetadata(
            experiment_id=new_experiment_id(),
            name=self.name,
            model_name=self.model.name,
            model_version=self.model.version,
            seed=self.seed,
            replications=self.replications,
            common_random_numbers=self.common_random_numbers,
            scenarios={s.name: dict(s.parameters) for s in self.scenarios},
            timestamp=utc_now(),
            git=git_commit(),
            environment=environment(),
            workers=self.workers,
            duration=self.model.duration,
            warmup=self.model.warmup,
        )
        tasks = [
            (s.name, r, self.replication_seed(s.name, r))
            for s in self.scenarios
            for r in range(self.replications)
        ]
        done: dict[tuple[str, int], ReplicationRecord] = {}
        ckpt = _Checkpoint(Path(checkpoint), meta) if checkpoint is not None else None
        if ckpt is not None:
            done = ckpt.load()
        todo = [t for t in tasks if (t[0], t[1]) not in done]
        params_by_scenario = {s.name: dict(s.parameters) for s in self.scenarios}
        raise_errors = self.on_error == "raise"
        t0 = time.perf_counter()
        completed = len(tasks) - len(todo)

        def handle(rec: ReplicationRecord) -> None:
            nonlocal completed
            rec.parameters = resolved[rec.scenario]
            done[(rec.scenario, rec.replication)] = rec
            if ckpt is not None:
                ckpt.append(rec)
            completed += 1
            if progress is not None:
                progress(completed, len(tasks))

        args = [
            (self.model, name, params_by_scenario[name], r, s, raise_errors) for name, r, s in todo
        ]
        if self.workers == 1 or len(args) <= 1:
            for a in args:
                handle(_run_task(a))
        else:
            _check_picklable(self.model)
            chunk = max(1, len(args) // (self.workers * 4))
            with cf.ProcessPoolExecutor(max_workers=self.workers) as pool:
                for rec in pool.map(_run_task, args, chunksize=chunk):
                    handle(rec)
        meta.runtime = time.perf_counter() - t0
        records = [done[(n, r)] for n, r, _ in tasks]
        return ExperimentResult(meta, records)


def _check_picklable(model: Model) -> None:
    try:
        pickle.dumps(model)
    except Exception as exc:
        raise ConfigError(
            "workers > 1 needs a picklable model: define the build function at module "
            "level (not a lambda or nested function) and guard scripts with "
            f"`if __name__ == '__main__':`. ({exc})"
        ) from exc


class _Checkpoint:
    def __init__(self, path: Path, meta: ExperimentMetadata) -> None:
        self.path = path
        self.fingerprint = {
            "model": meta.model_name,
            "model_version": meta.model_version,
            "seed": meta.seed,
            "crn": meta.common_random_numbers,
            "scenarios": to_jsonable(meta.scenarios),
        }

    def load(self) -> dict[tuple[str, int], ReplicationRecord]:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"fingerprint": self.fingerprint}) + "\n")
            return {}
        out: dict[tuple[str, int], ReplicationRecord] = {}
        with self.path.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline() or "{}")
            if header.get("fingerprint") != self.fingerprint:
                raise ConfigError(
                    f"checkpoint {self.path} belongs to a different experiment definition; "
                    "use a new checkpoint file"
                )
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = ReplicationRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue  # a torn last line from an interrupted write
                out[(rec.scenario, rec.replication)] = rec
        return out

    def append(self, rec: ReplicationRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(to_jsonable(rec.to_dict()), allow_nan=False) + "\n")


class ExperimentResult:
    """All replication records plus provenance, with analysis and export helpers."""

    def __init__(self, metadata: ExperimentMetadata, records: Sequence[ReplicationRecord]) -> None:
        self.metadata = metadata
        self.records = list(records)

    # -- access ----------------------------------------------------------------

    @property
    def scenarios(self) -> list[str]:
        return list(dict.fromkeys(r.scenario for r in self.records))

    @property
    def metric_names(self) -> list[str]:
        names: dict[str, None] = {}
        for r in self.records:
            for k in r.metrics:
                names.setdefault(k, None)
        return sorted(names)

    @property
    def errors(self) -> list[ReplicationRecord]:
        return [r for r in self.records if r.status == "error"]

    def derive(self, fn: Callable[[Mapping[str, float]], Mapping[str, float]]) -> ExperimentResult:
        """Add derived metrics to every successful record in place, e.g. ``result.derive(costs.metrics)``."""
        for r in self.records:
            if r.status == "ok":
                r.metrics.update(fn(r.metrics))
        return self

    def values(
        self, metric: str, scenario: str | None = None
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Per-replication values of ``metric`` (ordered by replication; NaN if missing)."""
        sc = scenario if scenario is not None else self.scenarios[0]
        recs = sorted(
            (r for r in self.records if r.scenario == sc and r.status == "ok"),
            key=lambda r: r.replication,
        )
        if not recs and sc not in self.scenarios:
            raise KeyError(f"no scenario {sc!r}; have {self.scenarios}")
        return np.asarray([r.metrics.get(metric, math.nan) for r in recs], dtype=float)

    def seeds(self, scenario: str) -> list[int]:
        return [
            r.seed
            for r in sorted(self.records, key=lambda r: r.replication)
            if r.scenario == scenario
        ]

    # -- analysis --------------------------------------------------------------

    def summary(
        self,
        metrics: Iterable[str] | None = None,
        *,
        confidence: float = 0.95,
        scenarios: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """One row per (scenario, metric) with mean, std and a t confidence interval."""
        rows = []
        names = list(metrics) if metrics is not None else self.metric_names
        for sc in scenarios if scenarios is not None else self.scenarios:
            for m in names:
                s = summarize(self.values(m, sc), confidence)
                rows.append(
                    {
                        "scenario": sc,
                        "metric": m,
                        "n": s.n,
                        "mean": s.mean,
                        "std": s.std,
                        "ci_low": s.ci_low,
                        "ci_high": s.ci_high,
                        "half_width": s.half_width,
                        "min": s.min,
                        "median": s.median,
                        "max": s.max,
                        "confidence": confidence,
                    }
                )
        return rows

    def replication_advice(
        self,
        metric: str,
        scenario: str | None = None,
        *,
        relative_precision: float = 0.05,
        confidence: float = 0.95,
    ) -> ReplicationAdvice:
        return required_replications(self.values(metric, scenario), relative_precision, confidence)

    def compare(
        self,
        baseline: str = BASELINE,
        scenarios: Iterable[str] | None = None,
        metrics: Iterable[str] | None = None,
        *,
        confidence: float = 0.95,
    ) -> Any:
        from simulsi.analysis.comparison import compare

        return compare(self, baseline, scenarios, metrics, confidence=confidence)

    def format_summary(self, metrics: Iterable[str] | None = None, confidence: float = 0.95) -> str:
        from simulsi.analysis.report import format_table

        rows = self.summary(metrics, confidence=confidence)
        return format_table(
            rows,
            ["scenario", "metric", "n", "mean", "ci_low", "ci_high", "std"],
        )

    # -- export ----------------------------------------------------------------

    def rows(self) -> list[dict[str, Any]]:
        """Long-form-friendly wide rows: one per replication with metric columns."""
        out = []
        for r in self.records:
            row: dict[str, Any] = {
                "experiment_id": self.metadata.experiment_id,
                "scenario": r.scenario,
                "replication": r.replication,
                "seed": r.seed,
                "runtime": r.runtime,
                "events": r.events,
                "status": r.status,
            }
            for k, v in r.parameters.items():
                row[f"param.{k}"] = v
            row.update(r.metrics)
            out.append(row)
        return out

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe needs: pip install 'simulsi[pandas]'") from exc
        return pd.DataFrame(self.rows())

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "summary": self.summary(),
        }

    def to_json(self, path: str | Path) -> Path:
        return write_json(self.to_dict(), path)

    def to_csv(self, path: str | Path) -> Path:
        return write_csv(self.rows(), path)

    def to_parquet(self, path: str | Path) -> Path:
        return write_parquet(self.rows(), path)

    def save(self, directory: str | Path, *, formats: Sequence[str] = ("json", "csv")) -> Path:
        """Write ``experiment.json`` (+ ``replications.csv`` / ``.parquet``) into ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.to_json(d / "experiment.json")
        if "csv" in formats:
            self.to_csv(d / "replications.csv")
            write_csv(self.summary(), d / "summary.csv")
        if "parquet" in formats:
            self.to_parquet(d / "replications.parquet")
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentResult:
        if "metadata" not in data or "records" not in data:
            raise ConfigError("not a SimulSI experiment file (missing metadata/records)")
        return cls(
            ExperimentMetadata.from_dict(data["metadata"]),
            [ReplicationRecord.from_dict(r) for r in data["records"]],
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentResult:
        """Load from an ``experiment.json`` file or a directory containing one."""
        p = Path(path)
        if p.is_dir():
            p = p / "experiment.json"
        return cls.from_dict(read_json(p))

    def __repr__(self) -> str:
        return (
            f"ExperimentResult({self.metadata.experiment_id}, scenarios={len(self.scenarios)}, "
            f"records={len(self.records)})"
        )
