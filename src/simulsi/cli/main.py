"""Command-line interface: ``simulsi <command>``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from simulsi._version import __version__
from simulsi.errors import ConfigError, SimulsiError

EXIT_OK, EXIT_ERROR, EXIT_INVALID = 0, 1, 2


def _print(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(f"error: {text}\n")


def _parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    """``key=value`` pairs; values are parsed as YAML scalars (safe) so numbers stay numbers."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--param expects key=value, got {item!r}")
        try:
            out[key.strip()] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"cannot parse value for {key!r}: {exc}") from exc
    return out


def _metrics_table(metrics: dict[str, float], only: Sequence[str] | None = None) -> str:
    from simulsi.analysis.report import format_table

    keys = [k for k in (only or sorted(metrics)) if k in metrics]
    return format_table([{"metric": k, "value": metrics[k]} for k in keys])


def _is_config(target: str) -> bool:
    return target.endswith((".yaml", ".yml"))


# -- commands ----------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    from simulsi.cli import templates

    d = Path(args.directory)
    d.mkdir(parents=True, exist_ok=True)
    files = {
        "model.py": templates.MODEL_PY,
        "experiment.yaml": templates.EXPERIMENT_YAML,
        "README.md": templates.README_MD.format(name=d.resolve().name),
    }
    existing = [f for f in files if (d / f).exists()]
    if existing and not args.force:
        _err(f"{', '.join(existing)} already exist in {d}; use --force to overwrite")
        return EXIT_ERROR
    for name, content in files.items():
        (d / name).write_text(content, encoding="utf-8")
    _print(f"created SimulSI project in {d}/ ({', '.join(files)})")
    _print(
        f"next: cd {d} && simulsi validate experiment.yaml && simulsi experiment experiment.yaml"
    )
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    from simulsi.config.schema import load_experiment, resolve_model
    from simulsi.validation import validate_model

    target: str = args.target
    try:
        if _is_config(target):
            exp, _cfg = load_experiment(target, allow_outside=args.allow_outside)
            _print(
                f"configuration OK: {len(exp.scenarios)} scenario(s) x {exp.replications} replication(s)"
            )
            reports = [
                (s.name, validate_model(exp.model, s.parameters, smoke_duration=args.smoke))
                for s in exp.scenarios
            ]
        else:
            m = resolve_model(target)
            reports = [
                ("model", validate_model(m, _parse_params(args.param), smoke_duration=args.smoke))
            ]
    except (ConfigError, SimulsiError) as exc:
        _err(str(exc))
        return EXIT_INVALID
    ok = True
    for name, rep in reports:
        ok = ok and rep.ok
        _print(f"[{name}] {rep.format()}")
    return EXIT_OK if ok else EXIT_INVALID


def cmd_run(args: argparse.Namespace) -> int:
    from simulsi.config.schema import load_config, resolve_model

    params = _parse_params(args.param)
    target: str = args.target
    if _is_config(target):
        cfg = load_config(target)
        m = resolve_model(cfg.model, base_dir=Path(target).parent, allow_outside=args.allow_outside)
        params = {**cfg.parameters, **params}
        seed = args.seed if args.seed is not None else cfg.simulation.seed
        duration = args.duration or cfg.simulation.duration
    else:
        m = resolve_model(target)
        seed = args.seed if args.seed is not None else 0
        duration = args.duration
    result = m.simulate(
        params, seed=seed, duration=duration, warmup=args.warmup, trace=bool(args.trace)
    )
    if args.trace and result.log is not None:
        p = result.log.export(args.trace)
        sys.stderr.write(f"wrote {len(result.log)} log records to {p}\n")
    if args.json:
        from simulsi.serialization.io import to_jsonable

        _print(json.dumps(to_jsonable(result.to_dict()), indent=2, allow_nan=False))
        return EXIT_OK
    _print(
        f"{m.name}: seed={result.seed} t=[{result.start_time:g}, {result.end_time:g}] "
        f"events={result.events_processed} wall={result.wall_time:.3f}s"
    )
    _print(_metrics_table(result.metrics, args.metric))
    for w in result.warnings:
        _print(f"warning: {w}")
    return EXIT_OK


def cmd_experiment(args: argparse.Namespace) -> int:
    from simulsi.analysis.comparison import compare
    from simulsi.config.schema import load_experiment
    from simulsi.cost.model import CostModel

    exp, cfg = load_experiment(args.config, allow_outside=args.allow_outside)
    if args.replications:
        exp.replications = args.replications
    if args.workers:
        exp.workers = args.workers
    if args.seed is not None:
        exp.seed = args.seed
    total = len(exp.scenarios) * exp.replications
    _print(
        f"experiment {exp.name!r}: model={exp.model.name} scenarios={len(exp.scenarios)} "
        f"replications={exp.replications} workers={exp.workers}"
    )

    def progress(done: int, n: int) -> None:
        if not args.quiet and (done == n or done % max(1, n // 20) == 0):
            sys.stderr.write(f"\r  {done}/{n} runs")
            if done == n:
                sys.stderr.write("\n")

    result = exp.run(checkpoint=args.checkpoint, progress=progress)
    if cfg.cost:
        result.derive(CostModel.from_dict(cfg.cost).metrics)
    metrics = cfg.metrics
    if cfg.cost and metrics is not None:
        metrics = [*metrics, "cost.total", "cost.profit"]
    _print(result.format_summary(metrics, confidence=cfg.experiment.confidence))
    if len(result.scenarios) > 1 and cfg.baseline in result.scenarios:
        _print("")
        adjust = args.adjust or cfg.experiment.multiple_comparisons
        note = "" if adjust == "none" else f", {adjust}-adjusted"
        _print(
            f"Comparison with {cfg.baseline!r} ({cfg.experiment.confidence:.0%} CI{note}; * = significant):"
        )
        _print(
            compare(
                result,
                cfg.baseline,
                metrics=metrics,
                confidence=cfg.experiment.confidence,
                adjust=adjust,
            ).format()
        )
    if result.errors:
        _print(
            f"\n{len(result.errors)} of {total} runs failed; first error: {result.errors[0].error}"
        )
    out = args.output or cfg.output.directory
    if out:
        if args.output:
            directory = Path(args.output)
        else:
            from simulsi.serialization.io import safe_child

            base = Path(args.config).resolve().parent
            try:
                directory = safe_child(base, out) if not args.allow_outside else base / out
            except ValueError as exc:
                raise ConfigError(
                    f"output.directory {out!r} is outside the configuration directory; "
                    "use -o to choose a location explicitly"
                ) from exc
        result.save(directory, formats=cfg.output.formats)
        _print(f"\nsaved results to {directory}/")
    return EXIT_OK if not result.errors else EXIT_ERROR


def _load_result(path: str) -> Any:
    from simulsi.experiments.experiment import ExperimentResult

    return ExperimentResult.load(path)


def cmd_analyze(args: argparse.Namespace) -> int:
    from simulsi.analysis.comparison import compare
    from simulsi.analysis.report import format_table

    result = _load_result(args.results)
    md = result.metadata
    metrics = args.metric or None
    if args.json:
        payload: dict[str, Any] = {
            "metadata": md.to_dict(),
            "summary": result.summary(metrics, confidence=args.confidence),
        }
        if args.baseline in result.scenarios and len(result.scenarios) > 1:
            payload["comparison"] = compare(
                result,
                args.baseline,
                metrics=metrics,
                confidence=args.confidence,
                adjust=args.adjust or "none",
            ).to_dicts()
        from simulsi.serialization.io import to_jsonable

        _print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))
        return EXIT_OK
    _print(
        f"{md.experiment_id}  model={md.model_name} v{md.model_version}  seed={md.seed}  "
        f"replications={md.replications}  created={md.timestamp}"
    )
    if md.git and md.git.get("commit"):
        _print(f"git {md.git['commit'][:12]}{' (dirty)' if md.git.get('dirty') else ''}")
    _print("")
    _print(result.format_summary(metrics, confidence=args.confidence))
    if args.baseline in result.scenarios and len(result.scenarios) > 1:
        _print("")
        _print(
            compare(
                result,
                args.baseline,
                metrics=metrics,
                confidence=args.confidence,
                adjust=args.adjust or "none",
            ).format()
        )
    if args.precision:
        rows = []
        for sc in result.scenarios:
            for m in metrics or result.metric_names:
                adv = result.replication_advice(
                    m, sc, relative_precision=args.precision, confidence=args.confidence
                )
                rows.append(
                    {
                        "scenario": sc,
                        "metric": m,
                        "n": adv.n,
                        "rel_half_width": adv.relative_half_width,
                        "required_n": adv.required_n if adv.required_n is not None else math.nan,
                        "enough": "yes" if adv.sufficient else "no",
                        "note": adv.note,
                    }
                )
        _print("")
        _print(f"Replication analysis (target relative half-width {args.precision:g}):")
        _print(format_table(rows))
    return EXIT_OK


def cmd_benchmark(args: argparse.Namespace) -> int:
    from simulsi.analysis.report import format_table
    from simulsi.benchmarks import default_suite
    from simulsi.experiments.provenance import environment

    sizes = tuple(args.sizes) if args.sizes else (10_000, 100_000, 1_000_000)
    results = default_suite(
        sizes,
        memory=not args.no_memory,
        workers=tuple(args.workers),
        replications=args.replications,
    )
    rows = [r.to_dict() for r in results]
    if args.json:
        from simulsi.serialization.io import to_jsonable

        _print(json.dumps(to_jsonable({"environment": environment(), "results": rows}), indent=2))
        return EXIT_OK
    _print(
        format_table(
            rows, ["name", "size", "events", "seconds", "events_per_second", "peak_memory_mb"]
        )
    )
    env = environment()
    _print(f"\npython {env['python']} on {env['platform']} ({env['cpu_count']} CPUs)")
    return EXIT_OK


def cmd_visualize(args: argparse.Namespace) -> int:
    from simulsi.visualization import plots

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ext = "html" if args.backend == "plotly" else args.format
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        p = out / f"{name}.{ext}"
        plots.save_figure(fig, str(p))
        written.append(p)

    if args.target.endswith(".py") or args.target.startswith("builtin:"):
        from simulsi.config.schema import resolve_model

        m = resolve_model(args.target)
        res = m.simulate(_parse_params(args.param), seed=args.seed, trace=True, record_series=True)
        save(plots.plot_queue_length(res, backend=args.backend), "queue_length")
        save(plots.plot_utilization(res, backend=args.backend), "utilization")
        if res.log is not None and len(res.log):
            save(plots.plot_timeline(res.log, backend=args.backend), "timeline")
            save(
                plots.plot_entity_trajectories(
                    res.log, backend=args.backend, end_time=res.end_time
                ),
                "trajectories",
            )
    else:
        from simulsi.analysis.comparison import compare

        result = _load_result(args.target)
        metrics = args.metric or result.metric_names[:3]
        for m in metrics:
            safe = m.replace("/", "_")
            for sc in result.scenarios:
                save(
                    plots.plot_distribution(
                        result.values(m, sc), title=f"{m} - {sc}", label=m, backend=args.backend
                    ),
                    f"dist_{safe}_{sc}",
                )
            save(
                plots.plot_convergence(
                    result.values(m, result.scenarios[0]),
                    title=f"Convergence: {m}",
                    backend=args.backend,
                ),
                f"convergence_{safe}",
            )
        if args.baseline in result.scenarios and len(result.scenarios) > 1:
            cmp = compare(result, args.baseline, metrics=metrics)
            save(plots.plot_comparison(cmp, relative=True, backend=args.backend), "comparison")
    for p in written:
        _print(f"wrote {p}")
    return EXIT_OK


def cmd_ui(args: argparse.Namespace) -> int:
    from simulsi.config.schema import resolve_model
    from simulsi.web.server import serve

    models = {ref: resolve_model(ref) for ref in args.model or []}
    return serve(
        host=args.host,
        port=args.port,
        results=[Path(p) for p in args.results],
        models=models,
        open_browser=not args.no_browser,
    )


def cmd_warmup(args: argparse.Namespace) -> int:
    from simulsi.analysis.warmup import suggest_warmup
    from simulsi.config.schema import resolve_model

    m = resolve_model(args.target)
    adv = suggest_warmup(
        m,
        _parse_params(args.param),
        series=args.series,
        replications=args.replications,
        seed=args.seed,
        duration=args.duration,
        bins=args.bins,
    )
    if args.json:
        from simulsi.serialization.io import to_jsonable

        _print(json.dumps(to_jsonable(adv.to_dict()), indent=2, allow_nan=False))
        return EXIT_OK
    _print(f"series {adv.series!r}: {adv.replications} replications x {adv.duration:g} time units")
    _print(f"suggested warm-up (MSER-5): {adv.warmup:g}  (model currently uses {m.warmup:g})")
    if not adv.reliable:
        _print(f"warning: {adv.note}")
    return EXIT_OK


# -- parser ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simulsi", description="SimulSI - model the system, simulate the future."
    )
    p.add_argument("--version", action="version", version=f"simulsi {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create a starter project (model.py + experiment.yaml)")
    s.add_argument("directory", nargs="?", default=".")
    s.add_argument("--force", action="store_true", help="overwrite existing files")
    s.set_defaults(func=cmd_init)

    model_help = (
        "model reference: file.py[:attr], package.module:attr, builtin:mmc, or a config .yaml"
    )
    allow = {
        "action": "store_true",
        "help": "allow config files to reference model files outside their directory",
    }

    s = sub.add_parser("validate", help="validate a configuration or model (schema + smoke run)")
    s.add_argument("target", help=model_help)
    s.add_argument("--param", "-p", action="append", metavar="KEY=VALUE")
    s.add_argument(
        "--smoke", type=float, default=None, help="smoke-run length (default: 10%% of duration)"
    )
    s.add_argument("--allow-outside", **allow)  # type: ignore[arg-type]
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("run", help="run one replication and print its metrics")
    s.add_argument("target", help=model_help)
    s.add_argument("--param", "-p", action="append", metavar="KEY=VALUE")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--duration", type=float, default=None)
    s.add_argument(
        "--warmup",
        type=float,
        default=None,
        help="statistics reset time (default: the model's; scaled if --duration is shorter)",
    )
    s.add_argument("--metric", "-m", action="append", help="only show these metrics")
    s.add_argument("--trace", metavar="FILE", help="write the event log (.json/.csv/.parquet)")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    s.add_argument("--allow-outside", **allow)  # type: ignore[arg-type]
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("experiment", help="run an experiment from a YAML configuration")
    s.add_argument("config")
    s.add_argument("--replications", "-r", type=int)
    s.add_argument("--workers", "-w", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--output", "-o", help="results directory (overrides output.directory)")
    s.add_argument("--checkpoint", help="JSON-lines checkpoint file for resumable runs")
    s.add_argument(
        "--adjust",
        choices=["none", "bonferroni", "holm", "bh"],
        default=None,
        help="multiple-comparison correction for scenario comparisons",
    )
    s.add_argument("--quiet", "-q", action="store_true")
    s.add_argument("--allow-outside", **allow)  # type: ignore[arg-type]
    s.set_defaults(func=cmd_experiment)

    s = sub.add_parser("analyze", help="summarise and compare saved experiment results")
    s.add_argument("results", help="results directory or experiment.json")
    s.add_argument("--metric", "-m", action="append")
    s.add_argument("--baseline", default="baseline")
    s.add_argument("--confidence", type=float, default=0.95)
    s.add_argument(
        "--adjust",
        choices=["none", "bonferroni", "holm", "bh"],
        default=None,
        help="multiple-comparison correction for scenario comparisons",
    )
    s.add_argument(
        "--precision",
        type=float,
        help="relative precision target for replication advice, e.g. 0.05",
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("warmup", help="suggest a warm-up period (MSER-5 over replications)")
    s.add_argument("target", help="model reference")
    s.add_argument("--param", "-p", action="append", metavar="KEY=VALUE")
    s.add_argument("--series", help="time series to analyse (default: first resource queue_length)")
    s.add_argument("--replications", "-r", type=int, default=5)
    s.add_argument("--duration", type=float)
    s.add_argument("--bins", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_warmup)

    s = sub.add_parser("benchmark", help="measure engine and experiment throughput on this machine")
    s.add_argument("--sizes", type=int, nargs="+", help="event counts (default 10k 100k 1M)")
    s.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    s.add_argument("--replications", type=int, default=16)
    s.add_argument("--no-memory", action="store_true", help="skip tracemalloc runs")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("visualize", help="plot saved results, or run a model once and plot it")
    s.add_argument("target", help="results directory/experiment.json, or a model reference")
    s.add_argument("--out", "-o", default="plots")
    s.add_argument("--metric", "-m", action="append")
    s.add_argument("--baseline", default="baseline")
    s.add_argument("--backend", choices=["matplotlib", "plotly"], default="matplotlib")
    s.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    s.add_argument("--param", "-p", action="append", metavar="KEY=VALUE")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_visualize)

    s = sub.add_parser("ui", help="start the local web dashboard (127.0.0.1 only by default)")
    s.add_argument("results", nargs="*", help="saved experiment results to preload")
    s.add_argument(
        "--model", action="append", metavar="REF", help="extra model the dashboard may run"
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8642)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_ui)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code: int = args.func(args)
    except (ConfigError, SimulsiError, FileNotFoundError, KeyError) as exc:
        _err(str(exc) if not isinstance(exc, KeyError) else f"not found: {exc}")
        return EXIT_INVALID
    except ImportError as exc:
        _err(str(exc))
        return EXIT_ERROR
    except KeyboardInterrupt:
        _err("interrupted")
        return 130
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
