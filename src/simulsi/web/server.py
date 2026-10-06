"""Local web dashboard server (standard library only).

Serves the compiled React app from ``simulsi/web/static`` and a small JSON
API. Security model:

* binds to ``127.0.0.1`` by default; there is no authentication, so do not
  expose it on a network;
* rejects requests whose ``Host`` header is not the bound address (DNS
  rebinding) and cross-origin POSTs; POST bodies must be JSON and at most
  20 MB;
* runs only built-in models and models given explicitly with ``--model``;
  never executes code from requests or files;
* serves static files only from the package's static directory and result
  files only from the directories it was started with.
"""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import webbrowser
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from simulsi._version import __version__
from simulsi.analysis.comparison import compare
from simulsi.core.model import Model
from simulsi.errors import ConfigError, SimulsiError
from simulsi.experiments.experiment import Experiment, ExperimentResult
from simulsi.scenarios.scenario import Scenario
from simulsi.serialization.io import csv_text, to_jsonable
from simulsi.statistics.core import convergence, summarize

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 20_000_000
MAX_RUNS = 5_000
MAX_TRACE_RECORDS = 20_000
MAX_SERIES_POINTS = 4_000


class DashboardState:
    """Experiments and models the dashboard knows about (in memory)."""

    def __init__(self, models: Mapping[str, Model] | None = None) -> None:
        from simulsi.models import BUILTIN_MODELS

        self.models: dict[str, Model] = {f"builtin:{k}": v for k, v in BUILTIN_MODELS.items()}
        self.models.update(models or {})
        self.results: dict[str, tuple[ExperimentResult, str]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._job_counter = 0

    def start_job(self, exp: Experiment, source: str) -> str:
        """Run ``exp`` in a background thread; poll :meth:`job` for progress."""
        with self._lock:
            self._job_counter += 1
            jid = f"j{self._job_counter}"
            total = len(exp.scenarios) * exp.replications
            self.jobs[jid] = {"id": jid, "status": "running", "done": 0, "total": total,
                              "result_id": None, "error": None}

        def progress(done: int, total: int) -> None:
            self.jobs[jid].update(done=done, total=total)

        def work() -> None:
            try:
                rid = self.add_result(exp.run(progress=progress), source)
                self.jobs[jid].update(status="done", result_id=rid)
            except Exception as exc:  # reported to the client, not raised in the thread
                self.jobs[jid].update(status="error", error=f"{type(exc).__name__}: {exc}")

        threading.Thread(target=work, name=f"simulsi-{jid}", daemon=True).start()
        return jid

    def job(self, jid: str) -> dict[str, Any]:
        if jid not in self.jobs:
            raise KeyError(f"no job {jid!r}")
        return dict(self.jobs[jid])

    def add_result(self, result: ExperimentResult, source: str) -> str:
        with self._lock:
            self._counter += 1
            rid = f"r{self._counter}"
            self.results[rid] = (result, source)
        return rid

    def load_path(self, path: Path) -> str:
        return self.add_result(ExperimentResult.load(path), str(path))

    def get(self, rid: str) -> ExperimentResult:
        if rid not in self.results:
            raise KeyError(f"no result {rid!r}")
        return self.results[rid][0]

    def model(self, ref: str) -> Model:
        if ref not in self.models:
            raise KeyError(f"unknown model {ref!r}; available: {sorted(self.models)}")
        return self.models[ref]


# -- API payloads -------------------------------------------------------------------


def result_index(state: DashboardState) -> list[dict[str, Any]]:
    out = []
    for rid, (res, source) in state.results.items():
        md = res.metadata
        out.append({
            "id": rid, "source": source, "experiment_id": md.experiment_id, "name": md.name,
            "model": md.model_name, "model_version": md.model_version, "timestamp": md.timestamp,
            "scenarios": res.scenarios, "replications": md.replications,
        })
    return out


def result_detail(res: ExperimentResult, confidence: float = 0.95) -> dict[str, Any]:
    return {
        "metadata": res.metadata.to_dict(),
        "scenarios": res.scenarios,
        "metrics": res.metric_names,
        "summary": res.summary(confidence=confidence),
        "errors": [{"scenario": r.scenario, "replication": r.replication, "error": r.error} for r in res.errors],
        "parameters": {sc: next((r.parameters for r in res.records if r.scenario == sc), {}) for sc in res.scenarios},
    }


def metric_detail(res: ExperimentResult, metric: str, scenario: str) -> dict[str, Any]:
    values = res.values(metric, scenario)
    s = summarize(values)
    c = convergence(values)
    return {
        "metric": metric, "scenario": scenario, "values": values.tolist(),
        "summary": s.to_dict(), "convergence": c.to_dict(),
    }


def _downsample(points: list[tuple[float, float]], limit: int = MAX_SERIES_POINTS) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    step = math.ceil(len(points) / limit)
    return points[::step] + [points[-1]]


def run_trace(model: Model, params: Mapping[str, Any], seed: int, duration: float | None) -> dict[str, Any]:
    horizon = duration if duration is not None else model.duration
    res = model.simulate(params, seed=seed, duration=horizon, trace=True, record_series=True,
                         max_log_records=MAX_TRACE_RECORDS)
    log = res.log
    return {
        "seed": res.seed, "end_time": res.end_time, "events": res.events_processed,
        "metrics": res.metrics, "warnings": res.warnings,
        "series": {k: _downsample(v) for k, v in res.series.items()},
        "log": [] if log is None else [r.to_dict() for r in log.records],
        "log_dropped": 0 if log is None else log.dropped,
    }


def build_experiment(state: DashboardState, body: Mapping[str, Any]) -> Experiment:
    """Validate a run request and turn it into an experiment (raises on bad input)."""
    model = state.model(str(body.get("model", "")))
    raw = body.get("scenarios") or [{"name": "baseline", "parameters": {}}]
    if not isinstance(raw, list) or len(raw) > 50:
        raise ConfigError("scenarios must be a list of at most 50 entries")
    scenarios = [Scenario(str(s.get("name", "")), dict(s.get("parameters") or {})) for s in raw]
    reps = int(body.get("replications", 10))
    if reps < 1 or reps * len(scenarios) > MAX_RUNS:
        raise ConfigError(f"replications x scenarios must be between 1 and {MAX_RUNS}")
    duration = body.get("duration")
    if duration is not None:
        if not 0 < float(duration) <= 1e7:
            raise ConfigError("duration must be in (0, 1e7]")
        model = model.with_options(duration=float(duration), warmup=min(model.warmup, float(duration) / 2))
    exp = Experiment(model, scenarios, replications=reps, seed=int(body.get("seed", 0)),
                     workers=1, on_error="record")
    issues = exp.validate()
    if issues:
        raise ConfigError("; ".join(issues))
    return exp


# -- HTTP handler ------------------------------------------------------------------------


def make_handler(state: DashboardState, allowed_hosts: set[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"simulsi/{__version__}"

        def log_message(self, format: str, *args: Any) -> None:  # quiet by default
            return

        # -- helpers --
        def _send(self, status: int, body: bytes, ctype: str, extra: Mapping[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'",
            )
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, status: int = 200) -> None:
            body = json.dumps(to_jsonable(data), allow_nan=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"})

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return urlparse(origin).netloc in allowed_hosts

        # -- routing --
        def do_GET(self) -> None:
            if not self._host_ok():
                self._error(HTTPStatus.FORBIDDEN, "invalid Host header")
                return
            url = urlparse(self.path)
            if url.path.startswith("/api/"):
                self._api_get(url.path, parse_qs(url.query))
            else:
                self._static(url.path)

        def do_POST(self) -> None:
            if not self._host_ok() or not self._origin_ok():
                self._error(HTTPStatus.FORBIDDEN, "cross-origin requests are not allowed")
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body must be 1..{MAX_BODY} bytes")
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                self._api_post(urlparse(self.path).path, body)
            except (ValueError, ConfigError, SimulsiError, KeyError, TypeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

        def _api_get(self, path: str, query: dict[str, list[str]]) -> None:
            q = {k: v[0] for k, v in query.items()}
            parts = [p for p in path.split("/") if p][1:]  # drop "api"
            try:
                if parts == ["info"]:
                    self._json({"version": __version__, "models": sorted(state.models)})
                elif parts == ["models"]:
                    self._json([{"id": k, **m.describe()} for k, m in sorted(state.models.items())])
                elif parts == ["results"]:
                    self._json(result_index(state))
                elif len(parts) == 2 and parts[0] == "jobs":
                    self._json(state.job(parts[1]))
                elif len(parts) == 2 and parts[0] == "results":
                    self._json(result_detail(state.get(parts[1]), float(q.get("confidence", 0.95))))
                elif len(parts) == 3 and parts[0] == "results" and parts[2] == "metric":
                    res = state.get(parts[1])
                    self._json(metric_detail(res, q["metric"], q.get("scenario", res.scenarios[0])))
                elif len(parts) == 3 and parts[0] == "results" and parts[2] == "compare":
                    res = state.get(parts[1])
                    metrics = [m for m in q.get("metrics", "").split(",") if m] or None
                    cmp = compare(res, q.get("baseline", res.scenarios[0]), metrics=metrics,
                                  confidence=float(q.get("confidence", 0.95)))
                    self._json(cmp.to_dicts())
                elif len(parts) == 3 and parts[0] == "results" and parts[2] == "export.json":
                    body = json.dumps(to_jsonable(state.get(parts[1]).to_dict()), indent=2, allow_nan=False)
                    self._send(200, body.encode(), "application/json",
                               {"Content-Disposition": f'attachment; filename="{parts[1]}-experiment.json"'})
                elif len(parts) == 3 and parts[0] == "results" and parts[2] == "export.csv":
                    data = csv_text(state.get(parts[1]).rows()).encode("utf-8")
                    self._send(200, data, "text/csv; charset=utf-8",
                               {"Content-Disposition": f'attachment; filename="{parts[1]}-replications.csv"'})
                else:
                    self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc).strip("'\""))
            except (ValueError, ConfigError, SimulsiError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

        def _api_post(self, path: str, body: dict[str, Any]) -> None:
            if path == "/api/run":
                exp = build_experiment(state, body)
                jid = state.start_job(exp, f"dashboard run ({exp.model.name})")
                self._json(state.job(jid), HTTPStatus.ACCEPTED)
            elif path == "/api/trace":
                model = state.model(str(body.get("model", "")))
                duration = body.get("duration")
                if duration is not None and not (0 < float(duration) <= 1e7):
                    raise ConfigError("duration must be in (0, 1e7]")
                self._json(run_trace(model, dict(body.get("parameters") or {}), int(body.get("seed", 0)),
                                     None if duration is None else float(duration)))
            elif path == "/api/results":
                res = ExperimentResult.from_dict(body)
                self._json({"id": state.add_result(res, "uploaded in browser")}, HTTPStatus.CREATED)
            else:
                self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")

        def _static(self, path: str) -> None:
            if not STATIC_DIR.is_dir():
                msg = (b"<h1>SimulSI dashboard not built</h1><p>Run <code>cd web/frontend && npm ci && "
                       b"npm run build</code>, or install a wheel that includes it. The JSON API is "
                       b"available under /api/.</p>")
                self._send(HTTPStatus.OK, msg, "text/html; charset=utf-8")
                return
            rel = path.lstrip("/") or "index.html"
            root = STATIC_DIR.resolve()
            target = (root / rel).resolve()
            inside = target == root or root in target.parents
            if not inside or not target.is_file():
                target = root / "index.html"  # SPA fallback (also blocks traversal)
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            cache = "no-cache" if target.name == "index.html" else "public, max-age=31536000, immutable"
            self._send(HTTPStatus.OK, target.read_bytes(), ctype, {"Cache-Control": cache})

    return Handler


def create_server(
    state: DashboardState, host: str = "127.0.0.1", port: int = 8642
) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(state, set()))
    real_port = httpd.server_address[1]
    hosts = {f"{host}:{real_port}", f"localhost:{real_port}", f"127.0.0.1:{real_port}"}
    httpd.RequestHandlerClass = make_handler(state, hosts)
    return httpd


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8642,
    results: list[Path] | None = None,
    models: Mapping[str, Model] | None = None,
    open_browser: bool = True,
) -> int:
    state = DashboardState(models)
    for p in results or []:
        state.load_path(p)
    httpd = create_server(state, host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"SimulSI dashboard on {url}  (Ctrl+C to stop)")
    if host not in ("127.0.0.1", "localhost"):
        print("warning: the dashboard has no authentication; do not expose it to untrusted networks")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
