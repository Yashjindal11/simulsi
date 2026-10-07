from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from simulsi import Experiment, Scenario
from simulsi.models import mmc
from simulsi.web.server import DashboardState, create_server

pytestmark = pytest.mark.integration

SHORT = mmc.with_options(duration=300, warmup=30)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, DashboardState]]:
    d = tmp_path_factory.mktemp("res")
    Experiment(
        SHORT, [Scenario("baseline"), Scenario("two", {"servers": 2})], replications=4, seed=1
    ).run().save(d)
    state = DashboardState({"custom:short": SHORT})
    state.load_path(d)
    httpd = create_server(state, "127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()
    httpd.server_close()


def get(url: str, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            ctype = r.headers.get("Content-Type", "")
            return r.status, json.loads(body) if "json" in ctype else body
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(url: str, body: Any, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_info_models_results(server: tuple[str, DashboardState]) -> None:
    base, _ = server
    code, info = get(f"{base}/api/info")
    assert code == 200 and "builtin:mmc" in info["models"] and "custom:short" in info["models"]
    code, models = get(f"{base}/api/models")
    assert any(m["id"] == "builtin:mmc" and m["parameters"] for m in models)
    code, results = get(f"{base}/api/results")
    assert (
        code == 200 and results[0]["id"] == "r1" and results[0]["scenarios"] == ["baseline", "two"]
    )


def test_result_detail_metric_compare_export(server: tuple[str, DashboardState]) -> None:
    base, _ = server
    code, detail = get(f"{base}/api/results/r1")
    assert code == 200 and detail["metadata"]["seed"] == 1 and detail["summary"]
    assert detail["parameters"]["two"]["servers"] == 2
    code, metric = get(
        f"{base}/api/results/r1/metric?metric=resource.server.wait.mean&scenario=two"
    )
    assert code == 200 and len(metric["values"]) == 4 and metric["convergence"]["n"] == [1, 2, 3, 4]
    code, cmp = get(
        f"{base}/api/results/r1/compare?baseline=baseline&metrics=resource.server.wait.mean"
    )
    assert code == 200 and cmp[0]["method"] == "paired-t" and cmp[0]["scenario"] == "two"
    code, csv = get(f"{base}/api/results/r1/export.csv")
    assert code == 200 and csv.decode().startswith("experiment_id,")
    code, js = get(f"{base}/api/results/r1/export.json")
    assert code == 200 and js["metadata"]["seed"] == 1
    assert get(f"{base}/api/results/zzz")[0] == 404


def test_run_trace_and_upload(server: tuple[str, DashboardState], tmp_path: Path) -> None:
    base, _ = server
    code, out = post(
        f"{base}/api/run",
        {
            "model": "custom:short",
            "replications": 2,
            "seed": 3,
            "scenarios": [{"name": "baseline", "parameters": {"servers": 2}}],
        },
    )
    assert code == 202 and out["status"] in ("running", "done") and out["total"] == 2
    import time

    for _ in range(200):
        code, job = get(f"{base}/api/jobs/{out['id']}")
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["done"] == 2, job
    code, detail = get(f"{base}/api/results/{job['result_id']}")
    assert detail["metadata"]["replications"] == 2
    assert get(f"{base}/api/jobs/j999")[0] == 404
    code, trace = post(
        f"{base}/api/trace",
        {"model": "builtin:mmc", "parameters": {"servers": 2}, "seed": 1, "duration": 100},
    )
    assert code == 200 and trace["end_time"] == 100 and trace["log"]
    assert "resource.server.queue_length" in trace["series"]
    res = Experiment(SHORT, replications=2).run()
    code, _up = post(f"{base}/api/results", res.to_dict())
    assert code == 201


def test_rejects_bad_requests(server: tuple[str, DashboardState]) -> None:
    base, _ = server
    assert post(f"{base}/api/run", {"model": "os.system"})[0] == 400
    assert post(f"{base}/api/run", {"model": "builtin:mmc", "replications": 10**6})[0] == 400
    assert (
        post(
            f"{base}/api/run",
            {"model": "builtin:mmc", "scenarios": [{"name": "x", "parameters": {"bogus": 1}}]},
        )[0]
        == 400
    )
    assert post(f"{base}/api/results", {"not": "an experiment"})[0] == 400
    # cross-origin and DNS-rebinding protection
    assert (
        post(f"{base}/api/run", {"model": "builtin:mmc"}, {"Origin": "http://evil.example"})[0]
        == 403
    )
    assert get(f"{base}/api/info", {"Host": "evil.example"})[0] == 403
    req = urllib.request.Request(
        f"{base}/api/run",
        data=b"x=1",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 415


def test_static_fallback_and_traversal(server: tuple[str, DashboardState]) -> None:
    base, _ = server
    code, body = get(f"{base}/../../etc/passwd")
    assert code == 200 and b"root:" not in body
    code, body = get(f"{base}/")
    assert code == 200 and (b"<html" in body.lower() or b"not built" in body)


def _wait(base: str, jid: str) -> dict[str, Any]:
    import time

    job: dict[str, Any] = {}
    for _ in range(600):
        _, job = get(f"{base}/api/jobs/{jid}")
        if job["status"] != "running":
            break
        time.sleep(0.05)
    return job


def test_grid_sensitivity_montecarlo_and_report(server: tuple[str, DashboardState]) -> None:
    base, _ = server
    code, out = post(
        f"{base}/api/grid",
        {
            "model": "custom:short",
            "factors": {"servers": [1, 2], "arrival_rate": [0.5, 0.8]},
            "replications": 2,
            "seed": 1,
        },
    )
    assert code == 202 and out["total"] == 8
    job = _wait(base, out["id"])
    assert job["status"] == "done", job
    _, detail = get(f"{base}/api/results/{job['result_id']}")
    assert len(detail["scenarios"]) == 4
    assert detail["parameters"]["servers=2,arrival_rate=0.8"]["servers"] == 2

    code, out = post(
        f"{base}/api/sensitivity",
        {
            "model": "custom:short",
            "ranges": {"arrival_rate": [0.4, 0.8], "servers": [1, 3]},
            "r": 3,
            "outputs": ["resource.server.utilization"],
        },
    )
    assert code == 202
    job = _wait(base, out["id"])
    assert job["status"] == "done", job
    rows = job["output"]["rows"]
    assert {r["method"] for r in rows} == {"morris-mu_star", "morris-mu", "morris-sigma"}

    code, out = post(
        f"{base}/api/montecarlo",
        {
            "model": "custom:short",
            "iterations": 12,
            "outputs": ["resource.server.utilization"],
            "inputs": {"arrival_rate": {"distribution": "uniform", "low": 0.4, "high": 0.8}},
        },
    )
    assert code == 202
    job = _wait(base, out["id"])
    assert job["status"] == "done", job
    util = job["output"]["outputs"]["resource.server.utilization"]
    assert len(util["values"]) == 12 and 0 < util["quantiles"]["p50"] < 1
    assert job["output"]["sensitivity"]

    for bad in (
        ("/api/grid", {"model": "custom:short", "factors": {}}),
        ("/api/sensitivity", {"model": "custom:short", "ranges": {}}),
        ("/api/montecarlo", {"model": "custom:short", "inputs": {}}),
        (
            "/api/montecarlo",
            {"model": "custom:short", "iterations": 10**6, "inputs": {"arrival_rate": 0.5}},
        ),
    ):
        assert post(f"{base}{bad[0]}", bad[1])[0] == 400, bad

    req = urllib.request.Request(f"{base}/api/results/r1/report.html")
    with urllib.request.urlopen(req, timeout=30) as r:
        page = r.read().decode()
        assert r.headers["Content-Type"].startswith("text/html")
        assert "attachment" in r.headers["Content-Disposition"]
    assert "<svg" in page and "Differences from baseline" in page
