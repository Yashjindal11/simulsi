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
    assert code == 201
    code, detail = get(f"{base}/api/results/{out['id']}")
    assert detail["metadata"]["replications"] == 2
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
