from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simulsi import Simulation
from simulsi.cli import main
from simulsi.processes import TraceExhausted, TraceReplay, load_trace, trace_arrivals
from simulsi.randomness import fit_distribution, from_spec, load_values

RNG = np.random.default_rng(2026)

# -- fitting -------------------------------------------------------------------------


def test_fit_recovers_gamma_and_lognormal() -> None:
    g = fit_distribution(RNG.gamma(2.0, 3.0, size=2_000))
    assert g.best.name == "gamma"
    assert g.best.params["shape"] == pytest.approx(2.0, rel=0.1)
    assert g.best.params["scale"] == pytest.approx(3.0, rel=0.1)
    assert g.best.ks_pvalue > 0.01
    assert from_spec(g.best.distribution.to_spec()) == g.best.distribution

    ln = fit_distribution(RNG.lognormal(1.0, 0.5, size=2_000))
    assert ln.best.name == "lognormal"
    assert ln.best.params["mu"] == pytest.approx(1.0, abs=0.05)
    assert ln.get("normal").aic > ln.best.aic
    with pytest.raises(KeyError):
        ln.get("weibull")


def test_fit_prefers_simpler_family_by_bic() -> None:
    rep = fit_distribution(RNG.exponential(4.0, size=3_000), criterion="bic")
    assert rep.best.name in {"exponential", "gamma"}
    assert rep.get("exponential").bic <= rep.get("gamma").bic + 10
    assert rep.get("exponential").params["mean"] == pytest.approx(4.0, rel=0.08)
    ks = fit_distribution(RNG.exponential(4.0, size=500), criterion="ks")
    assert ks.results == sorted(ks.results, key=lambda r: r.ks_statistic)


def test_fit_skips_families_with_wrong_support() -> None:
    rep = fit_distribution(RNG.normal(0.0, 2.0, size=1_000))
    assert rep.best.name == "normal"
    assert set(rep.skipped) == {"exponential", "gamma", "lognormal", "poisson"}
    counts = fit_distribution(RNG.poisson(3.5, size=1_000).astype(float))
    assert counts.best.name == "poisson"
    assert math.isnan(counts.best.ks_pvalue)
    assert counts.best.params["lam"] == pytest.approx(3.5, rel=0.08)
    tri = fit_distribution(RNG.triangular(1, 2, 6, size=3_000), ["triangular", "uniform"])
    assert tri.best.name == "triangular"
    assert tri.best.params["mode"] == pytest.approx(2.0, abs=0.4)


def test_fit_report_output_and_errors() -> None:
    rep = fit_distribution([1.2, 3.4, 2.2, 5.1, 0.7, 2.9])
    text = rep.format()
    assert "ranked by aic" in text and "only 6 observations" in text
    json.dumps(rep.to_dict())
    flat = fit_distribution([2.0, 2.0, 2.0])
    assert not flat.results and "Constant" in flat.skipped["normal"]
    with pytest.raises(ValueError, match="no candidate"):
        _ = flat.best
    for bad in ([1.0], [1.0, math.nan]):
        with pytest.raises(ValueError):
            fit_distribution(bad)
    with pytest.raises(ValueError, match="unknown candidate"):
        fit_distribution([1.0, 2.0], ["weibull"])
    with pytest.raises(ValueError, match="criterion"):
        fit_distribution([1.0, 2.0], criterion="r2")  # type: ignore[arg-type]


def test_fit_report_plot() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    rep = fit_distribution(RNG.gamma(2.0, 1.0, size=300))
    assert rep.plot(top=2).get_legend() is not None
    counts = fit_distribution(RNG.poisson(2.0, size=300).astype(float), ["poisson"])
    counts.plot()


def test_load_values(tmp_path: Path) -> None:
    csv_file = tmp_path / "t.csv"
    csv_file.write_text("id,service,label\n1,3.5,a\n2,,b\n3,4.25,c\n")
    assert load_values(csv_file) == [1.0, 2.0, 3.0]
    assert load_values(csv_file, "service") == [3.5, 4.25]
    with pytest.raises(ValueError, match="no column"):
        load_values(csv_file, "missing")
    with pytest.raises(ValueError, match="not a number"):
        load_values(csv_file, "label")
    txt = tmp_path / "t.txt"
    txt.write_text("1.5 2.5\n3.5, 4\n")
    assert load_values(txt) == [1.5, 2.5, 3.5, 4.0]
    with pytest.raises(ValueError):
        load_values(txt, "x")
    (tmp_path / "e.csv").write_text("a\n")
    with pytest.raises(ValueError, match="no data"):
        load_values(tmp_path / "e.csv")
    (tmp_path / "s.csv").write_text("a\nx\n")
    with pytest.raises(ValueError, match="no numeric column"):
        load_values(tmp_path / "s.csv")


def test_fit_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "service.csv"
    f.write_text("service\n" + "\n".join(f"{v:.5f}" for v in RNG.gamma(3, 2, size=400)))
    assert main(["fit", str(f)]) == 0
    out = capsys.readouterr().out
    assert "best fit as a config spec" in out and "distribution: gamma" in out
    assert main(["fit", str(f), "--json", "--candidates", "normal", "exponential"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in data["results"]} == {"normal", "exponential"}
    one = tmp_path / "one.txt"
    one.write_text("5")
    assert main(["fit", str(one)]) != 0


# -- traces ----------------------------------------------------------------------------


def test_trace_arrivals_and_replay_reproduce_lindley_waits(tmp_path: Path) -> None:
    f = tmp_path / "history.csv"
    rows = [(0.0, 4.0), (1.0, 2.0), (2.5, 1.0), (9.0, 3.0), (9.5, 0.5)]
    f.write_text("time,service,kind\n" + "\n".join(f"{t},{s},x" for t, s in rows))
    trace = load_trace(f)
    assert trace[0] == {"time": 0.0, "service": 4.0, "kind": "x"}

    sim = Simulation(seed=1)
    server = sim.resource("server", 1)
    waits: list[float] = []

    def job(sim: Simulation, record: dict[str, Any]) -> Any:
        t0 = sim.now
        req = yield sim.request(server)
        waits.append(sim.now - t0)
        yield record["service"]
        server.release(req)

    trace_arrivals(sim, trace, job)
    sim.run()
    # Lindley recursion: W[n+1] = max(0, W[n] + S[n] - A[n+1])
    expected = [0.0]
    for (t0, s0), (t1, _) in pairwise(rows):
        expected.append(max(0.0, expected[-1] + s0 - (t1 - t0)))
    assert waits == pytest.approx(expected)


def test_trace_arrivals_with_plain_times_and_validation() -> None:
    sim = Simulation(seed=1)
    seen: list[tuple[float, float]] = []
    sim.run(until=2)
    trace_arrivals(sim, [0.0, 0.0, 1.5], lambda sim, t: seen.append((sim.now, t)))
    sim.run()
    assert seen == [(2.0, 0.0), (2.0, 0.0), (3.5, 1.5)]
    with pytest.raises(ValueError, match="sorted"):
        trace_arrivals(sim, [1.0, 0.5], lambda sim, t: None)
    with pytest.raises(ValueError, match="sorted"):
        trace_arrivals(sim, [{"at": -1.0}], lambda sim, t: None, time="at")


def test_trace_replay() -> None:
    r = TraceReplay([3, 1, 2])
    assert [r.sample(), r.sample(None), r.sample()] == [3, 1, 2]
    assert r.remaining == 0 and len(r) == 3
    with pytest.raises(TraceExhausted):
        r.sample()
    c = TraceReplay([1.0, 2.0], cycle=True)
    assert [c.sample() for _ in range(5)] == [1.0, 2.0, 1.0, 2.0, 1.0]
    assert c.remaining == math.inf
    with pytest.raises(ValueError):
        TraceReplay([])
