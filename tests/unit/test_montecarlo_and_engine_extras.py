from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as st

from simulsi import Event, EventStatus, Simulation, monte_carlo
from simulsi.events import EventQueue
from simulsi.experiments import sample_inputs
from simulsi.randomness import (
    Categorical,
    Constant,
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    RandomStream,
    Triangular,
    Uniform,
)

LHS_CASES = [
    (Uniform(2, 5), st.uniform(2, 3)),
    (Normal(10, 2), st.norm(10, 2)),
    (Exponential(rate=0.5), st.expon(scale=2)),
    (Triangular(1, 2, 6), st.triang(0.2, loc=1, scale=5)),
    (LogNormal(0.3, 0.5), st.lognorm(0.5, scale=math.exp(0.3))),
    (Gamma(2.0, 1.5), st.gamma(2.0, scale=1.5)),
]


@pytest.mark.parametrize(("dist", "ref"), LHS_CASES, ids=lambda x: type(x).__name__)
def test_latin_hypercube_has_one_sample_per_stratum(dist: object, ref: object) -> None:
    n = 200
    xs = sample_inputs({"x": dist}, n, seed=1, sampling="lhs")["x"]  # type: ignore[dict-item]
    u = np.sort(ref.cdf(xs))  # type: ignore[attr-defined]
    assert np.all(np.floor(u * n).astype(int) == np.arange(n))


def test_lhs_constant_and_unsupported() -> None:
    assert np.all(sample_inputs({"c": Constant(3.0)}, 5, 0, "lhs")["c"] == 3.0)
    with pytest.raises(ValueError, match="latin hypercube"):
        sample_inputs({"c": Categorical(["a", "b"])}, 5, 0, "lhs")


def test_monte_carlo_result_helpers() -> None:
    mc = monte_carlo(lambda a: {"x": a, "y": 2 * a}, {"a": Uniform(0, 1)}, 50, seed=2)
    rows = mc.to_rows()
    assert len(rows) == 50 and set(rows[0]) == {"iteration", "input.a", "x", "y"}
    d = mc.to_dict()
    assert d["iterations"] == 50 and set(d["summary"]) == {"x", "y"}
    with pytest.raises(KeyError, match="several outputs"):
        mc.summary()
    with pytest.raises(KeyError):
        mc.summary("nope")
    with pytest.raises(ValueError):
        mc.probability("x", "!=", 0.5)  # type: ignore[arg-type]
    assert mc.quantile([0.1, 0.9], "x")[0] < mc.quantile([0.1, 0.9], "x")[1]
    with pytest.raises(ValueError):
        monte_carlo(lambda a: a, {"a": 1.0}, 0)


def test_monte_carlo_handles_outputs_missing_in_some_iterations() -> None:
    def f(a: float) -> dict[str, float]:
        return {"always": a, **({"sometimes": a} if a > 0.5 else {})}

    mc = monte_carlo(f, {"a": Uniform(0, 1)}, 200, seed=3)
    assert len(mc.outputs["sometimes"]) == 200
    assert np.isnan(mc.outputs["sometimes"]).any() and not np.isnan(mc.outputs["always"]).any()


def test_event_queue_inspection_and_clear() -> None:
    q = EventQueue()
    a, b = Event("a"), Event("b")
    q.push(a, 2.0)
    q.push(b, 1.0)
    assert q.peek() is b and q.peek_time() == 1.0
    assert [e.event_type for e in q] == ["b", "a"]
    q.clear()
    assert len(q) == 0 and not q and a.status is EventStatus.CANCELLED
    assert q.pop() is None and q.peek() is None and math.isinf(q.peek_time())


def test_event_describe_and_repr() -> None:
    sim = Simulation(seed=1)
    ev = sim.schedule(time=3, event_type="x", payload={"k": 1})
    d = ev.describe()
    assert d == {
        "event_id": 0,
        "timestamp": 3.0,
        "priority": 0,
        "event_type": "x",
        "status": "scheduled",
        "payload": {"k": 1},
    }
    assert "x" in repr(ev) and ev.pending
    assert sim.peek() == 3.0
    assert sim.step() is ev and sim.step() is None


def test_random_stream_helpers() -> None:
    r = RandomStream(5)
    assert 0 <= r.uniform(0, 1) <= 1
    assert r.integers(0, 3) in {0, 1, 2}
    assert isinstance(r.bernoulli(0.5), bool)
    assert r.choice(["a", "b"], p=[0, 1]) == "b"
    items = list(range(10))
    r.shuffle(items)
    assert sorted(items) == list(range(10))
    for value in (
        r.normal(),
        r.exponential(2),
        r.poisson(3),
        r.binomial(5, 0.5),
        r.gamma(2),
        r.lognormal(),
        r.triangular(0, 1, 2),
    ):
        assert math.isfinite(value)
    assert "seed=5" in repr(r)
