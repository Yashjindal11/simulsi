"""Monte Carlo simulation: propagate input uncertainty through a model."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from simulsi.core.model import Model
from simulsi.randomness.distributions import Distribution, DistributionLike, as_distribution
from simulsi.randomness.stream import RandomStream, derive_seed
from simulsi.statistics.core import Summary, bootstrap_ci, proportion_ci, summarize

FloatArray = npt.NDArray[np.float64]
Comparator = Literal[">", ">=", "<", "<="]


@dataclass
class MonteCarloResult:
    """Sampled inputs and outputs, one entry per iteration."""

    inputs: dict[str, npt.NDArray[Any]]
    outputs: dict[str, FloatArray]
    iterations: int
    seed: int
    runtime: float
    sampling: str = "random"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_names(self) -> list[str]:
        return list(self.outputs)

    def _out(self, name: str | None) -> FloatArray:
        if name is None:
            if len(self.outputs) != 1:
                raise KeyError(f"several outputs; choose one of {self.output_names}")
            name = next(iter(self.outputs))
        if name not in self.outputs:
            raise KeyError(f"no output {name!r}; have {self.output_names}")
        return self.outputs[name]

    def summary(self, output: str | None = None, confidence: float = 0.95) -> Summary:
        return summarize(self._out(output), confidence)

    def summaries(self, confidence: float = 0.95) -> dict[str, Summary]:
        return {k: summarize(v, confidence) for k, v in self.outputs.items()}

    def quantile(self, q: float | Sequence[float], output: str | None = None) -> Any:
        r = np.quantile(self._out(output)[~np.isnan(self._out(output))], q)
        return float(r) if np.ndim(r) == 0 else r.tolist()

    def percentile(self, p: float, output: str | None = None) -> float:
        return float(self.quantile(p / 100, output))

    def quantile_ci(
        self,
        q: float,
        output: str | None = None,
        *,
        confidence: float = 0.95,
        n_resamples: int = 2000,
    ) -> tuple[float, float]:
        """Bootstrap interval for a quantile (e.g. the 95th percentile of cost)."""

        def stat(a: FloatArray, axis: int) -> Any:
            return np.quantile(a, q, axis=axis)

        return bootstrap_ci(
            self._out(output),
            stat,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=self.seed,
        )

    def probability(
        self, output: str | None, op: Comparator, threshold: float, *, confidence: float = 0.95
    ) -> tuple[float, tuple[float, float]]:
        """``P(output op threshold)`` with a Wilson interval, e.g. ``probability("cost", ">", 1e6)``."""
        x = self._out(output)
        x = x[~np.isnan(x)]
        ops: dict[str, Callable[[FloatArray, float], npt.NDArray[np.bool_]]] = {
            ">": np.greater,
            ">=": np.greater_equal,
            "<": np.less,
            "<=": np.less_equal,
        }
        if op not in ops:
            raise ValueError(f"op must be one of {list(ops)}")
        k = int(ops[op](x, threshold).sum())
        return (k / len(x) if len(x) else math.nan, proportion_ci(k, len(x), confidence))

    def to_rows(self) -> list[dict[str, Any]]:
        rows = []
        for i in range(self.iterations):
            row: dict[str, Any] = {"iteration": i}
            for k, v in self.inputs.items():
                row[f"input.{k}"] = v[i].item() if hasattr(v[i], "item") else v[i]
            for k, out in self.outputs.items():
                row[k] = float(out[i])
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "seed": self.seed,
            "runtime": self.runtime,
            "sampling": self.sampling,
            "summary": {k: s.to_dict() for k, s in self.summaries().items()},
            "metadata": self.metadata,
        }


def _latin_hypercube(dist: Distribution[Any], stream: RandomStream, n: int) -> npt.NDArray[Any]:
    """Stratified sample via the inverse CDF (only for distributions scipy can invert)."""
    from simulsi.randomness import distributions as d

    u = (stream.generator.permutation(n) + stream.generator.random(n)) / n
    from scipy import stats as st

    if isinstance(dist, d.Uniform):
        return dist.low + u * (dist.high - dist.low)
    if isinstance(dist, d.Normal):
        return np.asarray(st.norm.ppf(u, dist.mean, dist.std))
    if isinstance(dist, d.Exponential):
        return np.asarray(st.expon.ppf(u, scale=1 / dist.rate))
    if isinstance(dist, d.Triangular):
        span = dist.high - dist.low
        c = (dist.mode - dist.low) / span
        return np.asarray(st.triang.ppf(u, c, loc=dist.low, scale=span))
    if isinstance(dist, d.LogNormal):
        return np.asarray(st.lognorm.ppf(u, dist.sigma, scale=math.exp(dist.mu)))
    if isinstance(dist, d.Gamma):
        return np.asarray(st.gamma.ppf(u, dist.shape, scale=dist.scale))
    if isinstance(dist, d.Constant):
        return np.full(n, dist.value)
    raise ValueError(f"latin hypercube sampling does not support {type(dist).__name__}")


def sample_inputs(
    parameters: Mapping[str, DistributionLike | Mapping[str, Any]],
    iterations: int,
    seed: int,
    sampling: Literal["random", "lhs"] = "random",
) -> dict[str, npt.NDArray[Any]]:
    """Draw ``iterations`` values for every uncertain parameter (independent streams)."""
    root = RandomStream(seed)
    out: dict[str, npt.NDArray[Any]] = {}
    for name, spec in parameters.items():
        dist = as_distribution(spec)
        stream = root.stream(f"input:{name}")
        out[name] = (
            _latin_hypercube(dist, stream, iterations)
            if sampling == "lhs"
            else dist.sample_n(stream, iterations)
        )
    return out


def _accepts_rng(fn: Callable[..., Any]) -> bool:
    try:
        return "rng" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def monte_carlo(
    model: Callable[..., Any] | Model,
    parameters: Mapping[str, DistributionLike | Mapping[str, Any]],
    iterations: int = 10_000,
    *,
    seed: int = 0,
    fixed: Mapping[str, Any] | None = None,
    vectorized: bool = False,
    objective: Callable[[Mapping[str, Any]], float] | None = None,
    sampling: Literal["random", "lhs"] = "random",
) -> MonteCarloResult:
    """Sample uncertain ``parameters`` and evaluate ``model`` for each draw.

    ``model`` may be:

    * a function ``f(**params) -> float | dict[str, float]``; if it has an
      ``rng`` argument it also receives a per-iteration :class:`RandomStream`
      for internal randomness;
    * with ``vectorized=True``, a function receiving whole arrays (one per
      parameter) and returning an array or a dict of arrays - much faster;
    * a :class:`~simulsi.core.model.Model`, in which case every iteration is
      one simulation run (outputs = the run's flat metrics).

    ``objective`` derives an extra output named ``"objective"`` from each
    iteration's outputs (and fixed/sampled inputs).
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    t0 = time.perf_counter()
    inputs = sample_inputs(parameters, iterations, seed, sampling)
    fixed = dict(fixed or {})
    outputs: dict[str, list[float]] | dict[str, FloatArray]

    if vectorized:
        if isinstance(model, Model):
            raise ValueError("vectorized=True is not available for simulation models")
        res = model(**fixed, **inputs)
        arrays = res if isinstance(res, Mapping) else {"value": res}
        outputs = {
            k: np.broadcast_to(np.asarray(v, dtype=float), (iterations,)).copy()
            for k, v in arrays.items()
        }
        if objective is not None:
            outputs["objective"] = np.asarray(
                objective({**fixed, **inputs, **outputs}), dtype=float
            )
    else:
        lists: dict[str, list[float]] = {}
        rng_root = RandomStream(derive_seed(seed, "iterations"))
        wants_rng = not isinstance(model, Model) and _accepts_rng(model)
        for i in range(iterations):
            params = {
                **fixed,
                **{k: v[i].item() if hasattr(v[i], "item") else v[i] for k, v in inputs.items()},
            }
            if isinstance(model, Model):
                values: Any = model.simulate(
                    params, seed=derive_seed(seed, "replication", i)
                ).metrics
            elif wants_rng:
                values = model(**params, rng=RandomStream(derive_seed(rng_root.seed, i)))
            else:
                values = model(**params)
            row = dict(values) if isinstance(values, Mapping) else {"value": float(values)}
            if objective is not None:
                row["objective"] = float(objective({**params, **row}))
            for k, v in row.items():
                lists.setdefault(k, [math.nan] * i).append(float(v))
            for k in lists:
                if len(lists[k]) < i + 1:
                    lists[k].append(math.nan)
        outputs = {k: np.asarray(v, dtype=float) for k, v in lists.items()}
    return MonteCarloResult(
        inputs=inputs,
        outputs=dict(outputs),
        iterations=iterations,
        seed=seed,
        runtime=time.perf_counter() - t0,
        sampling=sampling,
        metadata={
            "parameters": {
                k: as_distribution(v).to_spec() if as_distribution(v).kind else repr(v)
                for k, v in parameters.items()
            },
            "fixed": fixed,
        },
    )
