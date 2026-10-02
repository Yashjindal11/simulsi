"""A thin bridge from simulation models to optimisation libraries.

SimulSI does not ship an optimiser. :class:`Objective` turns a model + metric
into a plain function that scipy.optimize, OR-Tools callbacks, evolutionary
or Bayesian optimisation libraries can call - none of which is a dependency.

>>> from simulsi.models import mmc
>>> obj = Objective(mmc, "resource.server.wait.mean", ["servers"], replications=2)
>>> obj.bounds
[(1.0, None)]
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from simulsi.core.model import Model
from simulsi.errors import ConfigError


@dataclass
class Evaluation:
    parameters: dict[str, Any]
    value: float
    metrics: dict[str, float] = field(default_factory=dict)


class Objective:
    """``f(x) -> float`` for a vector (or mapping) of decision variables.

    * ``x`` is ordered like ``parameters``; integer parameters are rounded.
    * Every evaluation uses the same replication seeds (common random
      numbers), so differences between candidates are not swamped by noise.
      The objective is still *stochastic* across seeds: validate the optimum
      with a fresh :class:`~simulsi.Experiment` using a different seed.
    * ``transform`` maps the per-run metrics to a scalar, e.g. a cost model:
      ``transform=lambda m: costs.calculate(m).total_cost``.
    * Invalid parameter vectors (outside bounds) return ``invalid_value``
      instead of raising, which suits derivative-free optimisers.
    """

    def __init__(
        self,
        model: Model,
        metric: str | None,
        parameters: Sequence[str],
        *,
        fixed: Mapping[str, Any] | None = None,
        replications: int = 5,
        seed: int = 0,
        minimize: bool = True,
        transform: Callable[[Mapping[str, float]], float] | None = None,
        invalid_value: float = math.inf,
    ) -> None:
        if metric is None and transform is None:
            raise ConfigError("give a metric or a transform")
        for p in parameters:
            if p not in model.parameters:
                raise ConfigError(f"model has no parameter {p!r}")
        self.model = model
        self.metric = metric
        self.parameters = list(parameters)
        self.fixed = dict(fixed or {})
        self.replications = replications
        self.seed = seed
        self.sign = 1.0 if minimize else -1.0
        self.transform = transform
        self.invalid_value = invalid_value
        self.history: list[Evaluation] = []
        self._cache: dict[tuple[Any, ...], float] = {}

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [
            (self.model.parameters[p].low, self.model.parameters[p].high) for p in self.parameters
        ]

    def to_params(self, x: Sequence[float] | Mapping[str, Any]) -> dict[str, Any]:
        values = dict(x) if isinstance(x, Mapping) else dict(zip(self.parameters, x, strict=True))
        out = dict(self.fixed)
        for p in self.parameters:
            v = values[p]
            if self.model.parameters[p].kind == "int":
                v = round(float(v))
            out[p] = float(v) if self.model.parameters[p].kind == "float" else v
        return out

    def evaluate(self, x: Sequence[float] | Mapping[str, Any]) -> Evaluation:
        """Run the replications for ``x`` and return the (unsigned) objective value."""
        from simulsi.randomness.stream import derive_seed

        params = self.to_params(x)
        if self.model.check_parameters(params):
            return Evaluation(params, math.nan)
        total = 0.0
        sums: dict[str, float] = {}
        for r in range(self.replications):
            m = self.model.simulate(params, seed=derive_seed(self.seed, "replication", r)).metrics
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
            if self.transform is not None:
                total += float(self.transform(m))
            else:
                assert self.metric is not None
                if self.metric not in m:
                    raise KeyError(f"model produced no metric {self.metric!r}")
                total += m[self.metric]
        ev = Evaluation(
            params, total / self.replications, {k: v / self.replications for k, v in sums.items()}
        )
        self.history.append(ev)
        return ev

    def __call__(self, x: Sequence[float] | Mapping[str, Any]) -> float:
        params = self.to_params(x)
        key = tuple(sorted((k, repr(v)) for k, v in params.items()))
        if key in self._cache:
            return self._cache[key]
        ev = self.evaluate(params)
        value = self.invalid_value if math.isnan(ev.value) else self.sign * ev.value
        self._cache[key] = value
        return value

    def best(self) -> Evaluation | None:
        valid = [e for e in self.history if not math.isnan(e.value)]
        if not valid:
            return None
        return min(valid, key=lambda e: self.sign * e.value)
