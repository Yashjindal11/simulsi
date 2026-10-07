"""Flow helpers: batching, split/join and routing.

* :func:`batch` - collect ``size`` items from a queue (optionally with a timeout).
* :func:`split` / :func:`join` - fork work into parallel processes and wait
  for all of them.
* :class:`Router` - send entities down one of several paths with given
  probabilities, counting how many take each path.
* :func:`shortest_queue` - pick the resource with the fewest waiting.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from simulsi.randomness.distributions import Categorical

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation
    from simulsi.processes.process import Process, ProcessGenerator
    from simulsi.queues.queue import Queue
    from simulsi.resources.resource import Resource

T = TypeVar("T")


def batch(
    queue: Queue[T], size: int, *, timeout: float | None = None
) -> Generator[Any, Any, list[T]]:
    """Take ``size`` items from ``queue``: ``items = yield from batch(q, 10)``.

    With ``timeout``, stops early once that much time has passed since the
    call and returns the (possibly shorter, possibly empty) batch collected
    so far.
    """
    if size < 1:
        raise ValueError("batch size must be >= 1")
    sim = queue._require_sim()
    deadline = None if timeout is None else sim.now + timeout
    items: list[T] = []
    while len(items) < size:
        get = queue.get()
        if deadline is None or get.triggered:
            items.append((yield get))
            continue
        remaining = deadline - sim.now
        if remaining <= 0:
            queue.cancel(get)
            break
        timer = sim.timeout(remaining)
        done = yield sim.any_of(get, timer)
        if get in done:
            timer.cancel()
            items.append(done[get])
        else:
            queue.cancel(get)
            break
    return items


def split(sim: Simulation, generators: Iterable[ProcessGenerator]) -> list[Process]:
    """Start each generator as a parallel process; pass the result to :func:`join`."""
    return [sim.process(g) for g in generators]


def join(processes: Sequence[Process]) -> Generator[Any, Any, list[Any]]:
    """Wait for all ``processes``; returns their return values in order.

    ``results = yield from join(split(sim, [cut(sim), paint(sim)]))``. If a
    process fails, its exception is raised here.
    """
    if processes:
        yield processes[0].sim.all_of(processes)
    return [p.value for p in processes]


class Router:
    """Probabilistic routing: ``router.choose()`` returns one of the route names.

    Each route is counted in the metric ``route.<name>.<route>``. Draws come
    from the stream ``route:<name>``, so adding a router does not change the
    other random numbers in the model.
    """

    def __init__(self, sim: Simulation, name: str, routes: Mapping[str, float]) -> None:
        self.sim = sim
        self.name = name
        self._dist = Categorical(dict(routes))
        self._stream = sim.stream(f"route:{name}")
        self._counters = {r: sim.metrics.counter(f"route.{name}.{r}") for r in routes}

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(self._counters)

    def choose(self) -> str:
        route: str = self._dist.sample(self._stream)
        self._counters[route].increment()
        return route


def shortest_queue(resources: Sequence[Resource]) -> Resource:
    """The resource with the fewest waiting requests; ties go to the most free
    units, then to the earliest in ``resources``."""
    if not resources:
        raise ValueError("shortest_queue needs at least one resource")
    return min(resources, key=lambda r: (r.queue_size, -r.available))
