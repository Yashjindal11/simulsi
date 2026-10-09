"""Movement on networks: travel times, shortest paths and trips that take simulated time.

A :class:`Network` is a set of places joined by links with a travel time
(roads between depots, aisles in a warehouse, routes between airports).
Shortest paths use Dijkstra's algorithm and are cached per origin.
:meth:`Network.trip` is a process step: ``yield from net.trip(sim, a, b)``
walks the path edge by edge so the simulation clock advances by the
travel time, optionally reporting each leg (to animate or log positions).

Build one from explicit links, from a rectangular :func:`grid_network`,
or from coordinates with :meth:`Network.from_coordinates` (straight-line
or great-circle distance divided by a speed).

>>> net = grid_network(3, 3, cell_time=2.0)
>>> net.travel_time((0, 0), (2, 1))
6.0
>>> net.path((0, 0), (0, 2))
[(0, 0), (0, 1), (0, 2)]
>>> net.nearest((2, 2), [(0, 0), (1, 2)])
(1, 2)
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation

Node = Hashable
EARTH_RADIUS_KM = 6371.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two ``(latitude, longitude)`` points in degrees."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


class Network:
    """A weighted graph of places with travel times on its links."""

    def __init__(
        self,
        links: Iterable[tuple[Node, Node, float]] = (),
        *,
        directed: bool = False,
        name: str = "network",
    ) -> None:
        self.name = name
        self.directed = directed
        self._adj: dict[Node, dict[Node, float]] = {}
        self._cache: dict[Node, tuple[dict[Node, float], dict[Node, Node]]] = {}
        for a, b, t in links:
            self.add_link(a, b, t)

    @classmethod
    def from_coordinates(
        cls,
        points: Mapping[Node, tuple[float, float]],
        links: Iterable[tuple[Node, Node]] | None = None,
        *,
        speed: float = 1.0,
        metric: Literal["euclidean", "haversine"] = "euclidean",
        directed: bool = False,
        name: str = "network",
    ) -> Network:
        """Links weighted by distance / ``speed``; all pairs are linked when ``links`` is None.

        With ``metric="haversine"`` the points are ``(lat, lon)`` degrees and
        distances are km, so ``speed`` is km per time unit.
        """
        if speed <= 0:
            raise ValueError("speed must be > 0")

        def dist(a: Node, b: Node) -> float:
            if metric == "haversine":
                return haversine_km(points[a], points[b])
            return math.dist(points[a], points[b])

        keys = list(points)
        pairs = (
            links
            if links is not None
            else [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]
        )
        net = cls(directed=directed, name=name)
        for a in keys:
            net._adj.setdefault(a, {})
        for a, b in pairs:
            net.add_link(a, b, dist(a, b) / speed)
        return net

    # -- structure ----------------------------------------------------------------

    def add_link(self, a: Node, b: Node, time: float) -> None:
        """Add (or replace) a link; in an undirected network it works both ways."""
        if time < 0 or math.isnan(time):
            raise ValueError(f"travel time {a!r}->{b!r} must be >= 0")
        self._adj.setdefault(a, {})[b] = float(time)
        self._adj.setdefault(b, {})
        if not self.directed:
            self._adj[b][a] = float(time)
        self._cache.clear()

    def remove_link(self, a: Node, b: Node) -> None:
        """Close a link (a road works, a closed airspace sector)."""
        self._adj.get(a, {}).pop(b, None)
        if not self.directed:
            self._adj.get(b, {}).pop(a, None)
        self._cache.clear()

    @property
    def nodes(self) -> list[Node]:
        return list(self._adj)

    def neighbours(self, node: Node) -> dict[Node, float]:
        return dict(self._adj[node])

    def __contains__(self, node: object) -> bool:
        return node in self._adj

    def __len__(self) -> int:
        return len(self._adj)

    # -- shortest paths ----------------------------------------------------------

    def _tree(self, origin: Node) -> tuple[dict[Node, float], dict[Node, Node]]:
        if origin not in self._adj:
            raise KeyError(f"unknown place {origin!r}")
        if origin not in self._cache:
            dist: dict[Node, float] = {origin: 0.0}
            prev: dict[Node, Node] = {}
            heap: list[tuple[float, int, Node]] = [(0.0, 0, origin)]
            tie = 1
            done: set[Node] = set()
            while heap:
                d, _, u = heapq.heappop(heap)
                if u in done:
                    continue
                done.add(u)
                for v, w in self._adj[u].items():
                    nd = d + w
                    if nd < dist.get(v, math.inf):
                        dist[v], prev[v] = nd, u
                        heapq.heappush(heap, (nd, tie, v))
                        tie += 1
            self._cache[origin] = (dist, prev)
        return self._cache[origin]

    def travel_time(self, a: Node, b: Node) -> float:
        """Shortest travel time from ``a`` to ``b`` (``inf`` if unreachable)."""
        if b not in self._adj:
            raise KeyError(f"unknown place {b!r}")
        return self._tree(a)[0].get(b, math.inf)

    def path(self, a: Node, b: Node) -> list[Node]:
        """The places on a shortest route from ``a`` to ``b``, both included."""
        dist, prev = self._tree(a)
        if b not in dist:
            raise ValueError(f"no route from {a!r} to {b!r}")
        out = [b]
        while out[-1] != a:
            out.append(prev[out[-1]])
        return out[::-1]

    def nearest(self, origin: Node, candidates: Iterable[Node]) -> Node | None:
        """The candidate closest to ``origin`` by travel time (None if none is reachable)."""
        dist = self._tree(origin)[0]
        best = min(candidates, key=lambda c: dist.get(c, math.inf), default=None)
        return best if best is not None and best in dist else None

    def matrix(self, places: Sequence[Node] | None = None) -> list[list[float]]:
        """All-pairs travel times between ``places`` (default: every node)."""
        ps = list(places) if places is not None else self.nodes
        return [[self.travel_time(a, b) for b in ps] for a in ps]

    # -- movement ------------------------------------------------------------------

    def trip(
        self,
        sim: Simulation,
        a: Node,
        b: Node,
        *,
        speed_factor: float = 1.0,
        on_leg: Callable[[Node, Node, float], Any] | None = None,
    ) -> Iterator[float]:
        """Travel from ``a`` to ``b`` inside a process: ``yield from net.trip(sim, a, b)``.

        Each leg takes its travel time divided by ``speed_factor``;
        ``on_leg(from, to, start_time)`` is called as each leg starts. The
        trip time is recorded as ``network.<name>.trip_time``.
        """
        if speed_factor <= 0:
            raise ValueError("speed_factor must be > 0")
        start = sim.now
        route = self.path(a, b)
        for u, v in itertools.pairwise(route):
            if on_leg is not None:
                on_leg(u, v, sim.now)
            yield self._adj[u][v] / speed_factor
        sim.metrics.observe(f"network.{self.name}.trip_time", sim.now - start)

    def __repr__(self) -> str:
        links = sum(len(v) for v in self._adj.values())
        return f"Network({self.name!r}, places={len(self)}, links={links})"


def grid_network(
    width: int,
    height: int,
    *,
    cell_time: float = 1.0,
    diagonal: bool = False,
    blocked: Iterable[tuple[int, int]] = (),
    name: str = "grid",
) -> Network:
    """A ``width`` x ``height`` grid of ``(x, y)`` cells, linked to their 4 (or 8) neighbours.

    Cells in ``blocked`` are left out (walls, shelving, closed streets).
    Diagonal moves take ``cell_time * sqrt(2)``.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    walls = set(blocked)
    cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in walls]
    net = Network(name=name)
    steps = [(1, 0, 1.0), (0, 1, 1.0)]
    if diagonal:
        steps += [(1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))]
    present = set(cells)
    for x, y in cells:
        net._adj.setdefault((x, y), {})
        for dx, dy, f in steps:
            nb = (x + dx, y + dy)
            if nb in present:
                net.add_link((x, y), nb, cell_time * f)
    return net
