from __future__ import annotations

import math
from typing import Any

import pytest

from simulsi import Simulation
from simulsi.spatial import Network, grid_network, haversine_km


def test_shortest_paths_closures_and_directed_links() -> None:
    net = Network([("A", "B", 4), ("B", "C", 3), ("A", "C", 10), ("C", "D", 1)], name="roads")
    assert net.travel_time("A", "D") == 8.0 and net.path("A", "D") == ["A", "B", "C", "D"]
    net.remove_link("B", "C")
    assert net.travel_time("A", "D") == 11.0
    net.add_link("E", "F", 1)
    assert net.travel_time("A", "E") == math.inf and net.nearest("A", ["E", "D"]) == "D"
    assert net.nearest("A", ["E"]) is None
    with pytest.raises(ValueError, match="no route"):
        net.path("A", "F")
    with pytest.raises(KeyError):
        net.travel_time("A", "Z")
    with pytest.raises(ValueError):
        net.add_link("A", "B", -1)
    one_way = Network([("x", "y", 2)], directed=True)
    assert one_way.travel_time("x", "y") == 2 and one_way.travel_time("y", "x") == math.inf
    assert one_way.matrix() == [[0.0, 2.0], [math.inf, 0.0]]
    assert "x" in one_way and len(one_way) == 2 and "places=2" in repr(one_way)


def test_grid_with_walls_and_diagonals() -> None:
    walls = [(1, 0), (1, 1)]
    net = grid_network(3, 3, blocked=walls)
    assert (1, 0) not in net and net.travel_time((0, 0), (2, 0)) == 6.0
    diag = grid_network(3, 3, diagonal=True)
    assert diag.travel_time((0, 0), (2, 2)) == pytest.approx(2 * math.sqrt(2))
    with pytest.raises(ValueError):
        grid_network(0, 3)


def test_coordinates_and_haversine() -> None:
    jfk, lax = (40.6413, -73.7781), (33.9416, -118.4085)
    assert haversine_km(jfk, lax) == pytest.approx(3983, rel=0.01)
    air = Network.from_coordinates(
        {"JFK": jfk, "LAX": lax, "ORD": (41.9742, -87.9073)}, speed=800 / 60, metric="haversine"
    )
    assert 290 < air.travel_time("JFK", "LAX") < 310  # minutes at 800 km/h
    flat = Network.from_coordinates({"a": (0, 0), "b": (3, 4)}, [("a", "b")], speed=0.5)
    assert flat.travel_time("a", "b") == 10.0
    with pytest.raises(ValueError):
        Network.from_coordinates({"a": (0, 0)}, speed=0)


def test_trip_advances_the_clock_and_reports_legs() -> None:
    net = grid_network(4, 1, cell_time=5.0, name="aisle")
    sim = Simulation()
    legs: list[tuple[Any, Any, float]] = []

    def picker(sim: Simulation) -> Any:
        yield from net.trip(sim, (0, 0), (3, 0), on_leg=lambda u, v, t: legs.append((u, v, t)))
        yield from net.trip(sim, (3, 0), (0, 0), speed_factor=2.0)

    sim.process(picker(sim))
    result = sim.run()
    assert sim.now == 22.5
    assert [t for _, _, t in legs] == [0.0, 5.0, 10.0]
    assert result.metrics["network.aisle.trip_time.count"] == 2
    with pytest.raises(ValueError):
        next(net.trip(sim, (0, 0), (1, 0), speed_factor=0))
