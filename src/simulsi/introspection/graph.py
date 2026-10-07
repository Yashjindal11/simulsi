"""Model introspection as a graph.

Two complementary views, both *observed* from a run (SimulSI models are
ordinary Python, so the flow cannot be read off the source reliably):

* **flow graph** - where entities go: ``arrival:<type> -> queue/resource ->
  ... -> exit:<type>``, with transition counts. Built from the event log, so
  run with ``trace=True`` and associate entities with requests (pass
  ``entity=`` to ``sim.process`` or ``sim.request``).
* **state graph** - the state machine of each entity type, from entity
  histories (``old_state -> new_state`` counts).

Plus the static inventory: resources, queues, entity types, process names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simulsi.core.model import Model
    from simulsi.core.simulation import Simulation
    from simulsi.core.trace import LogRecord


@dataclass
class Node:
    id: str
    kind: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    count: int
    label: str = ""


@dataclass
class ModelGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    kind: str = "flow"

    def add_node(self, id: str, kind: str, label: str | None = None, **attrs: Any) -> None:
        if id not in self.nodes:
            self.nodes[id] = Node(id, kind, label or id, dict(attrs))

    def successors(self, node: str) -> list[str]:
        return [e.target for e in self.edges if e.source == node]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "nodes": [n.__dict__ for n in self.nodes.values()],
            "edges": [e.__dict__ for e in self.edges],
        }

    def to_dot(self) -> str:
        shapes = {
            "resource": "box",
            "queue": "cds",
            "source": "ellipse",
            "sink": "doublecircle",
            "state": "ellipse",
        }
        lines = [f'digraph "{self.kind}" {{', "  rankdir=LR;"]
        for n in self.nodes.values():
            lines.append(
                f'  "{_esc(n.id)}" [label="{_esc(n.label)}", shape={shapes.get(n.kind, "ellipse")}];'
            )
        for e in self.edges:
            lines.append(f'  "{_esc(e.source)}" -> "{_esc(e.target)}" [label="{e.count}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        ids = {nid: f"n{i}" for i, nid in enumerate(self.nodes)}
        lines = ["flowchart LR"]
        for nid, n in self.nodes.items():
            text = _esc(n.label).replace('"', "'")
            lines.append(
                f'  {ids[nid]}["{text}"]' if n.kind != "queue" else f'  {ids[nid]}[("{text}")]'
            )
        for e in self.edges:
            lines.append(f"  {ids[e.source]} -->|{e.count}| {ids[e.target]}")
        return "\n".join(lines)


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def inventory(sim: Simulation) -> dict[str, Any]:
    """Static view of the components a simulation has registered."""
    return {
        "resources": {
            n: {"capacity": r.capacity, "discipline": r.describe()["discipline"]}
            for n, r in sim.resources.items()
        },
        "queues": {n: q.describe() for n, q in sim.queues.items()},
        "containers": {n: c.describe() for n, c in sim.containers.items()},
        "entity_types": sorted(set(sim._created) | set(sim._id_counters)),
        "processes_started": sim._processes_started,
        "pending_events": len(sim.event_queue),
    }


def flow_graph(records: Iterable[LogRecord]) -> ModelGraph:
    """Entity flow between arrival, queues, resources and exit, from an event log."""
    g = ModelGraph(kind="flow")
    last: dict[str, str] = {}
    etype: dict[str, str] = {}
    counts: Counter[tuple[str, str]] = Counter()

    def move(entity: str, node: str) -> None:
        prev = last.get(entity)
        if prev is not None and prev != node:
            counts[(prev, node)] += 1
        last[entity] = node

    for r in records:
        if r.entity is None:
            continue
        if r.event_type == "entity.created":
            t = str(r.metadata.get("entity_type") or r.entity.rsplit("-", 1)[0])
            etype[r.entity] = t
            node = f"arrival:{t}"
            g.add_node(node, "source", f"Arrival ({t})")
            last[r.entity] = node
        elif r.event_type == "resource.request" and r.resource:
            g.add_node(f"wait:{r.resource}", "queue", f"Queue ({r.resource})")
            move(r.entity, f"wait:{r.resource}")
        elif r.event_type == "resource.acquire" and r.resource:
            g.add_node(f"resource:{r.resource}", "resource", f"Service ({r.resource})")
            move(r.entity, f"resource:{r.resource}")
        elif r.event_type in ("queue.put",) and r.resource:
            g.add_node(f"queue:{r.resource}", "queue", f"Queue ({r.resource})")
            move(r.entity, f"queue:{r.resource}")
        elif r.event_type == "resource.renege" and r.resource:
            g.add_node("reneged", "sink", "Reneged")
            move(r.entity, "reneged")
        elif r.event_type == "entity.state" and r.new_state == "disposed":
            t = etype.get(r.entity, r.entity.rsplit("-", 1)[0])
            g.add_node(f"exit:{t}", "sink", f"Departure ({t})")
            move(r.entity, f"exit:{t}")
    g.edges = [Edge(s, t, n) for (s, t), n in sorted(counts.items())]
    return g


def state_graph(sim: Simulation, entity_type: str | None = None) -> ModelGraph:
    """State-transition counts from the histories of (still registered or disposed) entities.

    Only entities still reachable are included; for complete coverage run with
    ``trace=True`` and use :func:`state_graph_from_log`.
    """
    g = ModelGraph(kind="state")
    counts: Counter[tuple[str, str]] = Counter()
    for e in sim.entities.values():
        if entity_type is not None and e.entity_type != entity_type:
            continue
        for h in e.history:
            counts[(f"{e.entity_type}:{h.old_state}", f"{e.entity_type}:{h.new_state}")] += 1
    for (s, t), n in counts.items():
        g.add_node(s, "state", s)
        g.add_node(t, "state", t)
        g.edges.append(Edge(s, t, n))
    return g


def state_graph_from_log(records: Iterable[LogRecord]) -> ModelGraph:
    g = ModelGraph(kind="state")
    counts: Counter[tuple[str, str]] = Counter()
    etype: dict[str, str] = {}
    for r in records:
        if r.event_type == "entity.created" and r.entity:
            etype[r.entity] = str(r.metadata.get("entity_type") or r.entity.rsplit("-", 1)[0])
        if r.event_type == "entity.state" and r.entity and r.old_state and r.new_state:
            t = etype.get(r.entity, r.entity.rsplit("-", 1)[0])
            counts[(f"{t}:{r.old_state}", f"{t}:{r.new_state}")] += 1
    for (s, t), n in sorted(counts.items()):
        g.add_node(s, "state", s)
        g.add_node(t, "state", t)
        g.edges.append(Edge(s, t, n))
    return g


def model_graph(
    model: Model,
    params: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    duration: float | None = None,
    max_log_records: int = 200_000,
) -> dict[str, Any]:
    """Run ``model`` briefly with tracing and return inventory, flow and state graphs."""
    sim = model.create(params, seed=seed, trace=True, max_log_records=max_log_records)
    horizon = duration if duration is not None else model.duration
    sim.run(until=horizon)
    assert sim.log is not None
    return {
        "inventory": inventory(sim),
        "flow": flow_graph(sim.log),
        "states": state_graph_from_log(sim.log),
    }
