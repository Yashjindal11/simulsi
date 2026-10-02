"""Checkpoints of running simulations, restored by deterministic replay.

A running simulation contains suspended Python generators, which cannot be
serialised reliably. SimulSI therefore checkpoints *how to get back* rather
than the raw memory: the model reference, resolved parameters, seed,
simulation options, the number of events executed and the clock. Because a
seeded SimulSI run is deterministic, rebuilding the model and replaying
exactly that many events reproduces the state. A digest of all statistics
(and the pending-event count) is stored too and checked after replay, so a
model that is not deterministic, or that changed since the checkpoint, is
detected instead of silently diverging.

Requirements and costs:

* the simulation must come from :meth:`Model.create` / :meth:`Model.simulate`
  (so it can be rebuilt);
* the model must be reproducible from its seed (no global randomness, wall
  clock, unordered iteration that affects scheduling);
* restoring costs the time to re-run up to the checkpoint. Replay is usually
  much faster than the original interactive or segmented run, but it is not
  instant;
* state changed from *outside* the event loop between runs (e.g. calling
  ``resource.set_capacity`` from your script between two ``run`` calls) is
  not recorded. Schedule such changes as events instead.

Finish hooks (``sim.on_finish``) are not re-run during replay; they run again
at the end of the next ``run``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simulsi._version import __version__
from simulsi.errors import SimulsiError
from simulsi.serialization.io import read_json, to_jsonable, write_json

if TYPE_CHECKING:
    from simulsi.core.model import Model
    from simulsi.core.simulation import Simulation

FORMAT = "simulsi-checkpoint"
FORMAT_VERSION = 1


class CheckpointError(SimulsiError):
    """A checkpoint could not be written, resolved or faithfully restored."""


def _digest(sim: Simulation) -> str:
    hook_values = set(sim.metrics.values)
    flat = {k: v for k, v in sim.flat_metrics().items() if k not in hook_values}
    payload = json.dumps(to_jsonable(flat), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_reference(model: Model) -> dict[str, Any]:
    """How to find ``model`` again in another process (best effort)."""
    from simulsi.config.schema import loaded_model_files
    from simulsi.core.model import _import_object
    from simulsi.models import BUILTIN_MODELS

    ref: dict[str, Any] = {"name": model.name, "version": model.version}
    for key, m in BUILTIN_MODELS.items():
        if m is model:
            ref["builtin"] = key
            return ref
    module = getattr(model.build, "__module__", None)
    qualname = str(getattr(model.build, "__qualname__", "")).removesuffix(".build")
    if module and qualname and "<" not in qualname:
        try:
            found = _import_object(module, qualname) is model
        except (ImportError, AttributeError):
            found = False
        if found:
            files = loaded_model_files()
            if module in files:
                ref["file"] = files[module]
                ref["attr"] = qualname
            else:
                ref["module"] = module
                ref["attr"] = qualname
    return ref


def resolve_reference(ref: dict[str, Any]) -> Model:
    from simulsi.config.schema import resolve_model

    if "builtin" in ref:
        return resolve_model(f"builtin:{ref['builtin']}")
    if "file" in ref:
        return resolve_model(f"{ref['file']}:{ref['attr']}")
    if "module" in ref:
        return resolve_model(f"{ref['module']}:{ref['attr']}")
    raise CheckpointError(
        f"cannot locate model {ref.get('name')!r} automatically; pass it: "
        "Simulation.load_checkpoint(path, model=my_model)"
    )


def save_checkpoint(sim: Simulation, path: str | Path) -> Path:
    if sim._origin is None:
        raise CheckpointError(
            "checkpoints need a simulation built by Model.create() (so it can be rebuilt); "
            "wrap the setup code in a Model"
        )
    model, params, options = sim._origin
    data = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "simulsi": __version__,
        "model": model_reference(model),
        "parameters": params.to_jsonable(),
        "seed": sim.seed,
        "options": options,
        "now": sim.now,
        "events_processed": sim.events_processed,
        "pending_events": len(sim.event_queue),
        "next_event_time": sim.peek() if sim.event_queue else None,
        "metrics_digest": _digest(sim),
    }
    return write_json(data, path)


def load_checkpoint(path: str | Path, *, model: Model | None = None) -> Simulation:
    data = read_json(path)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise CheckpointError(f"{path} is not a SimulSI checkpoint")
    if data.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"unsupported checkpoint format version {data.get('format_version')}")
    ref = data["model"]
    m = model if model is not None else resolve_reference(ref)
    if m.name != ref.get("name") or m.version != ref.get("version"):
        raise CheckpointError(
            f"checkpoint was written by model {ref.get('name')!r} v{ref.get('version')}, "
            f"not {m.name!r} v{m.version}"
        )
    sim = m.create(data["parameters"], seed=int(data["seed"]), **dict(data.get("options") or {}))
    target = int(data["events_processed"])
    if sim._replay(target) != target:
        raise CheckpointError("replay ran out of events before reaching the checkpoint")
    now = float(data["now"])
    if sim.now > now:
        raise CheckpointError(f"replay overshot the checkpoint time ({sim.now} > {now})")
    sim.clock.advance_to(now)
    next_time = sim.peek() if sim.event_queue else None
    if (
        len(sim.event_queue) != int(data["pending_events"])
        or next_time != data.get("next_event_time")
        or _digest(sim) != data["metrics_digest"]
    ):
        raise CheckpointError(
            "replayed state differs from the checkpoint: the model is not deterministic for its "
            "seed, or it changed since the checkpoint was written"
        )
    return sim
