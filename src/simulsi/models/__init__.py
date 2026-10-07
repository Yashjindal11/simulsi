"""Built-in models: ``builtin:<name>`` in the CLI, configs and the dashboard.

* ``mmc`` - the M/M/c queue (checked against Erlang C)
* ``airline`` - aircraft rotations, delay propagation, crews, gates, spares, weather
* ``airport_turnaround`` - parallel ground-handling tasks, shared crews, banked schedules
* ``disruption_recovery`` - after a hub storm: delay vs cancel vs spare aircraft
* ``emergency_department`` - triage, acuity, beds, doctors, boarding and diversion
* ``restaurant`` - table mix, flexible seating, reservations and the kitchen
* ``ev_charging`` - fast/slow chargers under a shared grid limit
* ``inventory`` - the (s, S) reorder policy and its costs
* ``theme_park`` - ride queues, express passes and app-based routing
* ``warehouse`` - discrete vs batch picking, packing and truck cut-offs
* ``epidemic`` - stochastic SEIR with hospital capacity and a lockdown policy
* ``supply_chain`` - four-stage beer game and the bullwhip effect
* ``ride_hailing`` - matching, pickup-time feedback, cancellations, surge pricing
* ``cloud_autoscaling`` - bursty traffic, cold starts, timeouts, autoscaling
* ``traffic_signal`` - fixed vs actuated signal timing, compared with Webster

Every model has ``presets``: named what-if scenarios (``model.preset_scenarios()``).

Other packages can add models through the ``simulsi.models`` entry-point
group, for example in ``pyproject.toml``::

    [project.entry-points."simulsi.models"]
    bakery = "mypackage.models:bakery"

They then work as ``builtin:bakery`` in the CLI, configs and the dashboard.
"""

from simulsi.models.airline import airline
from simulsi.models.airport_turnaround import airport_turnaround
from simulsi.models.cloud_autoscaling import cloud_autoscaling
from simulsi.models.disruption_recovery import disruption_recovery
from simulsi.models.emergency_department import emergency_department
from simulsi.models.epidemic import epidemic
from simulsi.models.ev_charging import ev_charging
from simulsi.models.inventory import inventory
from simulsi.models.queueing import BUILTIN_MODELS, erlang_c, mmc
from simulsi.models.restaurant import restaurant
from simulsi.models.ride_hailing import ride_hailing
from simulsi.models.supply_chain import supply_chain
from simulsi.models.theme_park import theme_park
from simulsi.models.traffic_signal import traffic_signal, webster_delay
from simulsi.models.warehouse import warehouse

BUILTIN_MODELS.update(
    {
        m.name: m
        for m in (
            airline,
            airport_turnaround,
            disruption_recovery,
            emergency_department,
            epidemic,
            ev_charging,
            inventory,
            restaurant,
            theme_park,
            warehouse,
            supply_chain,
            ride_hailing,
            cloud_autoscaling,
            traffic_signal,
        )
    }
)


def load_plugin_models() -> dict[str, str]:
    """Register models from the ``simulsi.models`` entry points; returns load errors by name."""
    import warnings
    from importlib.metadata import entry_points

    from simulsi.core.model import Model

    errors: dict[str, str] = {}
    for ep in entry_points(group="simulsi.models"):
        if ep.name in BUILTIN_MODELS:
            continue
        try:
            obj = ep.load()
        except Exception as exc:  # a broken plugin must not break SimulSI
            errors[ep.name] = f"{type(exc).__name__}: {exc}"
            continue
        if not isinstance(obj, Model):
            errors[ep.name] = f"{ep.value} is not a simulsi Model"
            continue
        BUILTIN_MODELS[ep.name] = obj
    for name, err in errors.items():
        warnings.warn(f"simulsi model plugin {name!r} could not be loaded: {err}", stacklevel=2)
    return errors


load_plugin_models()

__all__ = [
    "BUILTIN_MODELS",
    "airline",
    "airport_turnaround",
    "cloud_autoscaling",
    "disruption_recovery",
    "emergency_department",
    "epidemic",
    "erlang_c",
    "ev_charging",
    "inventory",
    "load_plugin_models",
    "mmc",
    "restaurant",
    "ride_hailing",
    "supply_chain",
    "theme_park",
    "traffic_signal",
    "warehouse",
    "webster_delay",
]
