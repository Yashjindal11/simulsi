"""Built-in models: ``builtin:<name>`` in the CLI, configs and the dashboard.

* ``mmc`` - the M/M/c queue (checked against Erlang C)
* ``airline`` - aircraft rotations, delay propagation, crews, gates, spares, weather
* ``epidemic`` - stochastic SEIR with hospital capacity and a lockdown policy
* ``supply_chain`` - four-stage beer game and the bullwhip effect
* ``ride_hailing`` - matching, pickup-time feedback, cancellations, surge pricing
* ``cloud_autoscaling`` - bursty traffic, cold starts, timeouts, autoscaling
* ``traffic_signal`` - fixed vs actuated signal timing, compared with Webster

Every model has ``presets``: named what-if scenarios (``model.preset_scenarios()``).
"""

from simulsi.models.airline import airline
from simulsi.models.cloud_autoscaling import cloud_autoscaling
from simulsi.models.epidemic import epidemic
from simulsi.models.queueing import BUILTIN_MODELS, erlang_c, mmc
from simulsi.models.ride_hailing import ride_hailing
from simulsi.models.supply_chain import supply_chain
from simulsi.models.traffic_signal import traffic_signal, webster_delay

BUILTIN_MODELS.update(
    {
        m.name: m
        for m in (airline, epidemic, supply_chain, ride_hailing, cloud_autoscaling, traffic_signal)
    }
)

__all__ = [
    "BUILTIN_MODELS",
    "airline",
    "cloud_autoscaling",
    "epidemic",
    "erlang_c",
    "mmc",
    "ride_hailing",
    "supply_chain",
    "traffic_signal",
    "webster_delay",
]
