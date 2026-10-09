"""Airline operations: day-ahead forecasts, a live digital twin, recovery and planning.

Load tomorrow's schedule (``Schedule.from_csv``), say how the operation
works (``OpsConfig``: turn times, runway rates, gates, curfews, crew duty
limits, spares, standby crews) and how delays arise (``DelayModel``, which
can be fitted to history), then

* :func:`forecast` - per-flight on-time and cancellation probabilities,
  delay ranges, fragile rotations, connection risk and alerts;
* pass an :class:`OpsState` to forecast the rest of the day from *now*;
* :func:`compare_plans` / :func:`recover` - test or search controller
  actions (cancel, retime, swap aircraft) with common random numbers;
* :func:`plan_reserves`, :func:`optimize_buffers`, :func:`schedule_impact`
  - spares and standby crews, schedule padding, new flights;
* :func:`fit_delay_model`, :func:`backtest` - calibrate on history and
  check that probabilities mean what they say;
* :func:`turnaround`, :func:`recommend_mct`, :func:`overbooking`,
  :func:`checkin_staffing` - ground and passenger decisions.
"""

from simulsi.aviation.calibration import (
    History,
    backtest,
    calibrate,
    fit_delay_model,
    fit_late_turns,
    fit_recalibration,
    fit_turn_times,
    reliability_table,
)
from simulsi.aviation.config import (
    DelayModel,
    OpsConfig,
    WeatherEvent,
    closure,
    morning_fog,
    snow,
    thunderstorm,
)
from simulsi.aviation.engine import DayOutcome, FlightOutcome, network_model, simulate_day
from simulsi.aviation.forecast import Alert, Forecast, forecast, rolling_forecast, state_at
from simulsi.aviation.ground import TurnaroundTask, recommend_mct, recommend_min_turn, turnaround
from simulsi.aviation.passengers import (
    checkin_staffing,
    eu261_compensation,
    overbooking,
)
from simulsi.aviation.planning import optimize_buffers, plan_reserves, schedule_impact
from simulsi.aviation.recovery import compare_plans, recover
from simulsi.aviation.schedule import Connection, Flight, Schedule, format_time, parse_time
from simulsi.aviation.state import (
    Cancel,
    FlightStatus,
    OpsState,
    Retime,
    Swap,
    apply_actions,
    parse_action,
)

__all__ = [
    "Alert",
    "Cancel",
    "Connection",
    "DayOutcome",
    "DelayModel",
    "Flight",
    "FlightOutcome",
    "FlightStatus",
    "Forecast",
    "History",
    "OpsConfig",
    "OpsState",
    "Retime",
    "Schedule",
    "Swap",
    "TurnaroundTask",
    "WeatherEvent",
    "apply_actions",
    "backtest",
    "calibrate",
    "checkin_staffing",
    "closure",
    "compare_plans",
    "eu261_compensation",
    "fit_delay_model",
    "fit_late_turns",
    "fit_recalibration",
    "fit_turn_times",
    "forecast",
    "format_time",
    "morning_fog",
    "network_model",
    "optimize_buffers",
    "overbooking",
    "parse_action",
    "parse_time",
    "plan_reserves",
    "recommend_mct",
    "recommend_min_turn",
    "recover",
    "reliability_table",
    "rolling_forecast",
    "schedule_impact",
    "simulate_day",
    "snow",
    "state_at",
    "thunderstorm",
    "turnaround",
]
