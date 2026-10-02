"""Exception hierarchy for SimulSI."""

from __future__ import annotations

from typing import Any


class SimulsiError(Exception):
    """Base class for all SimulSI errors."""


class SchedulingError(SimulsiError):
    """An event could not be scheduled (e.g. in the past or with an invalid time)."""


class EventStateError(SimulsiError):
    """An operation is not valid for the event's current status."""


class CapacityError(SimulsiError):
    """A resource or queue was configured or used with an invalid capacity."""


class ResourceUsageError(SimulsiError):
    """A resource was released without being held, or used from the wrong simulation."""


class ConfigError(SimulsiError):
    """A configuration file or parameter set is invalid."""


class ModelValidationError(SimulsiError):
    """A model failed validation. ``issues`` holds every problem that was found."""

    def __init__(self, issues: list[Any]) -> None:
        self.issues = issues
        lines = "\n".join(f"  - {issue}" for issue in issues)
        super().__init__(f"model validation failed with {len(issues)} issue(s):\n{lines}")


class Interrupt(Exception):
    """Thrown into a process generator when another component interrupts it.

    ``cause`` carries whatever object the interrupter passed (for example a
    failure description), so the process can decide how to react.
    """

    def __init__(self, cause: Any = None) -> None:
        super().__init__(cause)
        self.cause = cause
