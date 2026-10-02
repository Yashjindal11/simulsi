"""Model validation: catch configuration and modelling mistakes early.

:func:`validate_model` combines static checks with a short *smoke run*:

========================  ===========================================================
code                      meaning
========================  ===========================================================
parameter                 missing/unknown/out-of-range/invalid-type parameter values
build-error               the build function raised
run-error                 the simulation raised (invalid event time, model bug, ...)
zero-capacity             a resource has capacity 0 at start
no-events                 nothing was scheduled: the model does nothing
resource-not-released     a process finished while still holding a resource unit
possible-deadlock         processes blocked with no pending events at the end
unreached-state           a declared entity state never occurred in the smoke run
undeclared-state          an entity entered a state not in the declared list
no-duration               the model has no duration, so runs may never terminate
========================  ===========================================================

The smoke run is evidence, not proof: a state not reached in a short run
might be reachable in a longer one, and deadlocks that need rare event
orderings may not show up.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from simulsi.core.model import Model
from simulsi.errors import ModelValidationError

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ModelValidationError(self.errors)

    def format(self) -> str:
        if not self.issues:
            return "OK: no issues found"
        head = f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([head, *(f"  {i}" for i in self.issues)])

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": [asdict(i) for i in self.issues], "info": self.info}


def validate_model(
    model: Model,
    params: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
    smoke_duration: float | None = None,
    expected_states: Mapping[str, Iterable[str]] | None = None,
) -> ValidationReport:
    """Check ``model`` with ``params``. ``smoke_duration`` defaults to 10% of the model duration."""
    report = ValidationReport()
    add = report.issues.append

    for msg in model.check_parameters(params):
        add(Issue("error", "parameter", msg))
    if report.errors:
        return report
    if model.duration is None:
        add(
            Issue(
                "warning",
                "no-duration",
                "model has no duration; runs stop only when events run out",
            )
        )

    try:
        sim = model.create(params, seed=seed, trace=expected_states is not None)
    except Exception as exc:
        add(Issue("error", "build-error", f"{type(exc).__name__}: {exc}"))
        return report

    for name, r in sim.resources.items():
        if r.capacity == 0:
            add(
                Issue(
                    "warning",
                    "zero-capacity",
                    f"resource {name!r} has capacity 0; requests will wait forever",
                )
            )

    horizon = smoke_duration
    if horizon is None:
        horizon = (model.duration or 100.0) * 0.1
    report.info["smoke_duration"] = horizon
    try:
        result = sim.run(until=horizon)
    except Exception as exc:
        add(Issue("error", "run-error", f"at t={sim.now:g}: {type(exc).__name__}: {exc}"))
        return report

    report.info["events"] = result.events_processed
    if result.events_processed == 0:
        add(
            Issue(
                "warning",
                "no-events",
                "no events were executed; did the build function start any process?",
            )
        )
    for w in result.warnings:
        if "while holding" in w:
            add(Issue("warning", "resource-not-released", w))
        elif "blocked" in w:
            add(Issue("warning", "possible-deadlock", w))
        else:
            add(Issue("warning", "runtime-warning", w))

    if expected_states is not None and sim.log is not None:
        seen: dict[str, set[str]] = {}
        etype: dict[str, str] = {}
        for rec in sim.log:
            if rec.event_type == "entity.created" and rec.entity:
                t = str(rec.metadata.get("entity_type", ""))
                etype[rec.entity] = t
                if rec.new_state:
                    seen.setdefault(t, set()).add(rec.new_state)
            elif rec.event_type == "entity.state" and rec.entity and rec.new_state:
                seen.setdefault(etype.get(rec.entity, ""), set()).add(rec.new_state)
        for t, states in expected_states.items():
            declared = set(states)
            got = seen.get(t, set())
            for s in sorted(declared - got):
                add(
                    Issue(
                        "warning", "unreached-state", f"{t}: state {s!r} never reached in smoke run"
                    )
                )
            for s in sorted(got - declared - {"created", "disposed"}):
                add(Issue("warning", "undeclared-state", f"{t}: entered undeclared state {s!r}"))
    return report
