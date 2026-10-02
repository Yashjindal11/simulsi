"""Processes: generator functions that describe behaviour over simulated time.

A process is a Python generator. Each ``yield`` hands the engine something to
wait for and the generator resumes with the result:

* a number or ``timedelta``      -> hold for that long (``yield 4.5``)
* ``sim.timeout(d, value)``      -> same, with a value returned from the yield
* ``sim.request(resource)``      -> resumes once a unit is granted
* ``queue.get()`` / ``put()``    -> resumes when an item arrives / space frees
* ``sim.signal()``               -> resumes when someone calls ``signal.succeed()``
* ``sim.all_of(...)`` / ``any_of(...)`` -> combinations
* another :class:`Process`      -> resumes when it finishes (with its return value)

Exceptions travel too: if the awaited thing fails, the exception is raised at
the ``yield``; an ``Interrupt`` is raised there when the process is interrupted.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from simulsi.errors import EventStateError, Interrupt
from simulsi.events.event import Event, Priority

if TYPE_CHECKING:
    from simulsi.core.simulation import Simulation
    from simulsi.entities.entity import Entity
    from simulsi.resources.resource import Request

ProcessGenerator = Generator[Any, Any, Any]
Callback = Callable[["Waitable"], None]

_PENDING: Any = object()


class Waitable:
    """Something a process can wait for. Triggers once, with a value or an exception."""

    __slots__ = ("_callbacks", "_exc", "_value", "sim")

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self._callbacks: list[Callback] = []
        self._value: Any = _PENDING
        self._exc: BaseException | None = None

    @property
    def triggered(self) -> bool:
        return self._value is not _PENDING or self._exc is not None

    @property
    def ok(self) -> bool:
        return self.triggered and self._exc is None

    @property
    def value(self) -> Any:
        if self._exc is not None:
            raise self._exc
        if self._value is _PENDING:
            raise EventStateError(f"{self!r} has not been triggered yet")
        return self._value

    @property
    def exception(self) -> BaseException | None:
        return self._exc

    def succeed(self, value: Any = None) -> Waitable:
        """Trigger successfully. Waiters resume at the current simulation time."""
        if self.triggered:
            raise EventStateError(f"{self!r} already triggered")
        self._value = value
        self._dispatch()
        return self

    def fail(self, exc: BaseException) -> Waitable:
        """Trigger with an exception, which is raised in every waiting process."""
        if self.triggered:
            raise EventStateError(f"{self!r} already triggered")
        self._exc = exc
        self._dispatch()
        return self

    def _dispatch(self) -> None:
        # Waiters are resumed through the event queue rather than synchronously,
        # so a release() deep inside one process never re-enters another one.
        if self._callbacks:
            self.sim._schedule_internal(self._run_callbacks, "_dispatch")

    def _run_callbacks(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            cb(self)

    def add_callback(self, cb: Callback) -> None:
        """Call ``cb(self)`` once triggered (immediately-scheduled if already triggered)."""
        if self.triggered:
            self.sim._schedule_internal(lambda: cb(self), "_dispatch")
        else:
            self._callbacks.append(cb)

    def remove_callback(self, cb: Callback) -> None:
        if cb in self._callbacks:
            self._callbacks.remove(cb)

    def __repr__(self) -> str:
        state = "pending" if not self.triggered else ("ok" if self._exc is None else "failed")
        return f"<{type(self).__name__} {state}>"


class Timeout(Waitable):
    """Triggers after ``delay`` time units."""

    __slots__ = ("_event", "delay")

    def __init__(self, sim: Simulation, delay: float, value: Any = None) -> None:
        super().__init__(sim)
        if delay < 0:
            raise ValueError(f"timeout delay must be >= 0, got {delay}")
        self.delay = delay
        self._event: Event = sim._schedule_internal(
            lambda: self._fire(value), "_timeout", delay=delay
        )

    def _fire(self, value: Any) -> None:
        # Runs as a top-level event, so waiters can be resumed synchronously.
        self._value = value
        self._run_callbacks()

    def cancel(self) -> None:
        """Withdraw the timeout if nobody is waiting on it any more."""
        self.sim.cancel(self._event)


class Signal(Waitable):
    """A one-shot flag that model code triggers explicitly with ``succeed``/``fail``."""

    __slots__ = ("name",)

    def __init__(self, sim: Simulation, name: str = "signal") -> None:
        super().__init__(sim)
        self.name = name


class Condition(Waitable):
    """Base for :class:`AllOf` / :class:`AnyOf`. Value: ``{waitable: value}`` of triggered children."""

    __slots__ = ("children",)

    def __init__(self, sim: Simulation, children: Iterable[Waitable]) -> None:
        super().__init__(sim)
        self.children = list(children)
        for child in self.children:
            if child.sim is not sim:
                raise EventStateError("cannot combine waitables from different simulations")
        if self._satisfied():
            self._value = self._collect()
            return
        for child in self.children:
            if child.triggered:
                if child._exc is not None:
                    self._exc = child._exc
                    return
            else:
                child.add_callback(self._on_child)

    def _satisfied(self) -> bool:
        raise NotImplementedError

    def _collect(self) -> dict[Waitable, Any]:
        return {c: c._value for c in self.children if c.ok}

    def _on_child(self, child: Waitable) -> None:
        if self.triggered:
            return
        if child._exc is not None:
            self._detach()
            self.fail(child._exc)
        elif self._satisfied():
            self._detach()
            self.succeed(self._collect())

    def _detach(self) -> None:
        for c in self.children:
            c.remove_callback(self._on_child)


class AllOf(Condition):
    __slots__ = ()

    def _satisfied(self) -> bool:
        return all(c.ok for c in self.children)


class AnyOf(Condition):
    __slots__ = ()

    def _satisfied(self) -> bool:
        return not self.children or any(c.ok for c in self.children)


class Process(Waitable):
    """A running generator. Also a :class:`Waitable` that triggers when the generator ends."""

    __slots__ = ("_gen", "_start_event", "_target", "entity", "held", "name")

    def __init__(
        self,
        sim: Simulation,
        generator: ProcessGenerator,
        *,
        name: str | None = None,
        entity: Entity | None = None,
    ) -> None:
        if not isinstance(generator, Generator):
            raise TypeError(
                "sim.process() needs a generator; did you forget to call the function, "
                f"or does it lack a `yield`? got {type(generator).__name__}"
            )
        super().__init__(sim)
        self._gen = generator
        self.name: str = name or str(getattr(generator, "__name__", "process"))
        self.entity = entity
        self.held: list[Request] = []
        self._target: Waitable | None = None
        self._start_event: Event | None = sim._schedule_internal(
            lambda: self._step(None, None), "_start", priority=Priority.NORMAL
        )

    @property
    def is_alive(self) -> bool:
        return not self.triggered

    @property
    def target(self) -> Waitable | None:
        """What the process is currently waiting for."""
        return self._target

    def _resume(self, waited: Waitable) -> None:
        if waited is not self._target:
            return  # stale wake-up (the process was interrupted and moved on)
        self._target = None
        if waited._exc is not None:
            self._step(None, waited._exc)
        else:
            self._step(waited._value, None)

    def _step(self, value: Any, exc: BaseException | None) -> None:
        self._start_event = None
        sim = self.sim
        previous = sim._active_process
        sim._active_process = self
        try:
            while True:
                try:
                    yielded = self._gen.throw(exc) if exc is not None else self._gen.send(value)
                except StopIteration as stop:
                    self._terminate(stop.value, None)
                    return
                except BaseException as err:
                    self._terminate(None, err)
                    return
                target = self._coerce(yielded)
                if target is None:
                    exc = TypeError(
                        f"process {self.name!r} yielded {yielded!r}; yield a number, "
                        "timedelta, or a Waitable (timeout, request, signal, process, ...)"
                    )
                    value = None
                    continue
                if target.triggered:
                    # Already done (e.g. a request granted on the spot): continue at once.
                    exc, value = target._exc, (None if target._exc else target._value)
                    continue
                self._target = target
                target.add_callback(self._resume)
                return
        finally:
            sim._active_process = previous

    def _coerce(self, yielded: Any) -> Waitable | None:
        if isinstance(yielded, Waitable):
            if yielded.sim is not self.sim:
                return None
            return yielded
        if isinstance(yielded, int | float | timedelta) and not isinstance(yielded, bool):
            return Timeout(self.sim, self.sim.clock.duration(yielded))
        return None

    def _terminate(self, value: Any, exc: BaseException | None) -> None:
        sim = self.sim
        sim._process_finished(self)
        if self.held:
            names = sorted({r.resource.name for r in self.held})
            sim.warn(f"process {self.name!r} finished while holding resource(s) {names}")
            sim.metrics.increment("sim.unreleased_resources", len(self.held))
        if exc is None:
            self._value = value
            self._dispatch()
            return
        self._exc = exc
        if self._callbacks:
            self._dispatch()
        else:
            # Nobody is waiting on this process: surface the error instead of swallowing it.
            raise exc

    def interrupt(self, cause: Any = None) -> None:
        """Raise :class:`~simulsi.errors.Interrupt` inside the process at the current time.

        A pending resource request the process was waiting on is withdrawn.
        """
        if not self.is_alive:
            raise EventStateError(f"cannot interrupt finished process {self.name!r}")
        if self is self.sim._active_process:
            raise EventStateError("a process cannot interrupt itself")
        err = Interrupt(cause)
        self.sim._schedule_internal(
            lambda: self._deliver_interrupt(err), "_interrupt", priority=Priority.URGENT
        )

    def _deliver_interrupt(self, err: Interrupt) -> None:
        if not self.is_alive:
            return
        if self._start_event is not None:
            self.sim.cancel(self._start_event)
        target = self._target
        if target is not None:
            target.remove_callback(self._resume)
            self._target = None
            from simulsi.resources.resource import Request

            if (isinstance(target, Request) and not target.triggered) or (
                isinstance(target, Timeout) and not target._callbacks
            ):
                target.cancel()
        self._step(None, err)

    def __repr__(self) -> str:
        return f"<Process {self.name!r} {'alive' if self.is_alive else 'finished'}>"
