"""Deterministic, event-aware ODE simulation built on SciPy.

The simulator splits integration intervals at every discontinuity.  Boluses,
finite and overlapping infusions, resets, time-varying covariates, ADDL doses,
and same-time observations are handled above ``scipy.integrate.solve_ivp`` so
that event semantics do not depend on a solver's root-finding convention.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .events import CanonicalEvent, EventTable, EventType, canonicalize_events


class ODESimulationError(RuntimeError):
    """A structured ODE or event-processing failure."""

    code = "ODE-SIMULATION-FAILED-001"

    def __init__(
        self,
        message: str,
        *,
        time: float | None = None,
        subject_id: Any = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.time = time
        self.subject_id = subject_id
        self.details = MappingProxyType({} if details is None else dict(details))
        context = []
        if subject_id is not None:
            context.append(f"ID={subject_id!r}")
        if time is not None:
            context.append(f"TIME={time:g}")
        suffix = "" if not context else " [" + ", ".join(context) + "]"
        super().__init__(message + suffix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "time": self.time,
            "subject_id": self.subject_id,
            "details": dict(self.details),
        }


class UnsupportedEventSemantics(ODESimulationError):
    """Raised rather than silently approximating unsupported event semantics."""

    code = "ODE-EVENT-UNSUPPORTED-001"


def _readonly_array(value: ArrayLike, *, ndim: int | None = None) -> NDArray[np.float64]:
    array = np.array(value, dtype=float, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array, got shape {array.shape}")
    array.setflags(write=False)
    return array


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ODEContext(Mapping[str, float]):
    """Explicit dynamic inputs passed to three-argument RHS callables.

    The context also implements the read-only mapping protocol by delegating to
    ``parameters``.  Thus both ``context.parameters["CL"]`` and the familiar
    shorthand ``context["CL"]`` are supported without making covariates or
    infusion rates implicit.
    """

    parameters: Mapping[str, float]
    covariates: Mapping[str, Any]
    infusion_rates: NDArray[np.float64]
    subject_id: Any = None

    def __getitem__(self, key: str) -> float:
        return self.parameters[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.parameters)

    def __len__(self) -> int:
        return len(self.parameters)


@dataclass(frozen=True, slots=True)
class EventSnapshot:
    """State observed at a canonical observation event."""

    row_id: str
    source_row_id: str
    subject_id: Any
    time: float
    state: NDArray[np.float64]
    dv: float | None
    mdv: int
    lloq: float | None
    occasion: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _readonly_array(self.state, ndim=1))


@dataclass(frozen=True, slots=True)
class ODESolverMetadata:
    """Numerical and semantic metadata retained for every successful run."""

    solver: str
    scipy_version: str
    rtol: float
    atol: tuple[float, ...]
    max_step: float
    success: bool
    message: str
    nfev: int
    njev: int
    nlu: int
    segments: int
    event_actions: int
    source_events: int
    generated_additional_doses: int
    generated_infusion_stops: int
    same_time_order: tuple[str, ...]
    sensitivity_method: str | None = None
    sensitivity_step: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "scipy_version": self.scipy_version,
            "rtol": self.rtol,
            "atol": list(self.atol),
            "max_step": self.max_step,
            "success": self.success,
            "message": self.message,
            "nfev": self.nfev,
            "njev": self.njev,
            "nlu": self.nlu,
            "segments": self.segments,
            "event_actions": self.event_actions,
            "source_events": self.source_events,
            "generated_additional_doses": self.generated_additional_doses,
            "generated_infusion_stops": self.generated_infusion_stops,
            "same_time_order": list(self.same_time_order),
            "sensitivity_method": self.sensitivity_method,
            "sensitivity_step": self.sensitivity_step,
        }


@dataclass(frozen=True, slots=True)
class ODESimulationResult:
    """State trajectories, event snapshots, sensitivities, and solver metadata."""

    times: NDArray[np.float64]
    states: NDArray[np.float64]
    state_names: tuple[str, ...]
    observations: tuple[EventSnapshot, ...]
    metadata: ODESolverMetadata
    subject_id: Any = None
    sensitivities: NDArray[np.float64] | None = None
    sensitivity_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        times = _readonly_array(self.times, ndim=1)
        states = _readonly_array(self.states, ndim=2)
        if states.shape != (times.size, len(self.state_names)):
            raise ValueError(
                f"states must have shape (len(times), len(state_names)); got {states.shape}"
            )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "sensitivity_parameters", tuple(self.sensitivity_parameters))
        if self.sensitivities is not None:
            sensitivities = _readonly_array(self.sensitivities, ndim=3)
            expected = (times.size, len(self.state_names), len(self.sensitivity_parameters))
            if sensitivities.shape != expected:
                raise ValueError(
                    f"sensitivities must have shape {expected}; got {sensitivities.shape}"
                )
            object.__setattr__(self, "sensitivities", sensitivities)

    def state(self, name_or_index: str | int) -> NDArray[np.float64]:
        """Return one read-only state trajectory."""

        if isinstance(name_or_index, str):
            try:
                index = self.state_names.index(name_or_index)
            except ValueError as exc:
                raise KeyError(name_or_index) from exc
        else:
            index = int(name_or_index)
        result = self.states[:, index]
        result.setflags(write=False)
        return result

    def sensitivity(self, state: str | int, parameter: str) -> NDArray[np.float64]:
        """Return ``d state / d parameter`` from finite differences."""

        if self.sensitivities is None:
            raise KeyError("this simulation did not request sensitivities")
        state_index = self.state_names.index(state) if isinstance(state, str) else int(state)
        try:
            parameter_index = self.sensitivity_parameters.index(parameter)
        except ValueError as exc:
            raise KeyError(parameter) from exc
        result = self.sensitivities[:, state_index, parameter_index]
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class SensitivityCheck:
    """Finite-difference sensitivity diagnostic."""

    parameter_names: tuple[str, ...]
    forward: NDArray[np.float64]
    central: NDArray[np.float64] | None
    maximum_scaled_difference: float | None
    step: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "forward", _readonly_array(self.forward, ndim=3))
        if self.central is not None:
            object.__setattr__(self, "central", _readonly_array(self.central, ndim=3))


@dataclass(frozen=True, slots=True)
class _Action:
    time: float
    priority: int
    source_position: int
    suborder: int
    kind: Literal["reset", "covariate", "infusion_stop", "dose", "infusion_start", "observation"]
    event: CanonicalEvent
    generated: bool = False

    @property
    def key(self) -> tuple[float, int, int, int, str]:
        return (
            self.time,
            self.priority,
            self.source_position,
            self.suborder,
            self.event.row_id,
        )


_SAME_TIME_ORDER = (
    "reset",
    "covariate_change",
    "infusion_stop",
    "dose_or_infusion_start",
    "observation",
)


def _coerce_events(
    events: EventTable | Iterable[Mapping[str, Any]] | None,
    *,
    covariate_columns: Sequence[str],
) -> EventTable:
    if events is None:
        return EventTable(())
    if isinstance(events, EventTable):
        return events
    return canonicalize_events(events, covariate_columns=covariate_columns)


def _actions(table: EventTable) -> tuple[list[_Action], int, int]:
    expanded = table.expand_additional()
    generated_additional = sum(event.generation == "ADDL" for event in expanded.events)
    explicit_stop_keys = Counter(
        (
            event.subject_id,
            event.time,
            event.compartment,
            round(event.effective_rate, 12),
        )
        for event in expanded
        if event.evid == EventType.INFUSION_STOP
    )
    actions: list[_Action] = []
    generated_stops = 0
    for event in expanded:
        if event.steady_state:
            raise UnsupportedEventSemantics(
                "steady-state (SS) initialization is preserved by the event table "
                "but requires a model-specific equilibrium implementation",
                time=event.time,
                subject_id=event.subject_id,
                details={"row_id": event.row_id, "SS": event.steady_state},
            )
        if event.evid in (EventType.RESET, EventType.RESET_AND_DOSE):
            actions.append(_Action(event.time, 0, event.source_position, 0, "reset", event))
        if event.evid == EventType.COVARIATE:
            actions.append(_Action(event.time, 1, event.source_position, 0, "covariate", event))
        if event.evid == EventType.INFUSION_STOP:
            actions.append(_Action(event.time, 2, event.source_position, 0, "infusion_stop", event))
        if event.evid in (EventType.DOSE, EventType.RESET_AND_DOSE):
            dose_time = event.time + event.lag
            kind: Literal["dose", "infusion_start"] = (
                "infusion_start" if event.is_infusion else "dose"
            )
            actions.append(_Action(dose_time, 3, event.source_position, 1, kind, event))
            if event.is_infusion:
                duration = event.infusion_duration
                if duration is None:
                    raise UnsupportedEventSemantics(
                        "an infusion start requires a finite duration",
                        time=event.time,
                        subject_id=event.subject_id,
                        details={"row_id": event.row_id},
                    )
                stop_time = dose_time + duration
                stop_key = (
                    event.subject_id,
                    stop_time,
                    event.compartment,
                    round(event.effective_rate, 12),
                )
                if explicit_stop_keys[stop_key]:
                    explicit_stop_keys[stop_key] -= 1
                else:
                    actions.append(
                        _Action(
                            stop_time,
                            2,
                            event.source_position,
                            0,
                            "infusion_stop",
                            event,
                            generated=True,
                        )
                    )
                    generated_stops += 1
        if event.evid == EventType.OBSERVATION:
            actions.append(_Action(event.time, 4, event.source_position, 0, "observation", event))
        if event.evid == EventType.OTHER and event.covariates:
            # An OTHER record with explicit covariates is a covariate update,
            # while preserving its EVID in provenance.
            actions.append(_Action(event.time, 1, event.source_position, 0, "covariate", event))
    actions.sort(key=lambda action: action.key)
    return actions, generated_additional, generated_stops


def _compartment_index(
    compartment: int | str | None,
    state_names: Sequence[str],
    compartment_map: Mapping[int | str, int | str] | None,
) -> int:
    if compartment_map is not None and compartment in compartment_map:
        mapped = compartment_map[compartment]
        if isinstance(mapped, str):
            try:
                return tuple(state_names).index(mapped)
            except ValueError as exc:
                raise ODESimulationError(
                    f"compartment map targets unknown state {mapped!r}"
                ) from exc
        index = int(mapped)
    elif compartment is None:
        index = 0
    elif isinstance(compartment, str):
        try:
            index = tuple(state_names).index(compartment)
        except ValueError as exc:
            raise ODESimulationError(
                f"named compartment {compartment!r} is not in state_names and has no mapping"
            ) from exc
    else:
        # NONMEM CMT is conventionally one-based.  Explicit zero remains useful
        # for direct Python APIs and maps to the first state.
        numeric = int(compartment)
        index = numeric - 1 if numeric > 0 else numeric
    if index < 0 or index >= len(state_names):
        raise ODESimulationError(
            f"compartment {compartment!r} resolves to out-of-range state index {index}"
        )
    return index


def _rhs_call_style(rhs: Callable[..., ArrayLike]) -> Literal[2, 3, 4]:
    try:
        parameters = tuple(signature(rhs).parameters.values())
    except (TypeError, ValueError):
        return 2
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters):
        return 3
    if len(positional) >= 4:
        return 4
    if len(positional) >= 3:
        return 3
    return 2


def _normalize_tolerances(
    atol: float | ArrayLike, state_count: int
) -> tuple[NDArray[np.float64], tuple[float, ...]]:
    values = np.asarray(atol, dtype=float)
    if values.ndim == 0:
        values = np.full(state_count, float(values))
    if values.shape != (state_count,):
        raise ValueError(f"atol must be scalar or have shape ({state_count},)")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("all absolute tolerances must be finite and positive")
    return values, tuple(float(value) for value in values)


def simulate_ode(
    rhs: Callable[..., ArrayLike],
    initial_state: ArrayLike,
    events: EventTable | Iterable[Mapping[str, Any]] | None = None,
    *,
    t_eval: ArrayLike | None = None,
    parameters: Mapping[str, float] | None = None,
    initial_covariates: Mapping[str, Any] | None = None,
    covariate_columns: Sequence[str] = (),
    state_names: Sequence[str] | None = None,
    compartment_map: Mapping[int | str, int | str] | None = None,
    subject_id: Any = None,
    initial_time: float = 0.0,
    final_time: float | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float | ArrayLike = 1e-10,
    max_step: float = np.inf,
    sensitivity_parameters: Sequence[str] | None = None,
    sensitivity_step: float = np.sqrt(np.finfo(float).eps),
    debug_finite_difference: bool = False,
) -> ODESimulationResult:
    """Simulate one subject with exact event-time discontinuities.

    Supported RHS signatures are ``rhs(t, y)``, ``rhs(t, y, context)`` and
    ``rhs(t, y, parameters, covariates)``.  For the three-argument form,
    ``context`` is an :class:`ODEContext`.  Infusion rates are always added to
    the returned state derivatives by the event manager.

    Forward finite-difference sensitivities can be requested by parameter name.
    Set ``debug_finite_difference=True`` to compute all numeric parameter
    sensitivities when no explicit list is supplied.
    """

    if not callable(rhs):
        raise TypeError("rhs must be callable")
    y0 = np.asarray(initial_state, dtype=float)
    if y0.ndim != 1 or y0.size == 0:
        raise ValueError("initial_state must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(y0)):
        raise ValueError("initial_state must contain only finite values")
    initial_time = float(initial_time)
    if not isfinite(initial_time):
        raise ValueError("initial_time must be finite")
    if not isfinite(float(rtol)) or rtol <= 0:
        raise ValueError("rtol must be finite and positive")
    if not (np.isinf(max_step) or isfinite(float(max_step))) or max_step <= 0:
        raise ValueError("max_step must be positive")
    atol_array, atol_tuple = _normalize_tolerances(atol, y0.size)
    names = (
        tuple(str(name) for name in state_names)
        if state_names is not None
        else tuple(f"state_{index + 1}" for index in range(y0.size))
    )
    if len(names) != y0.size or len(set(names)) != len(names):
        raise ValueError("state_names must be unique and match initial_state length")

    table = _coerce_events(events, covariate_columns=covariate_columns)
    if subject_id is None:
        if len(table.subjects) > 1:
            raise ValueError(
                "simulate_ode handles one subject at a time; select subject_id explicitly"
            )
        selected_subject = table.subjects[0] if table.subjects else None
    else:
        selected_subject = subject_id
        table = table.for_subject(subject_id)
    actions, generated_additional, generated_stops = _actions(table)
    if actions and actions[0].time < initial_time:
        raise ODESimulationError(
            "an event precedes initial_time; supply an earlier initial_time or precompute y0",
            time=actions[0].time,
            subject_id=selected_subject,
        )

    if t_eval is None:
        requested = np.array(
            sorted({action.time for action in actions if action.kind == "observation"}),
            dtype=float,
        )
        if requested.size == 0 and actions:
            requested = np.array(sorted({action.time for action in actions}), dtype=float)
        elif requested.size == 0:
            requested = np.array([initial_time], dtype=float)
    else:
        requested_raw = np.asarray(t_eval, dtype=float)
        if requested_raw.ndim == 0:
            requested_raw = requested_raw.reshape(1)
        if requested_raw.ndim != 1 or np.any(~np.isfinite(requested_raw)):
            raise ValueError("t_eval must be a one-dimensional sequence of finite times")
        if np.any(requested_raw < initial_time):
            raise ValueError("t_eval cannot precede initial_time")
        requested = np.unique(requested_raw)

    inferred_end_candidates = [initial_time]
    if actions:
        inferred_end_candidates.append(max(action.time for action in actions))
    if requested.size:
        inferred_end_candidates.append(float(requested[-1]))
    end_time = max(inferred_end_candidates) if final_time is None else float(final_time)
    if not isfinite(end_time) or end_time < initial_time:
        raise ValueError("final_time must be finite and no earlier than initial_time")
    if requested.size and requested[-1] > end_time:
        raise ValueError("t_eval cannot extend beyond final_time")
    active_actions = [action for action in actions if action.time <= end_time]

    parameter_values = {str(name): float(value) for name, value in (parameters or {}).items()}
    if any(not isfinite(value) for value in parameter_values.values()):
        raise ValueError("parameters must contain finite numeric values")
    parameter_view = _freeze_mapping(parameter_values)
    covariates = dict(initial_covariates or {})
    infusion_rates = np.zeros(y0.size, dtype=float)
    y = y0.copy()
    style = _rhs_call_style(rhs)
    requested_set = set(float(value) for value in requested)
    output_states: dict[float, NDArray[np.float64]] = {}
    observations: list[EventSnapshot] = []
    total_nfev = 0
    total_njev = 0
    total_nlu = 0
    segments = 0
    solver_messages: list[str] = []

    grouped: dict[float, list[_Action]] = {}
    for action in active_actions:
        grouped.setdefault(action.time, []).append(action)
    breakpoints = sorted(
        {
            *grouped.keys(),
            *(float(value) for value in requested),
            end_time,
        }
    )
    breakpoints = [value for value in breakpoints if value >= initial_time]

    def integrated_rhs(t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        rates_snapshot = np.array(infusion_rates, copy=True)
        rates_snapshot.setflags(write=False)
        context = ODEContext(
            parameter_view,
            _freeze_mapping(covariates),
            rates_snapshot,
            selected_subject,
        )
        try:
            if style == 4:
                raw = rhs(t, state, parameter_view, context.covariates)
            elif style == 3:
                raw = rhs(t, state, context)
            else:
                raw = rhs(t, state)
            derivative_value = np.asarray(raw, dtype=float)
        except Exception as exc:
            if isinstance(exc, ODESimulationError):
                raise
            raise ODESimulationError(
                f"RHS evaluation raised {type(exc).__name__}: {exc}",
                time=t,
                subject_id=selected_subject,
            ) from exc
        if derivative_value.shape != state.shape:
            raise ODESimulationError(
                f"RHS returned shape {derivative_value.shape}; expected {state.shape}",
                time=t,
                subject_id=selected_subject,
            )
        derivative_value = derivative_value + infusion_rates
        if np.any(~np.isfinite(derivative_value)):
            raise ODESimulationError(
                "RHS returned non-finite derivatives",
                time=t,
                subject_id=selected_subject,
            )
        return derivative_value

    current_time = initial_time
    for breakpoint in breakpoints:
        if breakpoint > current_time:
            try:
                solution = solve_ivp(
                    integrated_rhs,
                    (current_time, breakpoint),
                    y,
                    method=method,
                    rtol=rtol,
                    atol=atol_array,
                    max_step=max_step,
                )
            except ODESimulationError:
                raise
            except Exception as exc:
                raise ODESimulationError(
                    f"solver raised {type(exc).__name__}: {exc}",
                    time=current_time,
                    subject_id=selected_subject,
                    details={"segment_end": breakpoint, "method": method},
                ) from exc
            total_nfev += int(solution.nfev)
            total_njev += int(getattr(solution, "njev", 0))
            total_nlu += int(getattr(solution, "nlu", 0))
            segments += 1
            solver_messages.append(str(solution.message))
            if not solution.success:
                raise ODESimulationError(
                    f"solve_ivp failed: {solution.message}",
                    time=float(solution.t[-1]) if solution.t.size else current_time,
                    subject_id=selected_subject,
                    details={
                        "segment_start": current_time,
                        "segment_end": breakpoint,
                        "method": method,
                        "nfev": int(solution.nfev),
                    },
                )
            y = np.asarray(solution.y[:, -1], dtype=float)
            current_time = breakpoint

        for action in grouped.get(breakpoint, ()):
            event = action.event
            if action.kind == "reset":
                y[:] = 0.0
                infusion_rates[:] = 0.0
            elif action.kind == "covariate":
                covariates.update(event.covariates)
            elif action.kind == "infusion_stop":
                index = _compartment_index(event.compartment, names, compartment_map)
                rate = event.effective_rate
                infusion_rates[index] -= rate
                tolerance = max(1e-12, abs(rate) * 1e-10)
                if infusion_rates[index] < -tolerance:
                    raise ODESimulationError(
                        "infusion stop exceeds the active rate in its compartment",
                        time=breakpoint,
                        subject_id=selected_subject,
                        details={
                            "row_id": event.row_id,
                            "compartment_index": index,
                            "active_rate_after_stop": infusion_rates[index],
                        },
                    )
                if abs(infusion_rates[index]) <= tolerance:
                    infusion_rates[index] = 0.0
            elif action.kind == "dose":
                index = _compartment_index(event.compartment, names, compartment_map)
                amount = event.effective_amount
                if amount is None:
                    raise ODESimulationError(
                        "bolus dose amount is missing",
                        time=breakpoint,
                        subject_id=selected_subject,
                        details={"row_id": event.row_id},
                    )
                y[index] += amount
            elif action.kind == "infusion_start":
                index = _compartment_index(event.compartment, names, compartment_map)
                infusion_rates[index] += event.effective_rate
            elif action.kind == "observation":
                observations.append(
                    EventSnapshot(
                        row_id=event.row_id,
                        source_row_id=event.source_row_id,
                        subject_id=event.subject_id,
                        time=breakpoint,
                        state=y,
                        dv=event.dv,
                        mdv=event.mdv,
                        lloq=event.lloq,
                        occasion=event.occasion,
                    )
                )
        if breakpoint in requested_set:
            output_states[breakpoint] = y.copy()

    output_times = requested
    if output_times.size:
        trajectory = np.vstack([output_states[float(time)] for time in output_times])
    else:
        trajectory = np.empty((0, y0.size), dtype=float)

    selected_sensitivity_parameters: tuple[str, ...]
    if sensitivity_parameters is not None:
        selected_sensitivity_parameters = tuple(str(name) for name in sensitivity_parameters)
    elif debug_finite_difference:
        selected_sensitivity_parameters = tuple(parameter_values)
    else:
        selected_sensitivity_parameters = ()
    if len(set(selected_sensitivity_parameters)) != len(selected_sensitivity_parameters):
        raise ValueError("sensitivity_parameters must be unique")
    for name in selected_sensitivity_parameters:
        if name not in parameter_values:
            raise KeyError(f"unknown sensitivity parameter {name!r}")
    if not isfinite(float(sensitivity_step)) or sensitivity_step <= 0:
        raise ValueError("sensitivity_step must be finite and positive")

    metadata = ODESolverMetadata(
        solver=method,
        scipy_version=scipy.__version__,
        rtol=float(rtol),
        atol=atol_tuple,
        max_step=float(max_step),
        success=True,
        message="; ".join(dict.fromkeys(solver_messages)) or "no integration required",
        nfev=total_nfev,
        njev=total_njev,
        nlu=total_nlu,
        segments=segments,
        event_actions=len(active_actions),
        source_events=table.source_count,
        generated_additional_doses=generated_additional,
        generated_infusion_stops=generated_stops,
        same_time_order=_SAME_TIME_ORDER,
        sensitivity_method=(
            "forward-finite-difference" if selected_sensitivity_parameters else None
        ),
        sensitivity_step=(float(sensitivity_step) if selected_sensitivity_parameters else None),
    )
    result = ODESimulationResult(
        output_times,
        trajectory,
        names,
        tuple(observations),
        metadata,
        selected_subject,
    )
    if not selected_sensitivity_parameters:
        return result

    sensitivities = np.empty(
        (output_times.size, y0.size, len(selected_sensitivity_parameters)), dtype=float
    )
    for parameter_index, parameter_name in enumerate(selected_sensitivity_parameters):
        value = parameter_values[parameter_name]
        increment = float(sensitivity_step) * max(1.0, abs(value))
        perturbed = dict(parameter_values)
        perturbed[parameter_name] = value + increment
        perturbed_result = simulate_ode(
            rhs,
            y0,
            table,
            t_eval=output_times,
            parameters=perturbed,
            initial_covariates=initial_covariates,
            covariate_columns=covariate_columns,
            state_names=names,
            compartment_map=compartment_map,
            subject_id=selected_subject,
            initial_time=initial_time,
            final_time=end_time,
            method=method,
            rtol=rtol,
            atol=atol_array,
            max_step=max_step,
        )
        sensitivities[:, :, parameter_index] = (perturbed_result.states - result.states) / increment
    return ODESimulationResult(
        result.times,
        result.states,
        result.state_names,
        result.observations,
        metadata,
        result.subject_id,
        sensitivities,
        selected_sensitivity_parameters,
    )


def finite_difference_sensitivities(
    rhs: Callable[..., ArrayLike],
    initial_state: ArrayLike,
    events: EventTable | Iterable[Mapping[str, Any]] | None,
    *,
    parameters: Mapping[str, float],
    parameter_names: Sequence[str] | None = None,
    step: float = np.cbrt(np.finfo(float).eps),
    compare_central: bool = False,
    **simulation_options: Any,
) -> SensitivityCheck:
    """Compute forward sensitivities and optionally compare central differences.

    This is a validation/debug path, not an automatic-differentiation claim.
    Event times are held fixed while parameter values are perturbed.
    """

    names = tuple(parameters) if parameter_names is None else tuple(parameter_names)
    base = simulate_ode(
        rhs,
        initial_state,
        events,
        parameters=parameters,
        sensitivity_parameters=names,
        sensitivity_step=step,
        **simulation_options,
    )
    assert base.sensitivities is not None
    central: NDArray[np.float64] | None = None
    maximum_scaled_difference: float | None = None
    if compare_central:
        central = np.empty_like(base.sensitivities)
        for index, name in enumerate(names):
            if name not in parameters:
                raise KeyError(name)
            value = float(parameters[name])
            increment = step * max(1.0, abs(value))
            plus = dict(parameters)
            minus = dict(parameters)
            plus[name] = value + increment
            minus[name] = value - increment
            plus_result = simulate_ode(
                rhs,
                initial_state,
                events,
                parameters=plus,
                **simulation_options,
            )
            minus_result = simulate_ode(
                rhs,
                initial_state,
                events,
                parameters=minus,
                **simulation_options,
            )
            central[:, :, index] = (plus_result.states - minus_result.states) / (2.0 * increment)
        scale = np.maximum(1.0, np.abs(central))
        maximum_scaled_difference = float(np.max(np.abs(base.sensitivities - central) / scale))
    return SensitivityCheck(
        names,
        base.sensitivities,
        central,
        maximum_scaled_difference,
        float(step),
    )


def simulate_subjects(
    rhs: Callable[..., ArrayLike],
    initial_state: ArrayLike,
    events: EventTable | Iterable[Mapping[str, Any]],
    **options: Any,
) -> Mapping[Any, ODESimulationResult]:
    """Deterministically simulate every subject in an event table."""

    table = events if isinstance(events, EventTable) else canonicalize_events(events)
    return MappingProxyType(
        {
            subject: simulate_ode(rhs, initial_state, table, subject_id=subject, **options)
            for subject in table.subjects
        }
    )


__all__ = [
    "EventSnapshot",
    "ODEContext",
    "ODESimulationError",
    "ODESimulationResult",
    "ODESolverMetadata",
    "SensitivityCheck",
    "UnsupportedEventSemantics",
    "finite_difference_sensitivities",
    "simulate_ode",
    "simulate_subjects",
]
