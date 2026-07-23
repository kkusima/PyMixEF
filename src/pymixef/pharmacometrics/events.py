"""Canonical pharmacometric event records.

The event layer is deliberately independent of pandas.  Input records are
normalized to immutable :class:`CanonicalEvent` objects, sorted with a
documented tie-breaking rule, and accompanied by an immutable audit trail.

Same-time events are applied in this order:

1. reset (including the reset part of reset-and-dose);
2. covariate change;
3. infusion stop;
4. dose or infusion start (including the dose part of reset-and-dose);
5. observation;
6. other event.

Input order is the final stable tie breaker.  Consequently, an observation at
the same time as a dose sees the post-dose state.  This convention is explicit
instead of being inherited from a dataframe sort implementation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import IntEnum, StrEnum
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any

import numpy as np


class EventValidationError(ValueError):
    """Raised when an event record is ambiguous or internally inconsistent."""

    code = "EVENT-INVALID-001"

    def __init__(self, message: str, *, row: int | str | None = None) -> None:
        self.row = row
        suffix = "" if row is None else f" [row={row!r}]"
        super().__init__(message + suffix)


class EventType(IntEnum):
    """Canonical EVID values, extending the common NONMEM values.

    Values 0--4 retain their familiar meaning.  Values 5 and 6 are explicit
    PyMixEF records used for time-varying covariates and infusion stops.
    """

    OBSERVATION = 0
    DOSE = 1
    OTHER = 2
    RESET = 3
    RESET_AND_DOSE = 4
    COVARIATE = 5
    INFUSION_STOP = 6


class DoseAmountStatus(StrEnum):
    """Semantic state of the dose-amount field."""

    RECORDED = "recorded"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not-applicable"


_EVENT_NAMES: dict[str, EventType] = {
    "observation": EventType.OBSERVATION,
    "observe": EventType.OBSERVATION,
    "obs": EventType.OBSERVATION,
    "dose": EventType.DOSE,
    "bolus": EventType.DOSE,
    "infusion": EventType.DOSE,
    "infusion_start": EventType.DOSE,
    "other": EventType.OTHER,
    "reset": EventType.RESET,
    "reset_and_dose": EventType.RESET_AND_DOSE,
    "reset-dose": EventType.RESET_AND_DOSE,
    "resetdose": EventType.RESET_AND_DOSE,
    "covariate": EventType.COVARIATE,
    "covariate_change": EventType.COVARIATE,
    "infusion_stop": EventType.INFUSION_STOP,
}

# Reset-and-dose has the priority of reset.  The simulator performs its reset
# before adding the dose represented by the same record.
EVENT_PRIORITY: Mapping[EventType, int] = MappingProxyType(
    {
        EventType.RESET: 0,
        EventType.RESET_AND_DOSE: 0,
        EventType.COVARIATE: 1,
        EventType.INFUSION_STOP: 2,
        EventType.DOSE: 3,
        EventType.OBSERVATION: 4,
        EventType.OTHER: 5,
    }
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not value:
        return MappingProxyType({})
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _json_value(value: Any) -> Any:
    """Return a plain serialization-friendly representation."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, frozenset):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, EventType):
        return int(value)
    return value


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One machine-readable canonicalization or expansion action."""

    code: str
    action: str
    row_id: str | None = None
    source_row_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit entry without exposing mutable internals."""

        return {
            "code": self.code,
            "action": self.action,
            "row_id": self.row_id,
            "source_row_id": self.source_row_id,
            "details": _json_value(self.details),
        }


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """An immutable event in the PyMixEF canonical schema.

    ``amount_status`` distinguishes a recorded amount (including zero), an
    explicitly unknown dose amount, and a field that is structurally not
    applicable to a non-dose event.
    """

    subject_id: Hashable
    time: float
    evid: EventType
    amount: float | None = None
    amount_status: DoseAmountStatus | str | None = None
    rate: float | None = None
    duration: float | None = None
    compartment: int | str | None = None
    additional: int = 0
    interval: float | None = None
    steady_state: int = 0
    mdv: int = 1
    dv: float | None = None
    lloq: float | None = None
    occasion: Hashable | None = None
    bioavailability: float = 1.0
    lag: float = 0.0
    covariates: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    extras: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    row_id: str = ""
    source_row_id: str = ""
    source_position: int = 0
    generated: bool = False
    generation: str | None = None

    def __post_init__(self) -> None:
        event_type = EventType(self.evid)
        if self.amount_status is None:
            amount_status = (
                DoseAmountStatus.RECORDED
                if self.amount is not None
                else (
                    DoseAmountStatus.UNKNOWN
                    if event_type in (EventType.DOSE, EventType.RESET_AND_DOSE)
                    else DoseAmountStatus.NOT_APPLICABLE
                )
            )
        else:
            try:
                amount_status = DoseAmountStatus(str(self.amount_status))
            except ValueError as error:
                raise EventValidationError(
                    "AMT_STATUS must be recorded, unknown, or not-applicable",
                    row=self.row_id,
                ) from error
        if amount_status == DoseAmountStatus.RECORDED and self.amount is None:
            raise EventValidationError(
                "AMT_STATUS=recorded requires AMT, including AMT=0 for zero dose",
                row=self.row_id,
            )
        if amount_status != DoseAmountStatus.RECORDED and self.amount is not None:
            raise EventValidationError(
                "AMT is present but AMT_STATUS is not recorded", row=self.row_id
            )
        if (
            event_type in (EventType.DOSE, EventType.RESET_AND_DOSE)
            and amount_status == DoseAmountStatus.NOT_APPLICABLE
        ):
            raise EventValidationError(
                "Dose AMT may be recorded or explicitly unknown, not not-applicable",
                row=self.row_id,
            )
        if not isinstance(self.subject_id, Hashable):
            raise EventValidationError("ID must be hashable", row=self.source_position)
        if self.compartment is not None and not isinstance(
            self.compartment, (int, str, np.integer)
        ):
            raise EventValidationError("CMT must be an integer, string, or missing")
        if not isfinite(float(self.time)):
            raise EventValidationError("TIME must be finite", row=self.source_position)
        if self.additional < 0:
            raise EventValidationError("ADDL must be a non-negative integer", row=self.row_id)
        if self.additional and (self.interval is None or self.interval <= 0):
            raise EventValidationError("ADDL > 0 requires II > 0", row=self.row_id)
        if self.interval is not None and (not isfinite(self.interval) or self.interval <= 0):
            raise EventValidationError("II must be finite and positive", row=self.row_id)
        if self.amount is not None and not isfinite(self.amount):
            raise EventValidationError("AMT must be finite when present", row=self.row_id)
        if self.rate is not None and (not isfinite(self.rate) or self.rate < 0):
            raise EventValidationError("RATE must be finite and non-negative", row=self.row_id)
        if self.duration is not None and (not isfinite(self.duration) or self.duration <= 0):
            raise EventValidationError("DUR must be finite and positive", row=self.row_id)
        if self.bioavailability < 0 or not isfinite(self.bioavailability):
            raise EventValidationError("bioavailability must be finite and non-negative")
        if self.lag < 0 or not isfinite(self.lag):
            raise EventValidationError("lag must be finite and non-negative")
        if self.mdv not in (0, 1):
            raise EventValidationError("MDV must be 0 or 1", row=self.row_id)
        if self.steady_state not in (0, 1, 2, 3):
            raise EventValidationError("SS must be one of 0, 1, 2, or 3", row=self.row_id)
        object.__setattr__(self, "evid", event_type)
        object.__setattr__(self, "amount_status", amount_status)
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "covariates", _freeze_mapping(self.covariates))
        object.__setattr__(self, "extras", _freeze_mapping(self.extras))

    @property
    def is_observation(self) -> bool:
        return self.evid == EventType.OBSERVATION

    @property
    def is_dose(self) -> bool:
        return self.evid in (EventType.DOSE, EventType.RESET_AND_DOSE)

    @property
    def is_reset(self) -> bool:
        return self.evid in (EventType.RESET, EventType.RESET_AND_DOSE)

    @property
    def is_infusion(self) -> bool:
        return self.is_dose and self.effective_rate > 0

    @property
    def effective_rate(self) -> float:
        """Infusion rate after bioavailability, or zero for a bolus."""

        if self.rate is not None and self.rate > 0:
            return self.rate * self.bioavailability
        if self.duration is not None and self.amount is not None:
            return (self.amount / self.duration) * self.bioavailability
        return 0.0

    @property
    def effective_amount(self) -> float | None:
        return None if self.amount is None else self.amount * self.bioavailability

    @property
    def infusion_duration(self) -> float | None:
        if not self.is_infusion:
            return None
        if self.duration is not None:
            return self.duration
        if self.amount is not None and self.rate:
            return self.amount / self.rate
        return None

    @property
    def kind(self) -> str:
        if self.evid == EventType.DOSE and self.is_infusion:
            return "infusion_start"
        return {
            EventType.OBSERVATION: "observation",
            EventType.DOSE: "dose",
            EventType.OTHER: "other",
            EventType.RESET: "reset",
            EventType.RESET_AND_DOSE: "reset_and_dose",
            EventType.COVARIATE: "covariate",
            EventType.INFUSION_STOP: "infusion_stop",
        }[self.evid]

    id = property(lambda self: self.subject_id)
    event_type = property(lambda self: self.kind)
    amt = property(lambda self: self.amount)
    cmt = property(lambda self: self.compartment)
    addl = property(lambda self: self.additional)
    ii = property(lambda self: self.interval)
    ss = property(lambda self: self.steady_state)

    # Uppercase aliases make the canonical schema discoverable to users coming
    # from NONMEM-style data without making Python attribute names awkward.
    ID = property(lambda self: self.subject_id)
    TIME = property(lambda self: self.time)
    EVID = property(lambda self: int(self.evid))
    AMT = property(lambda self: self.amount)
    RATE = property(lambda self: self.rate)
    DUR = property(lambda self: self.duration)
    CMT = property(lambda self: self.compartment)
    ADDL = property(lambda self: self.additional)
    II = property(lambda self: self.interval)
    SS = property(lambda self: self.steady_state)
    MDV = property(lambda self: self.mdv)
    DV = property(lambda self: self.dv)
    LLOQ = property(lambda self: self.lloq)

    def to_record(self) -> dict[str, Any]:
        """Return a mutable, uppercase-keyed record for serialization."""

        result: dict[str, Any] = {
            "ID": self.subject_id,
            "TIME": self.time,
            "EVID": int(self.evid),
            "AMT": self.amount,
            "AMT_STATUS": self.amount_status.value,
            "RATE": self.rate,
            "DUR": self.duration,
            "CMT": self.compartment,
            "ADDL": self.additional,
            "II": self.interval,
            "SS": self.steady_state,
            "MDV": self.mdv,
            "DV": self.dv,
            "LLOQ": self.lloq,
            "OCCASION": self.occasion,
            "F": self.bioavailability,
            "ALAG": self.lag,
            "ROW_ID": self.row_id,
            "SOURCE_ROW_ID": self.source_row_id,
            "GENERATED": self.generated,
            "GENERATION": self.generation,
        }
        result.update(_json_value(self.covariates))
        result.update(_json_value(self.extras))
        return result


_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("ID", "id", "subject_id", "subject", "SUBJ"),
    "time": ("TIME", "time"),
    "evid": ("EVID", "evid", "event", "event_type"),
    "amount": ("AMT", "amt", "amount"),
    "amount_status": ("AMT_STATUS", "amt_status", "amount_status"),
    "rate": ("RATE", "rate"),
    "duration": ("DUR", "dur", "duration"),
    "compartment": ("CMT", "cmt", "compartment"),
    "additional": ("ADDL", "addl", "additional"),
    "interval": ("II", "ii", "interval"),
    "steady_state": ("SS", "ss", "steady_state"),
    "mdv": ("MDV", "mdv"),
    "dv": ("DV", "dv", "observation"),
    "lloq": ("LLOQ", "lloq"),
    "occasion": ("OCCASION", "occasion", "OCC", "occ"),
    "bioavailability": ("F", "f", "bioavailability"),
    "lag": ("ALAG", "alag", "lag"),
    "row_id": ("ROW_ID", "row_id"),
    "source_row_id": ("SOURCE_ROW_ID", "source_row_id"),
    "covariates": ("COVARIATES", "covariates"),
}
_KNOWN_KEYS = {alias for aliases in _ALIASES.values() for alias in aliases}


def _first(record: Mapping[str, Any], name: str, default: Any = None) -> Any:
    for alias in _ALIASES[name]:
        if alias in record:
            value = record[alias]
            if _is_missing(value):
                return default
            return value
    return default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _number(value: Any, *, name: str, row: int, integer: bool = False) -> float | int | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EventValidationError(f"{name} must be numeric", row=row) from exc
    if not isfinite(number):
        raise EventValidationError(f"{name} must be finite", row=row)
    if integer:
        rounded = int(number)
        if number != rounded:
            raise EventValidationError(f"{name} must be an integer", row=row)
        return rounded
    return number


def _event_type(value: Any, *, row: int) -> EventType:
    if value is None:
        return EventType.OBSERVATION
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "_")
        if key in _EVENT_NAMES:
            return _EVENT_NAMES[key]
    try:
        return EventType(int(value))
    except (TypeError, ValueError) as exc:
        raise EventValidationError(f"unsupported EVID/event type {value!r}", row=row) from exc


def _time(value: Any, *, row: int) -> tuple[float, bool]:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise EventValidationError("calendar TIME values must be timezone-aware", row=row)
        return float(value.timestamp()), True
    converted = _number(value, name="TIME", row=row)
    if converted is None:
        raise EventValidationError("TIME is required", row=row)
    return float(converted), False


def _normalize_record(
    record: Mapping[str, Any],
    position: int,
    *,
    covariate_columns: Sequence[str],
) -> tuple[CanonicalEvent, tuple[AuditEntry, ...]]:
    subject_id = _first(record, "id")
    if _is_missing(subject_id):
        raise EventValidationError("ID is required", row=position)
    canonical_time, converted_calendar = _time(_first(record, "time"), row=position)
    evid = _event_type(_first(record, "evid"), row=position)
    amount = _number(_first(record, "amount"), name="AMT", row=position)
    raw_amount_status = _first(record, "amount_status")
    if raw_amount_status is None:
        amount_status = (
            DoseAmountStatus.RECORDED
            if amount is not None
            else (
                DoseAmountStatus.UNKNOWN
                if evid in (EventType.DOSE, EventType.RESET_AND_DOSE)
                else DoseAmountStatus.NOT_APPLICABLE
            )
        )
    else:
        try:
            amount_status = DoseAmountStatus(
                str(raw_amount_status).strip().lower().replace("_", "-")
            )
        except ValueError as error:
            raise EventValidationError(
                "AMT_STATUS must be recorded, unknown, or not-applicable",
                row=position,
            ) from error
    rate = _number(_first(record, "rate"), name="RATE", row=position)
    duration = _number(_first(record, "duration"), name="DUR", row=position)
    additional = _number(_first(record, "additional", 0), name="ADDL", row=position, integer=True)
    interval = _number(_first(record, "interval"), name="II", row=position)
    steady_state = _number(_first(record, "steady_state", 0), name="SS", row=position, integer=True)
    dv = _number(_first(record, "dv"), name="DV", row=position)
    lloq = _number(_first(record, "lloq"), name="LLOQ", row=position)
    supplied_mdv = _first(record, "mdv")
    mdv = (
        int(_number(supplied_mdv, name="MDV", row=position, integer=True))
        if supplied_mdv is not None
        else int(not (evid == EventType.OBSERVATION and dv is not None))
    )
    bioavailability = _number(_first(record, "bioavailability", 1.0), name="F", row=position)
    lag = _number(_first(record, "lag", 0.0), name="ALAG", row=position)

    if (
        evid in (EventType.DOSE, EventType.RESET_AND_DOSE)
        and amount is None
        and raw_amount_status is None
    ):
        raise EventValidationError(
            "dose records without AMT require AMT_STATUS='unknown'; use AMT=0 "
            "for a recorded zero dose",
            row=position,
        )
    if evid == EventType.INFUSION_STOP and (rate is None or rate <= 0):
        raise EventValidationError("an explicit infusion stop requires RATE > 0", row=position)
    if rate is not None and rate < 0:
        raise EventValidationError(
            "negative or modeled RATE values are not silently interpreted", row=position
        )
    if duration is not None and duration <= 0:
        raise EventValidationError("DUR must be positive", row=position)
    if rate == 0:
        rate = None
    if duration is not None and rate is None:
        if amount is None:
            raise EventValidationError("DUR without RATE requires AMT", row=position)
        rate = amount / duration
    elif rate is not None and rate > 0 and duration is None and amount is not None:
        duration = amount / rate
    elif rate is not None and duration is not None and amount is not None:
        if not isclose(amount, rate * duration, rel_tol=1e-9, abs_tol=1e-12):
            raise EventValidationError(
                "AMT, RATE, and DUR are inconsistent (expected AMT = RATE * DUR)",
                row=position,
            )

    raw_row_id = _first(record, "row_id")
    row_id = str(raw_row_id) if raw_row_id is not None else f"row-{position:08d}"
    raw_source = _first(record, "source_row_id")
    source_row_id = str(raw_source) if raw_source is not None else row_id

    explicit_covariates = _first(record, "covariates", {})
    if explicit_covariates is None:
        explicit_covariates = {}
    if not isinstance(explicit_covariates, Mapping):
        raise EventValidationError("COVARIATES must be a mapping", row=position)
    covariates = dict(explicit_covariates)
    for column in covariate_columns:
        if column in record and not _is_missing(record[column]):
            covariates[column] = record[column]
    extras = {
        str(key): value
        for key, value in record.items()
        if key not in _KNOWN_KEYS and key not in covariate_columns
    }
    event = CanonicalEvent(
        subject_id=subject_id,
        time=canonical_time,
        evid=evid,
        amount=None if amount is None else float(amount),
        amount_status=amount_status,
        rate=None if rate is None else float(rate),
        duration=None if duration is None else float(duration),
        compartment=_first(record, "compartment"),
        additional=int(additional or 0),
        interval=None if interval is None else float(interval),
        steady_state=int(steady_state or 0),
        mdv=mdv,
        dv=None if dv is None else float(dv),
        lloq=None if lloq is None else float(lloq),
        occasion=_first(record, "occasion"),
        bioavailability=float(bioavailability if bioavailability is not None else 1.0),
        lag=float(lag if lag is not None else 0.0),
        covariates=covariates,
        extras=extras,
        row_id=row_id,
        source_row_id=source_row_id,
        source_position=position,
    )
    audit: list[AuditEntry] = [
        AuditEntry(
            code="EVENT-CANONICALIZED-001",
            action="canonicalized",
            row_id=row_id,
            source_row_id=source_row_id,
            details={"source_position": position},
        )
    ]
    if converted_calendar:
        audit.append(
            AuditEntry(
                code="EVENT-TIME-CONVERTED-001",
                action="calendar_time_to_utc_unix_seconds",
                row_id=row_id,
                source_row_id=source_row_id,
                details={"time": canonical_time},
            )
        )
    if duration is not None and _first(record, "rate") is None:
        audit.append(
            AuditEntry(
                code="EVENT-RATE-DERIVED-001",
                action="derived_rate_from_amount_and_duration",
                row_id=row_id,
                source_row_id=source_row_id,
                details={"rate": rate},
            )
        )
    if duration is not None and _first(record, "duration") is None:
        audit.append(
            AuditEntry(
                code="EVENT-DURATION-DERIVED-001",
                action="derived_duration_from_amount_and_rate",
                row_id=row_id,
                source_row_id=source_row_id,
                details={"duration": duration},
            )
        )
    return event, tuple(audit)


def _subject_key(value: Hashable) -> tuple[str, str]:
    """Provide deterministic ordering even for heterogeneous subject IDs."""

    return type(value).__qualname__, repr(value)


def _sort_key(event: CanonicalEvent) -> tuple[Any, ...]:
    return (
        _subject_key(event.subject_id),
        event.time,
        EVENT_PRIORITY[event.evid],
        event.source_position,
        event.row_id,
    )


@dataclass(frozen=True, slots=True)
class EventTable(Sequence[CanonicalEvent]):
    """Immutable, deterministically ordered collection of canonical events."""

    events: tuple[CanonicalEvent, ...]
    audit: tuple[AuditEntry, ...] = ()
    source_count: int = 0
    source_records: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted(tuple(self.events), key=_sort_key))
        ids = [event.row_id for event in normalized]
        if len(ids) != len(set(ids)):
            duplicate = next(row_id for row_id in ids if ids.count(row_id) > 1)
            raise EventValidationError(f"ROW_ID values must be unique; duplicate {duplicate!r}")
        object.__setattr__(self, "events", normalized)
        object.__setattr__(self, "audit", tuple(self.audit))
        object.__setattr__(
            self,
            "source_records",
            tuple(_freeze_mapping(record) for record in self.source_records),
        )
        if self.source_count == 0 and normalized:
            object.__setattr__(
                self, "source_count", len({event.source_row_id for event in normalized})
            )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]] | np.ndarray,
        *,
        covariate_columns: Sequence[str] = (),
        expand_additional: bool = False,
        expand_infusions: bool = False,
    ) -> EventTable:
        """Canonicalize mapping records or a NumPy structured array.

        Calendar times are converted to UTC Unix seconds and must be
        timezone-aware.  No input object is modified.
        """

        if isinstance(records, np.ndarray) and records.dtype.names:
            iterable: Iterable[Mapping[str, Any]] = (
                {name: row[name] for name in records.dtype.names} for row in records
            )
        else:
            iterable = records
        events: list[CanonicalEvent] = []
        audit: list[AuditEntry] = []
        source_records: list[Mapping[str, Any]] = []
        for position, record in enumerate(iterable):
            if not isinstance(record, Mapping):
                raise TypeError("each event record must be a mapping")
            source_records.append(_freeze_mapping(record))
            event, event_audit = _normalize_record(
                record, position, covariate_columns=covariate_columns
            )
            events.append(event)
            audit.extend(event_audit)
        table = cls(
            tuple(events),
            tuple(audit),
            source_count=len(events),
            source_records=tuple(source_records),
        )
        if expand_additional:
            table = table.expand_additional()
        if expand_infusions:
            table = table.expand_infusions()
        return table

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[CanonicalEvent]:
        return iter(self.events)

    def __getitem__(self, index: int | slice) -> CanonicalEvent | tuple[CanonicalEvent, ...]:
        return self.events[index]

    @property
    def subjects(self) -> tuple[Hashable, ...]:
        """Subject IDs in deterministic first-occurrence order."""

        return tuple(dict.fromkeys(event.subject_id for event in self.events))

    def for_subject(self, subject_id: Hashable) -> EventTable:
        """Return an immutable view containing one subject."""

        selected = tuple(event for event in self.events if event.subject_id == subject_id)
        return EventTable(selected, self.audit, self.source_count, self.source_records)

    def to_records(self) -> list[dict[str, Any]]:
        """Return plain records suitable for pandas, Polars, Arrow, or JSON."""

        return [event.to_record() for event in self.events]

    def to_source_records(self) -> list[dict[str, Any]]:
        """Return fresh copies of the retained, pre-canonicalization records."""

        return [
            {str(key): _json_value(value) for key, value in record.items()}
            for record in self.source_records
        ]

    def expand_additional(self) -> EventTable:
        """Materialize ADDL/II doses without mutating the source table.

        The returned records have ``ADDL=0`` so expansion is idempotent.  Every
        generated row retains the original ``source_row_id`` and receives a
        deterministic ``<row_id>:addl:<n>`` row identifier.
        """

        expanded: list[CanonicalEvent] = []
        audit = list(self.audit)
        for event in self.events:
            if event.additional == 0:
                expanded.append(event)
                continue
            if not event.is_dose:
                raise EventValidationError("ADDL is only valid on dose records", row=event.row_id)
            assert event.interval is not None  # validated by CanonicalEvent
            expanded.append(replace(event, additional=0))
            for number in range(1, event.additional + 1):
                generated = replace(
                    event,
                    time=event.time + number * event.interval,
                    additional=0,
                    row_id=f"{event.row_id}:addl:{number}",
                    generated=True,
                    generation="ADDL",
                    source_position=event.source_position,
                )
                expanded.append(generated)
                audit.append(
                    AuditEntry(
                        code="EVENT-ADDL-EXPANDED-001",
                        action="generated_additional_dose",
                        row_id=generated.row_id,
                        source_row_id=event.source_row_id,
                        details={
                            "dose_number": number,
                            "time": generated.time,
                            "interval": event.interval,
                        },
                    )
                )
        return EventTable(tuple(expanded), tuple(audit), self.source_count, self.source_records)

    def expand_infusions(self) -> EventTable:
        """Add explicit infusion-stop records for finite infusions.

        Explicit stops already present in the source are retained.  A finite
        start generated from ``AMT/RATE/DUR`` gets one deterministic stop.
        """

        expanded: list[CanonicalEvent] = list(self.events)
        audit = list(self.audit)
        explicit_stops = Counter(
            (
                event.subject_id,
                event.time,
                event.compartment,
                round(event.effective_rate, 12),
            )
            for event in self.events
            if event.evid == EventType.INFUSION_STOP
        )
        for event in self.events:
            if not event.is_infusion:
                continue
            duration = event.infusion_duration
            if duration is None:
                raise EventValidationError(
                    "finite infusion requires DUR or both AMT and RATE", row=event.row_id
                )
            row_id = f"{event.row_id}:infusion-stop"
            stop_key = (
                event.subject_id,
                event.time + event.lag + duration,
                event.compartment,
                round(event.effective_rate, 12),
            )
            if explicit_stops[stop_key]:
                explicit_stops[stop_key] -= 1
                continue
            stop = CanonicalEvent(
                subject_id=event.subject_id,
                time=event.time + event.lag + duration,
                evid=EventType.INFUSION_STOP,
                rate=event.effective_rate,
                compartment=event.compartment,
                occasion=event.occasion,
                covariates={},
                row_id=row_id,
                source_row_id=event.source_row_id,
                source_position=event.source_position,
                generated=True,
                generation="INFUSION_STOP",
            )
            expanded.append(stop)
            audit.append(
                AuditEntry(
                    code="EVENT-INFUSION-EXPANDED-001",
                    action="generated_infusion_stop",
                    row_id=row_id,
                    source_row_id=event.source_row_id,
                    details={"time": stop.time, "rate": stop.rate},
                )
            )
        return EventTable(tuple(expanded), tuple(audit), self.source_count, self.source_records)

    def provenance(self) -> tuple[dict[str, Any], ...]:
        """Return the complete immutable audit as fresh dictionaries."""

        return tuple(entry.to_dict() for entry in self.audit)


def canonicalize_events(
    records: Iterable[Mapping[str, Any]] | np.ndarray,
    *,
    covariate_columns: Sequence[str] = (),
    expand_additional: bool = False,
    expand_infusions: bool = False,
) -> EventTable:
    """Convenience wrapper around :meth:`EventTable.from_records`."""

    return EventTable.from_records(
        records,
        covariate_columns=covariate_columns,
        expand_additional=expand_additional,
        expand_infusions=expand_infusions,
    )


__all__ = [
    "EVENT_PRIORITY",
    "AuditEntry",
    "CanonicalEvent",
    "DoseAmountStatus",
    "EventTable",
    "EventType",
    "EventValidationError",
    "canonicalize_events",
]
