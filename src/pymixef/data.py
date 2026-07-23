"""Dependency-light input adapters, stable row identities, and data auditing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import DataError


class MissingnessKind(StrEnum):
    """Semantically distinct missing-record conditions."""

    MISSING_RESPONSE = "missing-response"
    CENSORED_RESPONSE = "censored-response"
    STRUCTURALLY_ABSENT_ENDPOINT = "structurally-absent-endpoint"
    INVALID_RECORD = "invalid-record"
    MISSING_COVARIATE = "missing-covariate"


_REASON_CODES: dict[MissingnessKind, str] = {
    MissingnessKind.MISSING_RESPONSE: "DATA-MISSING-RESPONSE-001",
    MissingnessKind.CENSORED_RESPONSE: "DATA-CENSORED-RESPONSE-001",
    MissingnessKind.STRUCTURALLY_ABSENT_ENDPOINT: "DATA-STRUCTURAL-ABSENCE-001",
    MissingnessKind.INVALID_RECORD: "DATA-INVALID-RECORD-001",
    MissingnessKind.MISSING_COVARIATE: "DATA-MISSING-COVARIATE-001",
}


def is_missing(value: Any) -> bool:
    """Return whether a scalar is missing without requiring pandas."""

    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, (complex, np.complexfloating)):
        return bool(np.isnan(value.real) or np.isnan(value.imag))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    try:
        result = value != value
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def missing_mask(values: ArrayLike) -> NDArray[np.bool_]:
    """Return a one-dimensional missing-value mask."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise DataError("Data columns must be one-dimensional.", code="DATA-COLUMN-SHAPE-001")
    if np.issubdtype(array.dtype, np.floating):
        return np.isnan(array)
    if np.issubdtype(array.dtype, np.complexfloating):
        return np.isnan(array.real) | np.isnan(array.imag)
    if np.issubdtype(array.dtype, np.datetime64) or np.issubdtype(array.dtype, np.timedelta64):
        return np.isnat(array)
    return np.fromiter((is_missing(item) for item in array), dtype=bool, count=array.size)


def _json_cell(value: Any) -> Any:
    if is_missing(value):
        return {"missing": True}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return repr(value)


def _row_identifier(index: Any, row: Sequence[Any], occurrence: int) -> str:
    payload = json.dumps(
        {
            "index": _json_cell(index),
            "row": [_json_cell(item) for item in row],
            "occurrence": occurrence,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """Normalized metadata for one input column."""

    name: str
    dtype: str
    categorical: bool = False
    levels: tuple[Any, ...] = ()
    ordered: bool = False
    missing_count: int = 0
    role: str | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "categorical": self.categorical,
            "levels": [_json_cell(item) for item in self.levels],
            "ordered": self.ordered,
            "missing_count": self.missing_count,
            "role": self.role,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ColumnarData(Mapping[str, NDArray[Any]]):
    """Immutable column-oriented internal table.

    Arrays are copied on ingestion and marked read-only.  The internal format is
    deliberately independent of pandas, Polars, Arrow, and xarray.
    """

    columns: Mapping[str, NDArray[Any]]
    row_ids: tuple[str, ...]
    source_index: tuple[Any, ...]
    schema: tuple[ColumnSchema, ...]
    source_type: str = "mapping"

    def __post_init__(self) -> None:
        normalized: dict[str, NDArray[Any]] = {}
        length: int | None = None
        for name, raw in self.columns.items():
            array = np.asarray(raw)
            if array.ndim != 1:
                raise DataError(
                    f"Column {name!r} must be one-dimensional.",
                    code="DATA-COLUMN-SHAPE-001",
                )
            if length is None:
                length = array.size
            elif array.size != length:
                raise DataError(
                    "All data columns must have equal lengths.",
                    code="DATA-LENGTH-001",
                )
            copied = np.array(array, copy=True)
            copied.setflags(write=False)
            normalized[str(name)] = copied
        length = length or 0
        if len(self.row_ids) != length or len(self.source_index) != length:
            raise DataError("Row identity metadata does not match the column lengths.")
        object.__setattr__(self, "columns", MappingProxyType(normalized))
        object.__setattr__(self, "row_ids", tuple(self.row_ids))
        object.__setattr__(self, "source_index", tuple(self.source_index))
        object.__setattr__(self, "schema", tuple(self.schema))

    def __getitem__(self, key: str) -> NDArray[Any]:
        try:
            return self.columns[key]
        except KeyError as exc:
            raise DataError(
                f"Data column {key!r} was not found.",
                code="DATA-COLUMN-NOT-FOUND-001",
                details={"available": list(self.columns)},
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def take(self, positions: ArrayLike) -> ColumnarData:
        """Return a row subset while preserving original stable identifiers."""

        index = np.asarray(positions)
        if index.dtype == bool:
            if index.size != self.n_rows:
                raise DataError("Boolean row mask has the wrong length.")
            index = np.flatnonzero(index)
        index = index.astype(int, copy=False).reshape(-1)
        return ColumnarData(
            {name: values[index] for name, values in self.columns.items()},
            tuple(self.row_ids[item] for item in index),
            tuple(self.source_index[item] for item in index),
            tuple(
                ColumnSchema(
                    name=item.name,
                    dtype=item.dtype,
                    categorical=item.categorical,
                    levels=item.levels,
                    ordered=item.ordered,
                    missing_count=int(np.sum(missing_mask(self.columns[item.name][index]))),
                    role=item.role,
                    unit=item.unit,
                )
                for item in self.schema
            ),
            self.source_type,
        )

    def to_dict(self, *, copy: bool = True) -> dict[str, NDArray[Any]]:
        """Return columns, copied by default."""

        return {
            name: np.array(value, copy=True) if copy else value
            for name, value in self.columns.items()
        }

    @property
    def fingerprint(self) -> str:
        """Stable content hash including schema, order, and missing markers."""

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                [item.to_dict() for item in self.schema],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for position in range(self.n_rows):
            digest.update(self.row_ids[position].encode())
            for column in self.columns.values():
                digest.update(
                    json.dumps(_json_cell(column[position]), sort_keys=True).encode("utf-8")
                )
        return digest.hexdigest()


def _extract_columns(
    data: Any,
    column_names: Sequence[str] | None,
) -> tuple[dict[str, Any], tuple[Any, ...], str, dict[str, tuple[tuple[Any, ...], bool]]]:
    category_metadata: dict[str, tuple[tuple[Any, ...], bool]] = {}
    if isinstance(data, ColumnarData):
        return (
            data.to_dict(),
            data.source_index,
            data.source_type,
            {item.name: (item.levels, item.ordered) for item in data.schema if item.categorical},
        )
    if isinstance(data, Mapping):
        columns = {str(name): values for name, values in data.items()}
        length = len(next(iter(columns.values()))) if columns else 0
        return columns, tuple(range(length)), type(data).__name__, category_metadata
    if isinstance(data, np.ndarray):
        if data.dtype.names:
            columns = {str(name): data[name] for name in data.dtype.names}
            return columns, tuple(range(len(data))), "numpy-structured", category_metadata
        if data.ndim != 2:
            raise DataError(
                "A plain NumPy input must be two-dimensional.",
                code="DATA-ARRAY-SHAPE-001",
            )
        if column_names is None or len(column_names) != data.shape[1]:
            raise DataError(
                "column_names must identify every NumPy array column.",
                code="DATA-COLUMN-NAMES-001",
            )
        columns = {str(name): data[:, position] for position, name in enumerate(column_names)}
        return columns, tuple(range(data.shape[0])), "numpy", category_metadata

    # Arrow-like tables expose a column_names collection and Python-list
    # conversion.  Check this before the broader dataframe protocol.
    if hasattr(data, "column_names") and hasattr(data, "column"):
        try:
            columns = {str(name): data.column(name).to_pylist() for name in data.column_names}
        except (AttributeError, TypeError):
            columns = {}
        else:
            length = len(next(iter(columns.values()))) if columns else 0
            return columns, tuple(range(length)), type(data).__name__, category_metadata

    # Polars-like dataframes use ``to_dict(as_series=False)`` and do not expose
    # a pandas Index.  This also avoids passing pandas-only ``copy=`` arguments
    # to a Polars Series.
    if hasattr(data, "to_dict") and not hasattr(data, "index"):
        try:
            raw = data.to_dict(as_series=False)
        except (TypeError, AttributeError):
            raw = None
        if isinstance(raw, Mapping):
            columns = {str(name): values for name, values in raw.items()}
            length = len(next(iter(columns.values()))) if columns else 0
            return columns, tuple(range(length)), type(data).__name__, category_metadata

    # pandas-like: columns, index and per-Series array conversion.
    if hasattr(data, "columns") and hasattr(data, "__getitem__"):
        names = tuple(str(name) for name in data.columns)
        columns = {}
        for original, name in zip(data.columns, names, strict=True):
            series = data[original]
            categorical = getattr(series, "cat", None)
            if categorical is not None:
                with suppress(AttributeError, TypeError):
                    category_metadata[name] = (
                        tuple(categorical.categories.tolist()),
                        bool(categorical.ordered),
                    )
            columns[name] = (
                series.to_numpy(copy=True) if hasattr(series, "to_numpy") else np.asarray(series)
            )
        raw_index = getattr(data, "index", range(len(next(iter(columns.values()), ()))))
        index = tuple(raw_index.tolist() if hasattr(raw_index, "tolist") else raw_index)
        return columns, index, f"{type(data).__module__}.{type(data).__name__}", category_metadata

    # Generic dataframe-like fallback.
    if hasattr(data, "to_dict"):
        try:
            raw = data.to_dict(as_series=False)
        except TypeError:
            raw = data.to_dict()
        if isinstance(raw, Mapping):
            columns = {str(name): values for name, values in raw.items()}
            length = len(next(iter(columns.values()))) if columns else 0
            return columns, tuple(range(length)), type(data).__name__, category_metadata

    # xarray-like tables can expose a dataframe adapter without importing xarray.
    if hasattr(data, "to_dataframe"):
        return _extract_columns(data.to_dataframe().reset_index(), column_names)

    raise DataError(
        f"Unsupported data input type {type(data).__module__}.{type(data).__name__}.",
        code="DATA-ADAPTER-001",
    )


def adapt_data(
    data: Any,
    *,
    column_names: Sequence[str] | None = None,
    roles: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
) -> ColumnarData:
    """Normalize a mapping or supported dataframe-like object."""

    if isinstance(data, ColumnarData):
        return data
    columns, source_index, source_type, categories = _extract_columns(data, column_names)
    normalized = {name: np.asarray(value) for name, value in columns.items()}
    lengths = {array.shape[0] for array in normalized.values() if array.ndim >= 1}
    if any(array.ndim != 1 for array in normalized.values()):
        raise DataError("All adapted columns must be one-dimensional.")
    if len(lengths) > 1:
        raise DataError(
            "All data columns must have equal lengths.",
            code="DATA-LENGTH-001",
        )
    length = lengths.pop() if lengths else 0
    if len(source_index) != length:
        source_index = tuple(range(length))
    names = tuple(normalized)
    occurrences: Counter[str] = Counter()
    row_ids: list[str] = []
    for position in range(length):
        row = tuple(normalized[name][position] for name in names)
        base = json.dumps(
            {"index": _json_cell(source_index[position]), "row": [_json_cell(x) for x in row]},
            sort_keys=True,
            default=repr,
        )
        occurrence = occurrences[base]
        occurrences[base] += 1
        row_ids.append(_row_identifier(source_index[position], row, occurrence))
    schema = tuple(
        ColumnSchema(
            name=name,
            dtype=str(array.dtype),
            categorical=name in categories,
            levels=categories.get(name, ((), False))[0],
            ordered=categories.get(name, ((), False))[1],
            missing_count=int(np.sum(missing_mask(array))),
            role=(roles or {}).get(name),
            unit=(units or {}).get(name),
        )
        for name, array in normalized.items()
    )
    return ColumnarData(normalized, tuple(row_ids), source_index, schema, source_type)


InputAdapter = adapt_data


class DataAdapter:
    """Namespace-style adapter for callers preferring an object API."""

    @staticmethod
    def adapt(
        data: Any,
        *,
        column_names: Sequence[str] | None = None,
        roles: Mapping[str, str] | None = None,
        units: Mapping[str, str] | None = None,
    ) -> ColumnarData:
        """Delegate to :func:`adapt_data`."""

        return adapt_data(
            data,
            column_names=column_names,
            roles=roles,
            units=units,
        )


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Audit disposition for one original input row."""

    row_id: str
    input_position: int
    source_index: Any
    action: str
    reason_code: str
    missingness: MissingnessKind | None = None
    columns: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "input_position": self.input_position,
            "source_index": _json_cell(self.source_index),
            "action": self.action,
            "reason_code": self.reason_code,
            "missingness": self.missingness.value if self.missingness else None,
            "columns": list(self.columns),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class DataAudit:
    """Complete reconciliation of source and analysis rows."""

    input_rows: int
    analysis_rows: int
    records: tuple[AuditRecord, ...]
    factor_levels: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    factor_ordered: Mapping[str, bool] = field(default_factory=dict)
    contrast_coding: Mapping[str, str] = field(default_factory=dict)
    transformations: tuple[Mapping[str, Any], ...] = ()
    source_fingerprint: str | None = None
    analysis_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if len(self.records) != self.input_rows:
            raise DataError("The data audit must contain one record per input row.")
        retained = sum(record.action == "retained" for record in self.records)
        if retained != self.analysis_rows:
            raise DataError("The data audit does not reconcile analysis row counts.")
        object.__setattr__(
            self,
            "factor_levels",
            MappingProxyType(
                {str(name): tuple(levels) for name, levels in self.factor_levels.items()}
            ),
        )
        object.__setattr__(
            self,
            "factor_ordered",
            MappingProxyType(
                {str(name): bool(ordered) for name, ordered in self.factor_ordered.items()}
            ),
        )
        object.__setattr__(
            self,
            "contrast_coding",
            MappingProxyType(dict(self.contrast_coding)),
        )
        object.__setattr__(
            self,
            "transformations",
            tuple(MappingProxyType(dict(item)) for item in self.transformations),
        )

    @property
    def excluded_rows(self) -> int:
        return self.input_rows - self.analysis_rows

    @property
    def excluded_row_ids(self) -> tuple[str, ...]:
        return tuple(record.row_id for record in self.records if record.action == "excluded")

    @property
    def reason_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(Counter(record.reason_code for record in self.records)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "analysis_rows": self.analysis_rows,
            "excluded_rows": self.excluded_rows,
            "records": [record.to_dict() for record in self.records],
            "reason_counts": dict(self.reason_counts),
            "factor_levels": {
                name: [_json_cell(item) for item in levels]
                for name, levels in self.factor_levels.items()
            },
            "factor_ordered": dict(self.factor_ordered),
            "contrast_coding": dict(self.contrast_coding),
            "transformations": [dict(item) for item in self.transformations],
            "source_fingerprint": self.source_fingerprint,
            "analysis_fingerprint": self.analysis_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class AuditedData:
    """Analysis subset paired with its complete source-row audit."""

    source: ColumnarData
    data: ColumnarData
    audit: DataAudit
    retained_positions: tuple[int, ...]

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self.data.row_ids

    def __iter__(self) -> Iterator[Any]:
        # Convenient, explicit two-value unpacking: ``data, audit = audit_data(...)``.
        yield self.data
        yield self.audit


@dataclass(frozen=True, slots=True)
class PatternMixtureRecord:
    """One response-scale adjustment applied to an explicitly imputed value."""

    row_id: str
    input_position: int
    source_index: Any
    stratum: tuple[Any, ...]
    before: float
    delta: float
    after: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit record."""

        return {
            "row_id": self.row_id,
            "input_position": self.input_position,
            "source_index": _json_cell(self.source_index),
            "stratum": [_json_cell(item) for item in self.stratum],
            "before": self.before,
            "delta": self.delta,
            "after": self.after,
        }


@dataclass(frozen=True, slots=True)
class PatternMixtureResult:
    """Adjusted completed data paired with a row-level sensitivity audit."""

    data: ColumnarData
    response: str
    imputed_column: str | None
    stratified_by: tuple[str, ...]
    records: tuple[PatternMixtureRecord, ...]
    source_fingerprint: str

    @property
    def adjusted_rows(self) -> int:
        """Number of explicitly imputed response values adjusted."""

        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable metadata without duplicating the full adjusted data."""

        return {
            "response": self.response,
            "imputed_column": self.imputed_column,
            "stratified_by": list(self.stratified_by),
            "adjusted_rows": self.adjusted_rows,
            "source_fingerprint": self.source_fingerprint,
            "adjusted_fingerprint": self.data.fingerprint,
            "records": [record.to_dict() for record in self.records],
        }


def _pattern_mixture_mask(
    imputed: str | ArrayLike,
    table: ColumnarData,
) -> tuple[NDArray[np.bool_], str | None]:
    raw = table[imputed] if isinstance(imputed, str) else np.asarray(imputed)
    mask = np.asarray(raw)
    if mask.ndim != 1 or mask.size != table.n_rows:
        raise DataError(
            "The imputed-value mask must be one-dimensional and align with the data.",
            code="DATA-SENSITIVITY-MASK-001",
        )
    if mask.dtype.kind != "b":
        raise DataError(
            "The imputed-value mask must contain Boolean values.",
            code="DATA-SENSITIVITY-MASK-001",
        )
    return mask.astype(bool, copy=False), imputed if isinstance(imputed, str) else None


def _pattern_mixture_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_pattern_mixture_key(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def pattern_mixture_adjust(
    data: Any,
    *,
    response: str,
    imputed: str | ArrayLike,
    delta: float | Mapping[Any, float],
    by: str | Sequence[str] | None = None,
) -> PatternMixtureResult:
    """Apply an audited delta adjustment only to explicitly imputed responses.

    This is a controlled pattern-mixture *data transformation*, not an
    imputation algorithm. Callers first complete the response under their
    primary missing-at-random procedure, then identify those imputed cells with
    ``imputed``. A scalar ``delta`` applies one response-scale shift. A mapping
    requires ``by`` and supplies a shift for every selected stratum; with
    multiple ``by`` columns, mapping keys are tuples in the same order.

    Observed response values and every non-response column are preserved
    exactly. The returned records retain source row identities, the base
    imputed value, applied shift, and adjusted value.
    """

    table = adapt_data(data)
    if response not in table.column_names:
        raise DataError(
            f"Response column {response!r} was not found.",
            code="DATA-COLUMN-NOT-FOUND-001",
        )
    if isinstance(imputed, str) and imputed == response:
        raise DataError(
            "The imputed-value indicator must be separate from the response column.",
            code="DATA-SENSITIVITY-MASK-001",
        )
    mask, imputed_column = _pattern_mixture_mask(imputed, table)
    if not np.any(mask):
        raise DataError(
            "The imputed-value mask selects no rows.",
            code="DATA-SENSITIVITY-EMPTY-001",
        )

    if by is None:
        strata_names: tuple[str, ...] = ()
    elif isinstance(by, str):
        strata_names = (by,)
    else:
        strata_names = tuple(str(name) for name in by)
    absent = tuple(name for name in strata_names if name not in table.column_names)
    if absent:
        raise DataError(
            f"Sensitivity-stratum columns are absent: {', '.join(absent)}.",
            code="DATA-COLUMN-NOT-FOUND-001",
            details={"columns": list(absent)},
        )

    if isinstance(delta, Mapping):
        if not strata_names:
            raise DataError(
                "A stratum-specific delta mapping requires at least one 'by' column.",
                code="DATA-SENSITIVITY-DELTA-001",
            )
        try:
            delta_map = {_pattern_mixture_key(key): float(value) for key, value in delta.items()}
        except (TypeError, ValueError, OverflowError) as error:
            raise DataError(
                "Every pattern-mixture delta must be numeric.",
                code="DATA-SENSITIVITY-DELTA-001",
            ) from error
        if not delta_map or not all(np.isfinite(value) for value in delta_map.values()):
            raise DataError(
                "Every pattern-mixture delta must be finite.",
                code="DATA-SENSITIVITY-DELTA-001",
            )
        scalar_delta: float | None = None
    else:
        try:
            scalar_delta = float(delta)
        except (TypeError, ValueError, OverflowError) as error:
            raise DataError(
                "The pattern-mixture delta must be numeric.",
                code="DATA-SENSITIVITY-DELTA-001",
            ) from error
        if not np.isfinite(scalar_delta):
            raise DataError(
                "The pattern-mixture delta must be finite.",
                code="DATA-SENSITIVITY-DELTA-001",
            )
        delta_map = {}

    original_response = np.asarray(table[response])
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in original_response
    ):
        raise DataError(
            "Pattern-mixture adjustment requires a numeric, non-Boolean response.",
            code="DATA-SENSITIVITY-RESPONSE-001",
        )
    try:
        numeric_response = np.asarray(
            [float(value) for value in original_response],
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DataError(
            "Pattern-mixture adjustment requires a numeric completed response.",
            code="DATA-SENSITIVITY-RESPONSE-001",
        ) from error
    if np.any(~np.isfinite(numeric_response)):
        raise DataError(
            "All responses must be completed and finite before delta adjustment.",
            code="DATA-SENSITIVITY-RESPONSE-001",
        )
    # Object storage is intentional: an adjusted cell may become floating point
    # while unselected integer observations (including values above 2**53) must
    # retain their exact original scalar values.
    adjusted_response = np.asarray(original_response, dtype=object).copy()

    records: list[PatternMixtureRecord] = []
    for position in np.flatnonzero(mask):
        stratum = tuple(_pattern_mixture_key(table[name][position]) for name in strata_names)
        lookup: Any = stratum[0] if len(stratum) == 1 else stratum
        if scalar_delta is None:
            if lookup not in delta_map:
                raise DataError(
                    f"No delta was supplied for selected stratum {lookup!r}.",
                    code="DATA-SENSITIVITY-DELTA-001",
                    details={
                        "position": int(position),
                        "stratum": [_json_cell(item) for item in stratum],
                    },
                )
            shift = delta_map[lookup]
        else:
            shift = scalar_delta
        before = float(numeric_response[position])
        after = before + shift
        if not np.isfinite(after):
            raise DataError(
                "A pattern-mixture adjustment overflowed the finite response scale.",
                code="DATA-SENSITIVITY-OVERFLOW-001",
                details={
                    "position": int(position),
                    "before": before,
                    "delta": shift,
                },
            )
        adjusted_response[position] = after
        records.append(
            PatternMixtureRecord(
                row_id=table.row_ids[position],
                input_position=int(position),
                source_index=table.source_index[position],
                stratum=stratum,
                before=before,
                delta=shift,
                after=after,
            )
        )

    adjusted_columns = table.to_dict()
    adjusted_columns[response] = adjusted_response
    adjusted_schema = tuple(
        ColumnSchema(
            name=item.name,
            dtype=str(adjusted_response.dtype) if item.name == response else item.dtype,
            categorical=item.categorical,
            levels=item.levels,
            ordered=item.ordered,
            missing_count=(
                int(np.sum(missing_mask(adjusted_response)))
                if item.name == response
                else item.missing_count
            ),
            role=item.role,
            unit=item.unit,
        )
        for item in table.schema
    )
    adjusted = ColumnarData(
        adjusted_columns,
        table.row_ids,
        table.source_index,
        adjusted_schema,
        table.source_type,
    )
    return PatternMixtureResult(
        data=adjusted,
        response=response,
        imputed_column=imputed_column,
        stratified_by=strata_names,
        records=tuple(records),
        source_fingerprint=table.fingerprint,
    )


def _condition_mask(condition: str | ArrayLike | None, data: ColumnarData) -> NDArray[np.bool_]:
    if condition is None:
        return np.zeros(data.n_rows, dtype=bool)
    raw = data[condition] if isinstance(condition, str) else np.asarray(condition)
    result = np.asarray(raw, dtype=bool).reshape(-1)
    if result.size != data.n_rows:
        raise DataError("A record-status mask has the wrong length.")
    return result


def audit_data(
    data: Any,
    *,
    response: str | None = None,
    covariates: Sequence[str] = (),
    censored: str | ArrayLike | None = None,
    structurally_absent: str | ArrayLike | None = None,
    invalid: str | ArrayLike | None = None,
    missing: str = "drop",
    factor_levels: Mapping[str, Sequence[Any]] | None = None,
    contrast_coding: Mapping[str, str] | None = None,
) -> AuditedData:
    """Apply the missingness contract and produce a row-complete audit.

    ``missing`` may be ``"drop"``, ``"raise"``, or ``"keep"``.  Invalid and
    structurally absent records are always excluded; censored records are
    retained even when their response value is absent.
    """

    table = adapt_data(data)
    if missing not in {"drop", "raise", "keep"}:
        raise DataError(
            "missing must be 'drop', 'raise', or 'keep'.",
            code="DATA-MISSING-POLICY-001",
        )
    required = ([response] if response else []) + list(covariates)
    absent = [name for name in required if name not in table]
    if absent:
        raise DataError(
            f"Required data columns are absent: {', '.join(absent)}.",
            code="DATA-COLUMN-NOT-FOUND-001",
            details={"columns": absent},
        )
    censored_mask = _condition_mask(censored, table)
    structural_mask = _condition_mask(structurally_absent, table)
    invalid_mask = _condition_mask(invalid, table)
    response_missing = (
        missing_mask(table[response]) if response is not None else np.zeros(table.n_rows, bool)
    )
    covariate_missing: dict[str, NDArray[np.bool_]] = {
        name: missing_mask(table[name]) for name in covariates
    }
    records: list[AuditRecord] = []
    retained: list[int] = []
    for position in range(table.n_rows):
        conditions: list[tuple[MissingnessKind, tuple[str, ...]]] = []
        if invalid_mask[position]:
            conditions.append((MissingnessKind.INVALID_RECORD, ()))
        if structural_mask[position]:
            conditions.append(
                (MissingnessKind.STRUCTURALLY_ABSENT_ENDPOINT, (response,) if response else ())
            )
        missing_columns = tuple(name for name, mask in covariate_missing.items() if mask[position])
        if missing_columns:
            conditions.append((MissingnessKind.MISSING_COVARIATE, missing_columns))
        if censored_mask[position]:
            conditions.append((MissingnessKind.CENSORED_RESPONSE, (response,) if response else ()))
        elif response_missing[position]:
            conditions.append((MissingnessKind.MISSING_RESPONSE, (response,) if response else ()))

        excluding = [
            item
            for item in conditions
            if item[0]
            in {
                MissingnessKind.INVALID_RECORD,
                MissingnessKind.STRUCTURALLY_ABSENT_ENDPOINT,
                MissingnessKind.MISSING_COVARIATE,
                MissingnessKind.MISSING_RESPONSE,
            }
        ]
        if missing == "keep":
            excluding = [
                item
                for item in excluding
                if item[0]
                in {
                    MissingnessKind.INVALID_RECORD,
                    MissingnessKind.STRUCTURALLY_ABSENT_ENDPOINT,
                }
            ]
        if missing == "raise" and excluding:
            excluding_kind, excluding_columns = excluding[0]
            raise DataError(
                f"Row {position} violates the missingness contract: {excluding_kind.value}.",
                code=_REASON_CODES[excluding_kind],
                details={
                    "row_id": table.row_ids[position],
                    "columns": list(excluding_columns),
                },
            )
        primary = excluding[0] if excluding else (conditions[0] if conditions else None)
        action = "excluded" if excluding else "retained"
        if action == "retained":
            retained.append(position)
        disposition_kind: MissingnessKind | None = primary[0] if primary else None
        columns = primary[1] if primary else ()
        records.append(
            AuditRecord(
                row_id=table.row_ids[position],
                input_position=position,
                source_index=table.source_index[position],
                action=action,
                reason_code=(
                    _REASON_CODES[disposition_kind] if disposition_kind else "DATA-RETAINED-001"
                ),
                missingness=disposition_kind,
                columns=columns,
                details={"all_conditions": [item[0].value for item in conditions]},
            )
        )
    analysis = table.take(retained)
    inferred_levels = {item.name: item.levels for item in table.schema if item.categorical}
    inferred_ordering = {item.name: item.ordered for item in table.schema if item.categorical}
    inferred_levels.update(
        {str(name): tuple(levels) for name, levels in (factor_levels or {}).items()}
    )
    audit = DataAudit(
        input_rows=table.n_rows,
        analysis_rows=analysis.n_rows,
        records=tuple(records),
        factor_levels=inferred_levels,
        factor_ordered=inferred_ordering,
        contrast_coding=contrast_coding or {},
        transformations=(),
        source_fingerprint=table.fingerprint,
        analysis_fingerprint=analysis.fingerprint,
    )
    return AuditedData(table, analysis, audit, tuple(retained))


prepare_data = audit_data


def find_duplicate_keys(data: Any, keys: Sequence[str]) -> tuple[tuple[Any, ...], ...]:
    """Return duplicate key values in deterministic first-seen order."""

    table = adapt_data(data)
    if not keys:
        raise DataError("At least one duplicate-detection key is required.")
    seen: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    duplicate_set: set[tuple[Any, ...]] = set()
    for position in range(table.n_rows):
        key = tuple(_json_cell(table[name][position]) for name in keys)
        if key in seen and key not in duplicate_set:
            duplicates.append(key)
            duplicate_set.add(key)
        seen.add(key)
    return tuple(duplicates)


def validate_monotonic_time(data: Any, *, group: str, time: str) -> None:
    """Raise if time decreases within a group in source row order."""

    table = adapt_data(data)
    previous: dict[Any, float] = {}
    for position, (group_value, time_value) in enumerate(
        zip(table[group], table[time], strict=True)
    ):
        if is_missing(group_value) or is_missing(time_value):
            continue
        numeric_time = float(time_value)
        if group_value in previous and numeric_time < previous[group_value]:
            raise DataError(
                f"Time decreases within group {group_value!r} at row {position}.",
                code="DATA-TIME-ORDER-001",
                details={"row_id": table.row_ids[position], "group": _json_cell(group_value)},
            )
        previous[group_value] = numeric_time


def stable_sort(data: Any, keys: Sequence[str]) -> tuple[ColumnarData, tuple[int, ...]]:
    """Stable lexicographic sort returning the source-position permutation."""

    table = adapt_data(data)
    if not keys:
        return table, tuple(range(table.n_rows))
    arrays = [table[name] for name in reversed(keys)]
    try:
        order = np.lexsort(arrays)
    except TypeError:
        order = np.asarray(
            sorted(
                range(table.n_rows),
                key=lambda position: tuple(repr(table[name][position]) for name in keys),
            )
        )
    return table.take(order), tuple(int(item) for item in order)


__all__ = [
    "AuditRecord",
    "AuditedData",
    "ColumnSchema",
    "ColumnarData",
    "DataAdapter",
    "DataAudit",
    "InputAdapter",
    "MissingnessKind",
    "PatternMixtureRecord",
    "PatternMixtureResult",
    "adapt_data",
    "audit_data",
    "find_duplicate_keys",
    "is_missing",
    "missing_mask",
    "pattern_mixture_adjust",
    "prepare_data",
    "stable_sort",
    "validate_monotonic_time",
]
