"""Structured warning catalog and emission helpers.

Warnings remain available as ordinary :class:`warnings.WarningMessage`
instances, while :class:`~pymixef._contracts.WarningRecord` provides the stable
representation stored in results and manifests.
"""

from __future__ import annotations

import json
import warnings as _warnings
from importlib.resources import files
from typing import Any

from ._contracts import WarningRecord


class PyMixEFWarning(UserWarning):
    """Python warning carrying a stable PyMixEF diagnostic code."""

    def __init__(self, record: WarningRecord) -> None:
        self.record = record
        super().__init__(f"[{record.code}] {record.message}")

    @property
    def code(self) -> str:
        """Stable warning code."""

        return self.record.code


class DataAuditWarning(PyMixEFWarning):
    """Warning associated with data adaptation or record exclusion."""


class CovarianceWarning(PyMixEFWarning):
    """Warning associated with singular or boundary covariance behavior."""


class NumericalWarning(PyMixEFWarning):
    """Warning associated with an unreliable numerical calculation."""


def load_warning_catalog() -> dict[str, dict[str, Any]]:
    """Load and return the packaged warning catalog keyed by stable code."""

    resource = files("pymixef").joinpath("warning_catalog.json")
    with resource.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    entries = raw.get("warnings", raw)
    return {str(item["code"]): dict(item) for item in entries}


def warning_record(
    code: str,
    *,
    message: str | None = None,
    severity: str | None = None,
    component: str | None = None,
    remediation: str | None = None,
    details: dict[str, Any] | None = None,
) -> WarningRecord:
    """Create a record, filling omitted fields from the packaged catalog."""

    catalog = load_warning_catalog()
    template = catalog.get(code, {})
    return WarningRecord(
        code=code,
        severity=severity or str(template.get("severity", "review")),
        message=message or str(template.get("message", "PyMixEF warning")),
        component=component or template.get("component"),
        remediation=remediation or template.get("remediation"),
        details=details or {},
    )


def emit_warning(
    code: str,
    *,
    category: type[PyMixEFWarning] = PyMixEFWarning,
    stacklevel: int = 2,
    **kwargs: Any,
) -> WarningRecord:
    """Create, emit, and return a structured warning record."""

    record = warning_record(code, **kwargs)
    _warnings.warn(category(record), stacklevel=stacklevel)
    return record


__all__ = [
    "CovarianceWarning",
    "DataAuditWarning",
    "NumericalWarning",
    "PyMixEFWarning",
    "emit_warning",
    "load_warning_catalog",
    "warning_record",
]
