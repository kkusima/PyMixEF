"""Small shared contracts used across PyMixEF's independently implemented layers.

This module deliberately depends only on the Python standard library.  Scientific
subsystems may import these types without creating dependency cycles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Maturity(StrEnum):
    """Evidence tier attached to every public capability."""

    EXPERIMENTAL = "experimental"
    STABLE = "stable"
    REFERENCE_VALIDATED = "reference-validated"
    REGULATED_WORKFLOW_SUPPORT = "regulated-workflow-support"


class ReproducibilityClass(StrEnum):
    """Numerical reproducibility guarantee declared by an engine."""

    BITWISE = "bitwise"
    DETERMINISTIC_TOLERANCE = "deterministic-with-tolerance"
    STOCHASTIC_MONTE_CARLO = "stochastic-with-monte-carlo-error"


@dataclass(frozen=True, slots=True)
class WarningRecord:
    """Machine-readable scientific or numerical warning."""

    code: str
    severity: str
    message: str
    component: str | None = None
    remediation: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "component": self.component,
            "remediation": self.remediation,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One exact, transformed, approximated, or unsupported interop construct."""

    construct: str
    status: str
    message: str
    source_location: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "construct": self.construct,
            "status": self.status,
            "message": self.message,
            "source_location": self.source_location,
        }


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
