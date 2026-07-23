"""Stable, machine-readable exceptions used throughout :mod:`pymixef`.

Public exceptions carry a code, remediation, and structured details.  The
human-readable message may improve over time; callers should branch on ``code``
or the exception type rather than parsing message text.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class PyMixEFError(Exception):
    """Base class for all expected PyMixEF failures."""

    default_code = "PYMIXEF-ERROR-001"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        remediation: str | None = None,
        details: Mapping[str, Any] | None = None,
        source_location: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = code or self.default_code
        self.remediation = remediation
        self.details = dict(details or {})
        self.source_location = source_location

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic record."""

        return {
            "code": self.code,
            "type": type(self).__name__,
            "message": self.message,
            "remediation": self.remediation,
            "details": dict(self.details),
            "source_location": self.source_location,
        }

    def __str__(self) -> str:
        location = f" at {self.source_location}" if self.source_location else ""
        return f"[{self.code}]{location} {self.message}"


class ValidationError(PyMixEFError):
    """A model failed deterministic semantic validation."""

    default_code = "MODEL-VALIDATION-001"


class FormulaError(ValidationError):
    """A formula is syntactically invalid, unsafe, or semantically ambiguous."""

    default_code = "FORMULA-SYNTAX-001"


class DataError(ValidationError):
    """Input data cannot be adapted without changing its meaning."""

    default_code = "DATA-INVALID-001"


class CovarianceError(ValidationError):
    """A covariance declaration or matrix is invalid."""

    default_code = "COV-INVALID-001"


class TransformError(ValidationError):
    """A parameter transform received a value outside its domain."""

    default_code = "TRANSFORM-DOMAIN-001"


class IRVersionError(PyMixEFError):
    """A serialized model IR uses an unsupported or unsafe schema version."""

    default_code = "IR-VERSION-001"


class IRValidationError(ValidationError):
    """A model IR violates the versioned schema's semantic invariants."""

    default_code = "IR-VALIDATION-001"


class PluginError(PyMixEFError):
    """A plugin registration or discovery operation failed."""

    default_code = "PLUGIN-ERROR-001"


class UnsupportedCapabilityError(PyMixEFError):
    """No selected engine can execute a requested scientific capability."""

    default_code = "ENGINE-UNSUPPORTED-001"


class EngineCompatibilityError(UnsupportedCapabilityError):
    """The selected estimator cannot represent the compiled model."""

    def __init__(
        self,
        message: str,
        *,
        suggested_engines: list[str] | tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        details = dict(kwargs.pop("details", {}) or {})
        details.setdefault("suggested_engines", list(suggested_engines))
        super().__init__(message, details=details, **kwargs)
        self.suggested_engines = tuple(suggested_engines)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["suggested_engines"] = list(self.suggested_engines)
        return result


UnsupportedEngineError = EngineCompatibilityError


class CompatibilityError(PyMixEFError):
    """Two scientific objects cannot safely be compared or translated."""

    default_code = "COMPATIBILITY-001"


__all__ = [
    "CompatibilityError",
    "CovarianceError",
    "DataError",
    "EngineCompatibilityError",
    "FormulaError",
    "IRValidationError",
    "IRVersionError",
    "PluginError",
    "PyMixEFError",
    "TransformError",
    "UnsupportedCapabilityError",
    "UnsupportedEngineError",
    "ValidationError",
]
