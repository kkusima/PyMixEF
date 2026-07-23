"""Shared compatibility report types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from .._contracts import CompatibilityIssue
from .._serialization import write_json

T = TypeVar("T")
_STATUSES = {"exact", "transformed", "approximated", "unsupported"}


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Machine-readable accounting for every translated construct."""

    source_format: str
    target_format: str
    issues: tuple[CompatibilityIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        invalid = [issue.status for issue in self.issues if issue.status not in _STATUSES]
        if invalid:
            raise ValueError(f"Unknown compatibility status values: {invalid}.")

    @property
    def supported(self) -> bool:
        return not any(issue.status == "unsupported" for issue in self.issues)

    def by_status(self, status: str) -> tuple[CompatibilityIssue, ...]:
        if status not in _STATUSES:
            raise ValueError(f"Unknown compatibility status {status!r}.")
        return tuple(issue for issue in self.issues if issue.status == status)

    def require_supported(self) -> None:
        unsupported = self.by_status("unsupported")
        if unsupported:
            details = "; ".join(f"{item.construct}: {item.message}" for item in unsupported)
            raise ValueError(
                "Interchange refused because unsupported constructs were found: " + details
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "supported": self.supported,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }

    def write(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class InterchangeResult(Generic[T]):
    """A translated value paired with its mandatory compatibility report."""

    value: T
    report: CompatibilityReport

    def require_supported(self) -> T:
        self.report.require_supported()
        return self.value


def issues(status: str, constructs: Iterable[tuple[str, str]]) -> tuple[CompatibilityIssue, ...]:
    """Create compatibility issues that share one translation status.

    Status validation is deferred to :class:`CompatibilityReport`, so callers
    should attach the returned issues to a report before exposing a result.
    """

    return tuple(
        CompatibilityIssue(construct=construct, status=status, message=message)
        for construct, message in constructs
    )
