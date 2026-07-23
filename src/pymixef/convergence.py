"""Backend-neutral convergence and numerical-quality reporting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._contracts import WarningRecord

_VALID_STATUSES = {"converged", "warning", "failed", "not-run"}


@dataclass(frozen=True, slots=True)
class HessianDiagnostics:
    """Definiteness and conditioning summary for an observed Hessian."""

    positive_definite: bool | None = None
    min_eigenvalue: float | None = None
    max_eigenvalue: float | None = None
    condition_number: float | None = None
    effective_rank: int | None = None

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, *, relative_tolerance: float = 1e-8
    ) -> HessianDiagnostics:
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("A Hessian must be a square matrix.")
        symmetric = (array + array.T) / 2.0
        values = np.linalg.eigvalsh(symmetric)
        tiny = float(np.finfo(np.float64).tiny)
        scale = max(float(np.max(np.abs(values))), tiny)
        rank = int(np.sum(np.abs(values) > relative_tolerance * scale))
        min_value = float(values[0])
        max_value = float(values[-1])
        absolute = np.abs(values)
        nonzero = absolute[absolute > relative_tolerance * scale]
        condition = float(np.max(absolute) / np.min(nonzero)) if nonzero.size else np.inf
        return cls(
            positive_definite=bool(min_value > relative_tolerance * scale),
            min_eigenvalue=min_value,
            max_eigenvalue=max_value,
            condition_number=condition,
            effective_rank=rank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_definite": self.positive_definite,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "condition_number": self.condition_number,
            "effective_rank": self.effective_rank,
        }


@dataclass(frozen=True, slots=True)
class BoundaryRecord:
    """One natural-scale parameter on or near a numerical boundary."""

    parameter: str
    value: float
    boundary: str = "zero"
    tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "boundary": self.boundary,
            "tolerance": self.tolerance,
        }


def _warning_from_value(value: WarningRecord | Mapping[str, Any]) -> WarningRecord:
    if isinstance(value, WarningRecord):
        return value
    return WarningRecord(
        code=str(value.get("code", "UNSPECIFIED-WARNING")),
        severity=str(value.get("severity", "review")),
        message=str(value.get("message", "")),
        component=value.get("component"),
        remediation=value.get("remediation"),
        details=dict(value.get("details", {})),
    )


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    """Structured convergence contract shared by every estimator."""

    status: str
    optimizer_terminated: bool
    optimizer_message: str = ""
    iterations: int | None = None
    objective_evaluations: int | None = None
    gradient_evaluations: int | None = None
    scaled_gradient_inf_norm: float | None = None
    parameter_step_norm: float | None = None
    hessian: HessianDiagnostics = field(default_factory=HessianDiagnostics)
    boundaries: tuple[BoundaryRecord, ...] = ()
    conditional_mode_failures: int = 0
    ode_failures: int = 0
    warnings: tuple[WarningRecord, ...] = ()
    engine_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"Unknown convergence status {self.status!r}; expected one of "
                f"{sorted(_VALID_STATUSES)}."
            )

    @property
    def trustworthy(self) -> bool:
        """Whether termination and numerical checks support routine interpretation."""

        return (
            self.status == "converged"
            and self.optimizer_terminated
            and self.conditional_mode_failures == 0
            and self.ode_failures == 0
            and self.hessian.positive_definite is not False
            and not any(w.severity in {"error", "critical"} for w in self.warnings)
        )

    @classmethod
    def assess(
        cls,
        *,
        optimizer_terminated: bool,
        gradient: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
        gradient_tolerance: float = 1e-4,
        boundaries: Iterable[BoundaryRecord] = (),
        warnings: Iterable[WarningRecord | Mapping[str, Any]] = (),
        **metrics: Any,
    ) -> ConvergenceReport:
        """Construct a report from common deterministic optimizer diagnostics."""

        gradient_norm = (
            float(np.linalg.norm(np.asarray(gradient, dtype=float), ord=np.inf))
            if gradient is not None
            else None
        )
        hessian_report = (
            HessianDiagnostics.from_matrix(hessian) if hessian is not None else HessianDiagnostics()
        )
        warning_records = tuple(_warning_from_value(item) for item in warnings)
        boundary_records = tuple(boundaries)
        failed = not optimizer_terminated
        suspect = (
            (gradient_norm is not None and gradient_norm > gradient_tolerance)
            or hessian_report.positive_definite is False
            or bool(boundary_records)
            or bool(warning_records)
        )
        status = "failed" if failed else ("warning" if suspect else "converged")
        return cls(
            status=status,
            optimizer_terminated=optimizer_terminated,
            scaled_gradient_inf_norm=gradient_norm,
            hessian=hessian_report,
            boundaries=boundary_records,
            warnings=warning_records,
            engine_metrics=metrics,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConvergenceReport:
        hessian_value = value.get("hessian")
        if hessian_value is None:
            hessian_value = {"positive_definite": value.get("hessian_positive_definite")}
        boundaries = tuple(
            item
            if isinstance(item, BoundaryRecord)
            else BoundaryRecord(
                parameter=str(item["parameter"]),
                value=float(item["value"]),
                boundary=str(item.get("boundary", "zero")),
                tolerance=item.get("tolerance"),
            )
            for item in value.get("boundaries", ())
        )
        warnings = tuple(_warning_from_value(item) for item in value.get("warnings", ()))
        known = {
            "status",
            "optimizer_terminated",
            "optimizer_message",
            "iterations",
            "objective_evaluations",
            "gradient_evaluations",
            "scaled_gradient_inf_norm",
            "parameter_step_norm",
            "hessian",
            "boundaries",
            "conditional_mode_failures",
            "ode_failures",
            "warnings",
            "engine_metrics",
        }
        extra = {key: item for key, item in value.items() if key not in known}
        engine_metrics = dict(value.get("engine_metrics", {}))
        engine_metrics.update(extra)
        optimizer_terminated = bool(value.get("optimizer_terminated", value.get("success", False)))
        status_value = value.get("status")
        if status_value is None:
            if not optimizer_terminated:
                status_value = "failed"
            elif warnings or value.get("singular") or value.get("boundary_parameters"):
                status_value = "warning"
            else:
                status_value = "converged"
        return cls(
            status=str(status_value),
            optimizer_terminated=optimizer_terminated,
            optimizer_message=str(value.get("optimizer_message", value.get("message", ""))),
            iterations=value.get("iterations"),
            objective_evaluations=value.get(
                "objective_evaluations", value.get("function_evaluations")
            ),
            gradient_evaluations=value.get("gradient_evaluations"),
            scaled_gradient_inf_norm=value.get("scaled_gradient_inf_norm"),
            parameter_step_norm=value.get("parameter_step_norm"),
            hessian=(
                hessian_value
                if isinstance(hessian_value, HessianDiagnostics)
                else HessianDiagnostics(**dict(hessian_value))
            ),
            boundaries=boundaries,
            conditional_mode_failures=int(value.get("conditional_mode_failures", 0)),
            ode_failures=int(value.get("ode_failures", 0)),
            warnings=warnings,
            engine_metrics=engine_metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "optimizer_terminated": self.optimizer_terminated,
            "optimizer_message": self.optimizer_message,
            "iterations": self.iterations,
            "objective_evaluations": self.objective_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "scaled_gradient_inf_norm": self.scaled_gradient_inf_norm,
            "parameter_step_norm": self.parameter_step_norm,
            "hessian": self.hessian.to_dict(),
            "boundaries": [item.to_dict() for item in self.boundaries],
            "conditional_mode_failures": self.conditional_mode_failures,
            "ode_failures": self.ode_failures,
            "warnings": [item.to_dict() for item in self.warnings],
            "engine_metrics": dict(self.engine_metrics),
        }
