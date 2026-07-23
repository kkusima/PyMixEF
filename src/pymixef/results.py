"""Stable backend-neutral fit result and archival format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._contracts import WarningRecord
from ._serialization import read_json, stable_hash, write_json
from .convergence import ConvergenceReport
from .diagnostics import DiagnosticTable, residual_table, vpc_table
from .errors import UnsupportedCapabilityError, ValidationError
from .provenance import RunManifest, fingerprint_model_ir
from .random import RandomStreamManager

_RESULT_SCHEMA_VERSION = "1.0.0"


def _as_array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional array; got shape {array.shape}.")
    return array


@dataclass(slots=True)
class FitResult:
    """Result contract shared by frequentist, stochastic, and Bayesian engines."""

    model_ir: Any
    parameters: Mapping[str, float]
    unconstrained_parameters: Mapping[str, float]
    parameter_covariance: np.ndarray | None
    fitted_values: np.ndarray
    residuals: np.ndarray
    random_effects: Mapping[str, Any]
    objective: float
    log_likelihood: float | None
    method: str
    engine: str
    convergence: ConvergenceReport
    manifest: RunManifest
    warnings: tuple[WarningRecord, ...] = ()
    diagnostic_data: Mapping[str, DiagnosticTable] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    result_schema_version: str = _RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.parameters = {str(key): float(value) for key, value in self.parameters.items()}
        self.unconstrained_parameters = {
            str(key): float(value) for key, value in self.unconstrained_parameters.items()
        }
        self.parameter_covariance = (
            None
            if self.parameter_covariance is None
            else _as_array(self.parameter_covariance, ndim=2)
        )
        self.fitted_values = _as_array(self.fitted_values, ndim=1)
        self.residuals = _as_array(self.residuals, ndim=1)
        if self.fitted_values.shape != self.residuals.shape:
            raise ValueError("Fitted values and residuals must align.")
        self.warnings = tuple(self.warnings)
        self.diagnostic_data = {
            str(name): (
                table if isinstance(table, DiagnosticTable) else DiagnosticTable.from_dict(table)
            )
            for name, table in self.diagnostic_data.items()
        }

    @property
    def success(self) -> bool:
        """A compatibility convenience; inspect ``convergence`` for real detail."""

        return self.convergence.trustworthy

    @property
    def n_observations(self) -> int:
        return int(self.fitted_values.size)

    def prediction(self, *, mode: str = "conditional") -> np.ndarray:
        """Return an explicitly named prediction mode for the analysis rows."""

        if mode == "conditional":
            return self.fitted_values.copy()
        if mode == "population":
            population = self.extra.get("population_fitted_values")
            if population is None:
                if not self.random_effects:
                    return self.fitted_values.copy()
                raise ValueError("Population predictions were not archived by this backend.")
            return np.asarray(population, dtype=float).copy()
        if mode == "new-subject":
            population = self.extra.get("population_fitted_values")
            if population is None:
                raise ValueError("New-subject predictions require archived population predictions.")
            return np.asarray(population, dtype=float).copy()
        raise ValueError("Prediction mode must be conditional, population, or new-subject.")

    def diagnostic(self, name: str) -> DiagnosticTable:
        try:
            return self.diagnostic_data[name]
        except KeyError as error:
            raise KeyError(
                f"Diagnostic table {name!r} is unavailable; choose from "
                f"{sorted(self.diagnostic_data)}."
            ) from error

    def residual_diagnostics(
        self,
        *,
        observed: Sequence[float] | None = None,
        variance: Sequence[float] | float | None = None,
    ) -> DiagnosticTable:
        if observed is None:
            observed_array = self.fitted_values + self.residuals
        else:
            observed_array = np.asarray(observed, dtype=float)
        return residual_table(
            observed_array,
            self.fitted_values,
            variance=variance,
            row_ids=self.extra.get("row_ids"),
            groups=self.extra.get("groups"),
        )

    def simulate(
        self,
        *,
        n_replicates: int = 1,
        seed: int | None = None,
        parameter_uncertainty: str = "none",
        random_effects: bool = True,
        residual_error: bool = True,
        output: str = "numpy",
        design: Any = None,
    ) -> np.ndarray | DiagnosticTable:
        """Simulate from archived Gaussian calculations or a backend simulator.

        The arguments intentionally mirror the blueprint.  A backend may archive a
        callable simulator for an in-memory result, but archival reloads use the
        standardized Gaussian fallback only when its assumptions are explicit.
        """

        if n_replicates < 1:
            raise ValueError("n_replicates must be positive.")
        if parameter_uncertainty not in {"none", "asymptotic"}:
            raise ValueError("parameter_uncertainty must be 'none' or 'asymptotic'.")
        simulator = self.extra.get("_simulator")
        if callable(simulator):
            values = np.asarray(
                simulator(
                    design=design,
                    n_replicates=n_replicates,
                    seed=seed,
                    parameter_uncertainty=parameter_uncertainty,
                    random_effects=random_effects,
                    residual_error=residual_error,
                )
            )
        else:
            if design is not None:
                raise UnsupportedCapabilityError(
                    "Portable simulation for a new design was not archived by this backend.",
                    code="SIM-DESIGN-UNSUPPORTED-001",
                    remediation="Refit and simulate through a backend with a portable design compiler.",
                )
            family = str(self.extra.get("family", "gaussian")).lower()
            if family not in {"gaussian", "normal"}:
                raise NotImplementedError(
                    "This reloaded result has no archived simulator for family "
                    f"{family!r}; refit with a backend supporting portable simulation."
                )
            rng = RandomStreamManager(
                0 if seed is None else int(seed), "pymixef-result-simulation"
            ).generator("observations")
            archived_population = self.extra.get("population_fitted_values")
            if archived_population is None:
                if self.random_effects:
                    raise UnsupportedCapabilityError(
                        "This result did not archive population predictions required "
                        "for new-replicate simulation.",
                        code="SIM-RANDOM-EFFECTS-UNSUPPORTED-001",
                    )
                mean = self.fitted_values.copy()
            else:
                mean = np.asarray(archived_population, dtype=float).reshape(-1)
            if mean.shape != self.fitted_values.shape:
                raise ValidationError(
                    "Archived population predictions do not align with analysis rows.",
                    code="RESULT-SIMULATION-SHAPE-001",
                )
            values = np.broadcast_to(mean, (n_replicates, mean.size)).copy()

            simulation_covariance = np.zeros((mean.size, mean.size), dtype=float)
            if random_effects:
                random_covariance = self.extra.get("random_effect_observation_covariance")
                if random_covariance is None and self.random_effects:
                    raise UnsupportedCapabilityError(
                        "Random-effect simulation covariance was not archived.",
                        code="SIM-RANDOM-EFFECTS-UNSUPPORTED-001",
                    )
                if random_covariance is not None:
                    simulation_covariance += _as_array(random_covariance, ndim=2)
            if residual_error:
                residual_covariance = self.extra.get("residual_covariance")
                if residual_covariance is None:
                    scale = float(
                        self.extra.get(
                            "residual_scale",
                            (
                                np.sqrt(np.mean(np.square(self.residuals)))
                                if self.residuals.size
                                else 0.0
                            ),
                        )
                    )
                    simulation_covariance += np.eye(mean.size) * scale**2
                else:
                    simulation_covariance += _as_array(residual_covariance, ndim=2)
            if simulation_covariance.shape != (mean.size, mean.size):
                raise ValidationError(
                    "Archived simulation covariance does not align with analysis rows.",
                    code="RESULT-SIMULATION-SHAPE-001",
                )
            if np.any(simulation_covariance):
                simulation_covariance = (simulation_covariance + simulation_covariance.T) / 2.0
                values += rng.multivariate_normal(
                    np.zeros(mean.size),
                    simulation_covariance,
                    size=n_replicates,
                    check_valid="raise",
                )
            if parameter_uncertainty == "asymptotic":
                fixed_design = self.extra.get("fixed_design")
                fixed_covariance = self.extra.get("fixed_effect_covariance")
                if fixed_design is None or fixed_covariance is None:
                    raise UnsupportedCapabilityError(
                        "Asymptotic prediction uncertainty requires archived fixed-"
                        "effect design and covariance matrices.",
                        code="SIM-PARAMETER-UNCERTAINTY-UNSUPPORTED-001",
                    )
                design_matrix = _as_array(fixed_design, ndim=2)
                covariance = _as_array(fixed_covariance, ndim=2)
                if design_matrix.shape[0] != mean.size or covariance.shape != (
                    design_matrix.shape[1],
                    design_matrix.shape[1],
                ):
                    raise ValidationError(
                        "Archived fixed-effect uncertainty matrices are inconsistent.",
                        code="RESULT-SIMULATION-SHAPE-001",
                    )
                draws = rng.multivariate_normal(
                    np.zeros(design_matrix.shape[1]),
                    covariance,
                    n_replicates,
                    check_valid="raise",
                )
                values += draws @ design_matrix.T
        if output == "numpy":
            return values
        if output == "table":
            replicate, observation = np.indices(values.shape)
            return DiagnosticTable(
                name="simulation",
                columns={
                    "replicate": replicate.ravel(),
                    "observation": observation.ravel(),
                    "value": values.ravel(),
                },
                metadata={
                    "seed": seed,
                    "parameter_uncertainty": parameter_uncertainty,
                    "random_effects": random_effects,
                    "residual_error": residual_error,
                    "design": "analysis-design" if design is None else "provided",
                    "simulation_mode": "new-replicate",
                },
            )
        raise ValueError("output must be 'numpy' or 'table'.")

    def vpc(
        self,
        *,
        data: Sequence[float] | None = None,
        independent: Sequence[float] | None = None,
        bins: str | int | Sequence[float] = "adaptive",
        prediction_corrected: bool = False,
        simulations: int = 1000,
        seed: int | None = None,
    ) -> DiagnosticTable:
        observed = (
            self.fitted_values + self.residuals if data is None else np.asarray(data, dtype=float)
        )
        simulated = self.simulate(n_replicates=simulations, seed=seed)
        assert isinstance(simulated, np.ndarray)
        return vpc_table(
            observed,
            simulated,
            independent=independent,
            bins=bins,
            prediction_corrected=prediction_corrected,
            seed=seed,
        )

    def to_dict(self) -> dict[str, Any]:
        extra = {key: value for key, value in self.extra.items() if not callable(value)}
        return {
            "result_schema_version": self.result_schema_version,
            "model_ir": (
                self.model_ir.to_dict() if hasattr(self.model_ir, "to_dict") else self.model_ir
            ),
            "parameters": dict(self.parameters),
            "unconstrained_parameters": dict(self.unconstrained_parameters),
            "parameter_covariance": self.parameter_covariance,
            "fitted_values": self.fitted_values,
            "residuals": self.residuals,
            "random_effects": dict(self.random_effects),
            "objective": self.objective,
            "log_likelihood": self.log_likelihood,
            "method": self.method,
            "engine": self.engine,
            "convergence": self.convergence.to_dict(),
            "manifest": self.manifest.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "diagnostic_data": {
                name: table.to_dict() for name, table in self.diagnostic_data.items()
            },
            "extra": extra,
        }

    def save(self, path: str | Path) -> Path:
        """Save the full result as versioned JSON; never pickle."""

        destination = write_json(path, self.to_dict())
        digest_path = destination.with_suffix(destination.suffix + ".sha256")
        digest_path.write_text(stable_hash(self.to_dict()) + "\n", encoding="ascii")
        return destination

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FitResult:
        version = str(value.get("result_schema_version", ""))
        if version != _RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported result schema {version!r}; this release supports "
                f"{_RESULT_SCHEMA_VERSION!r}."
            )
        return cls(
            model_ir=value["model_ir"],
            parameters=value.get("parameters", {}),
            unconstrained_parameters=value.get("unconstrained_parameters", {}),
            parameter_covariance=value.get("parameter_covariance"),
            fitted_values=np.asarray(value.get("fitted_values", []), dtype=float),
            residuals=np.asarray(value.get("residuals", []), dtype=float),
            random_effects=value.get("random_effects", {}),
            objective=float(value["objective"]),
            log_likelihood=(
                None if value.get("log_likelihood") is None else float(value["log_likelihood"])
            ),
            method=str(value["method"]),
            engine=str(value["engine"]),
            convergence=ConvergenceReport.from_dict(value["convergence"]),
            manifest=RunManifest.from_dict(value["manifest"]),
            warnings=tuple(
                WarningRecord(
                    code=str(item["code"]),
                    severity=str(item["severity"]),
                    message=str(item["message"]),
                    component=item.get("component"),
                    remediation=item.get("remediation"),
                    details=item.get("details", {}),
                )
                for item in value.get("warnings", ())
            ),
            diagnostic_data={
                name: DiagnosticTable.from_dict(table)
                for name, table in value.get("diagnostic_data", {}).items()
            },
            extra=value.get("extra", {}),
            result_schema_version=version,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_integrity: bool = True,
        require_sidecar: bool = False,
    ) -> FitResult:
        """Load an archived result and verify its hash sidecar when available.

        Legacy or externally produced JSON may omit the sidecar. Set
        ``require_sidecar=True`` when the calling workflow requires an integrity
        record. Integrity verification can be disabled only explicitly.
        """

        source = Path(path)
        value = read_json(source)
        digest_path = source.with_suffix(source.suffix + ".sha256")
        if require_sidecar and not digest_path.is_file():
            raise ValidationError(
                f"Required result integrity sidecar is missing: {digest_path}.",
                code="RESULT-INTEGRITY-MISSING-001",
                remediation="Restore the .sha256 sidecar or load with require_sidecar=False.",
            )
        if verify_integrity and digest_path.is_file():
            expected = digest_path.read_text(encoding="ascii").strip()
            observed = stable_hash(value)
            if expected != observed:
                raise ValidationError(
                    "Archived result does not match its integrity sidecar.",
                    code="RESULT-INTEGRITY-001",
                    remediation="Restore an untampered result and matching .sha256 sidecar.",
                    details={"expected": expected, "observed": observed},
                )
        result = cls.from_dict(value)
        if verify_integrity:
            try:
                observed_model_ir_hash = fingerprint_model_ir(result.model_ir)
            except (TypeError, ValueError) as error:
                raise ValidationError(
                    "Archived result contains an invalid model IR payload.",
                    code="RESULT-MODEL-IR-INTEGRITY-001",
                    remediation="Restore an untampered result with a valid model IR mapping.",
                    details={"error": str(error)},
                ) from error
            expected_model_ir_hash = result.manifest.model_ir_hash
            if (
                not isinstance(expected_model_ir_hash, str)
                or expected_model_ir_hash != observed_model_ir_hash
            ):
                raise ValidationError(
                    "Archived model IR does not match the run manifest.",
                    code="RESULT-MODEL-IR-INTEGRITY-001",
                    remediation="Restore an untampered model IR and matching run manifest.",
                    details={
                        "expected": expected_model_ir_hash,
                        "observed": observed_model_ir_hash,
                    },
                )
        if verify_integrity and result.manifest.output_hashes:
            output_values = {
                "parameters": result.parameters,
                "objective": result.objective,
                "fitted_values": result.fitted_values,
                "residuals": result.residuals,
            }
            failures = {
                name: {
                    "expected": expected,
                    "observed": (
                        stable_hash(output_values[name])
                        if name in output_values
                        else "<unknown output>"
                    ),
                }
                for name, expected in result.manifest.output_hashes.items()
                if name not in output_values or stable_hash(output_values[name]) != expected
            }
            if failures:
                raise ValidationError(
                    "Archived result outputs do not match the run manifest.",
                    code="RESULT-MANIFEST-INTEGRITY-001",
                    details={"failures": failures},
                )
        return result

    def summary(self) -> str:
        """Return a concise text summary separating estimates and convergence."""

        width = max((len(name) for name in self.parameters), default=9)
        parameter_lines = [
            f"  {name:<{width}}  {value: .8g}" for name, value in self.parameters.items()
        ]
        lines = [
            f"PyMixEF fit ({self.engine}, {self.method})",
            f"Observations: {self.n_observations}",
            f"Objective: {self.objective:.10g}",
            (
                "Log likelihood: unavailable"
                if self.log_likelihood is None
                else f"Log likelihood: {self.log_likelihood:.10g}"
            ),
            f"Convergence: {self.convergence.status}",
            "Parameters:",
            *(parameter_lines or ["  <none>"]),
        ]
        if self.convergence.warnings:
            lines.append("Warnings:")
            lines.extend(
                f"  [{warning.code}] {warning.message}" for warning in self.convergence.warnings
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
