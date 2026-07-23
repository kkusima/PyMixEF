"""Cross-platform parameter and objective comparison."""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._contracts import CompatibilityIssue
from ._serialization import read_json, write_json
from .diagnostics import DiagnosticTable
from .interoperability.base import CompatibilityReport
from .results import FitResult


def _reference_values(reference: Any) -> tuple[dict[str, float], float | None]:
    if isinstance(reference, FitResult):
        return dict(reference.parameters), reference.objective
    if isinstance(reference, (str, Path)):
        path = Path(reference)
        if path.is_dir():
            candidates = (
                path / "result.json",
                path / "parameters.json",
                path / "reference.json",
            )
            existing = next((candidate for candidate in candidates if candidate.exists()), None)
            if existing is None:
                raise FileNotFoundError(
                    "Reference directory must contain result.json, parameters.json, "
                    "or reference.json."
                )
            value = read_json(existing)
        else:
            value = read_json(path)
    elif isinstance(reference, Mapping):
        value = reference
    else:
        raise TypeError("reference must be a FitResult, mapping, JSON file, or directory.")
    if isinstance(value, FitResult):
        return dict(value.parameters), value.objective
    if "parameters" in value:
        parameters = value["parameters"]
        objective = value.get("objective")
    else:
        parameters = value
        objective = None
    return (
        {str(name): float(number) for name, number in parameters.items()},
        None if objective is None else float(objective),
    )


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Aligned comparison plus its convention-compatibility report."""

    table: DiagnosticTable
    compatibility: CompatibilityReport
    objective_difference: float | None
    conventions: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "objective_difference": self.objective_difference,
            "conventions": dict(self.conventions),
        }

    def assert_within(self, tolerances: Mapping[str, float]) -> None:
        absolute = np.asarray(self.table.columns["absolute_difference"], dtype=float)
        relative = np.asarray(self.table.columns["relative_difference"], dtype=float)
        if "parameters_abs" in tolerances and np.any(absolute > tolerances["parameters_abs"]):
            raise AssertionError(
                f"Absolute parameter difference {np.max(absolute):.6g} exceeds "
                f"{tolerances['parameters_abs']:.6g}."
            )
        fixed_relative = tolerances.get("fixed_effects_rel", tolerances.get("parameters_rel"))
        if fixed_relative is not None and np.any(relative > fixed_relative):
            raise AssertionError(
                f"Relative parameter difference {np.max(relative):.6g} exceeds "
                f"{fixed_relative:.6g}."
            )
        if (
            "objective_abs" in tolerances
            and self.objective_difference is not None
            and abs(self.objective_difference) > tolerances["objective_abs"]
        ):
            raise AssertionError(
                f"Objective difference {self.objective_difference:.6g} exceeds "
                f"{tolerances['objective_abs']:.6g}."
            )

    def write_report(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix.lower() == ".json":
            return write_json(destination, self.to_dict())
        if destination.suffix.lower() != ".html":
            raise ValueError("Comparison reports support .json and .html.")
        rows = "\n".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(self.table.columns[column][row]))}</td>"
                for column in self.table.columns
            )
            + "</tr>"
            for row in range(len(self.table))
        )
        header = "".join(f"<th>{html.escape(column)}</th>" for column in self.table.columns)
        document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>PyMixEF comparison</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto}}
table{{border-collapse:collapse}}th,td{{padding:.4rem;border:1px solid #bbb}}</style>
<h1>PyMixEF cross-platform comparison</h1>
<p>Compatibility: {"supported" if self.compatibility.supported else "unsupported constructs present"}</p>
<p>Objective difference: {html.escape(str(self.objective_difference))}</p>
<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>
</html>
"""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8")
        return destination


@dataclass(frozen=True, slots=True)
class ApproximationSensitivityResult:
    """Cross-setting refit comparisons with explicit failure accounting."""

    table: DiagnosticTable
    fits: Mapping[str, FitResult]
    failures: tuple[Mapping[str, Any], ...]
    baseline: str
    settings: Mapping[str, Mapping[str, Any]]
    materiality: Mapping[str, float]

    @property
    def successful_scenarios(self) -> int:
        """Number of settings whose fit completed successfully."""

        return len(self.fits)

    @property
    def failed_scenarios(self) -> int:
        """Number of settings whose fit failed or returned unusable output."""

        return len(self.failures)

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison and scenario metadata without duplicating fits."""

        return {
            "table": self.table.to_dict(),
            "failures": [dict(item) for item in self.failures],
            "baseline": self.baseline,
            "settings": {name: dict(value) for name, value in self.settings.items()},
            "materiality": dict(self.materiality),
            "successful_scenarios": self.successful_scenarios,
            "failed_scenarios": self.failed_scenarios,
        }


def _aligned_standard_errors(
    fit: FitResult,
    parameter_names: tuple[str, ...],
    *,
    scenario: str,
) -> np.ndarray:
    covariance = fit.parameter_covariance
    if covariance is None:
        raise RuntimeError(
            f"Scenario {scenario!r} did not return parameter covariance "
            "required for standard-error sensitivity."
        )
    matrix = np.asarray(covariance, dtype=float)
    expected = (len(parameter_names), len(parameter_names))
    if matrix.shape != expected:
        raise RuntimeError(
            f"Scenario {scenario!r} parameter covariance has shape "
            f"{matrix.shape}; expected {expected}."
        )
    if np.any(~np.isfinite(matrix)) or not np.allclose(
        matrix,
        matrix.T,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"Scenario {scenario!r} parameter covariance must be finite and symmetric."
        )
    diagonal = np.diag(matrix)
    scale = max(float(np.max(np.abs(diagonal))), 1.0)
    if np.any(diagonal < -1e-10 * scale):
        raise RuntimeError(f"Scenario {scenario!r} parameter covariance has negative variances.")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1e-10 * eigenvalue_scale:
        raise RuntimeError(
            f"Scenario {scenario!r} parameter covariance must be positive semidefinite."
        )
    return np.sqrt(np.maximum(diagonal, 0.0))


def approximation_sensitivity(
    fit_function: Callable[[Mapping[str, Any]], FitResult],
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    materiality: Mapping[str, float],
    baseline: str | None = None,
) -> ApproximationSensitivityResult:
    """Refit named approximation settings and compare aligned outputs.

    ``fit_function`` receives an independent deep copy of one settings mapping
    per scenario. The baseline must succeed; other failures are retained with
    scenario, exception type, and message. Successful fits must expose the same
    parameter names and an aligned parameter covariance matrix. The tidy result
    reports parameter, standard-error, and objective changes relative to the
    baseline.

    ``materiality`` must define nonnegative thresholds named
    ``parameter_relative``, ``standard_error_relative``, and
    ``objective_absolute``. Per-row flags preserve the caller's scientific
    decision rule instead of embedding an undocumented universal threshold.

    This callback-based contract supports sensitivity analyses such as Laplace
    optimizer tolerances, MMRM covariance/degree-of-freedom choices, or ODE
    solver tolerances without embedding engine-specific assumptions.
    """

    copied_settings = {str(name): deepcopy(dict(settings)) for name, settings in scenarios.items()}
    if len(copied_settings) < 2:
        raise ValueError("Approximation sensitivity requires at least two named scenarios.")
    if len(copied_settings) != len(scenarios):
        raise ValueError("Scenario names must be unique after string normalization.")
    baseline_name = next(iter(copied_settings)) if baseline is None else str(baseline)
    if baseline_name not in copied_settings:
        raise KeyError(f"Baseline scenario {baseline_name!r} is absent.")
    required_materiality = {
        "parameter_relative",
        "standard_error_relative",
        "objective_absolute",
    }
    if set(materiality) != required_materiality:
        raise ValueError(
            "materiality must define exactly parameter_relative, "
            "standard_error_relative, and objective_absolute."
        )
    materiality_thresholds = {
        name: float(materiality[name]) for name in sorted(required_materiality)
    }
    if any(not np.isfinite(value) or value < 0 for value in materiality_thresholds.values()):
        raise ValueError("materiality thresholds must be finite and nonnegative.")

    ordered_names = (baseline_name,) + tuple(
        name for name in copied_settings if name != baseline_name
    )
    fits: dict[str, FitResult] = {}
    failures: list[dict[str, Any]] = []
    for name in ordered_names:
        try:
            fit = fit_function(deepcopy(copied_settings[name]))
            if not isinstance(fit, FitResult):
                raise TypeError("fit_function must return a FitResult")
            if fit.convergence.status == "failed":
                raise RuntimeError("fit returned failed convergence status")
            if not fit.parameters or not all(
                np.isfinite(float(value)) for value in fit.parameters.values()
            ):
                raise RuntimeError("fit returned empty or non-finite parameters")
            if not np.isfinite(float(fit.objective)):
                raise RuntimeError("fit returned a non-finite objective")
            fits[name] = fit
        except Exception as error:
            failure = {
                "scenario": name,
                "type": type(error).__name__,
                "message": str(error),
            }
            failures.append(failure)
            if name == baseline_name:
                raise RuntimeError(
                    f"Baseline scenario {baseline_name!r} failed: {error}"
                ) from error

    baseline_fit = fits[baseline_name]
    parameter_names = tuple(baseline_fit.parameters)
    baseline_values = np.asarray(
        [baseline_fit.parameters[name] for name in parameter_names],
        dtype=float,
    )
    baseline_standard_errors = _aligned_standard_errors(
        baseline_fit,
        parameter_names,
        scenario=baseline_name,
    )
    output: dict[str, list[Any]] = {
        "scenario": [],
        "parameter": [],
        "estimate": [],
        "baseline_estimate": [],
        "difference": [],
        "absolute_difference": [],
        "relative_difference": [],
        "standard_error": [],
        "baseline_standard_error": [],
        "standard_error_difference": [],
        "relative_standard_error_difference": [],
        "objective": [],
        "objective_difference": [],
        "material_parameter": [],
        "material_standard_error": [],
        "material_objective": [],
        "material": [],
    }
    for name in ordered_names:
        scenario_fit = fits.get(name)
        if scenario_fit is None:
            continue
        if set(scenario_fit.parameters) != set(parameter_names):
            failures.append(
                {
                    "scenario": name,
                    "type": "ParameterAlignmentError",
                    "message": "parameter set differs from the baseline",
                }
            )
            del fits[name]
            continue
        estimates = np.asarray(
            [scenario_fit.parameters[item] for item in parameter_names], dtype=float
        )
        try:
            standard_errors = _aligned_standard_errors(
                scenario_fit,
                parameter_names,
                scenario=name,
            )
        except (RuntimeError, ValueError) as error:
            failures.append(
                {
                    "scenario": name,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            del fits[name]
            continue
        differences = estimates - baseline_values
        relatives = np.divide(
            np.abs(differences),
            np.maximum.reduce(
                [
                    np.abs(baseline_values),
                    np.abs(estimates),
                    np.full(len(estimates), np.finfo(float).eps),
                ]
            ),
        )
        standard_error_differences = standard_errors - baseline_standard_errors
        standard_error_relatives = np.divide(
            np.abs(standard_error_differences),
            np.maximum.reduce(
                [
                    np.abs(baseline_standard_errors),
                    np.abs(standard_errors),
                    np.full(len(standard_errors), np.finfo(float).eps),
                ]
            ),
        )
        objective_difference = float(scenario_fit.objective - baseline_fit.objective)
        objective_material = (
            abs(objective_difference) > materiality_thresholds["objective_absolute"]
        )
        for position, parameter in enumerate(parameter_names):
            parameter_material = relatives[position] > materiality_thresholds["parameter_relative"]
            standard_error_material = (
                standard_error_relatives[position]
                > materiality_thresholds["standard_error_relative"]
            )
            output["scenario"].append(name)
            output["parameter"].append(parameter)
            output["estimate"].append(estimates[position])
            output["baseline_estimate"].append(baseline_values[position])
            output["difference"].append(differences[position])
            output["absolute_difference"].append(abs(differences[position]))
            output["relative_difference"].append(relatives[position])
            output["standard_error"].append(standard_errors[position])
            output["baseline_standard_error"].append(baseline_standard_errors[position])
            output["standard_error_difference"].append(standard_error_differences[position])
            output["relative_standard_error_difference"].append(standard_error_relatives[position])
            output["objective"].append(float(scenario_fit.objective))
            output["objective_difference"].append(objective_difference)
            output["material_parameter"].append(parameter_material)
            output["material_standard_error"].append(standard_error_material)
            output["material_objective"].append(objective_material)
            output["material"].append(
                parameter_material or standard_error_material or objective_material
            )

    return ApproximationSensitivityResult(
        table=DiagnosticTable(
            name="approximation_sensitivity",
            columns={name: np.asarray(values) for name, values in output.items()},
            metadata={
                "baseline": baseline_name,
                "scenario_order": list(ordered_names),
                "settings": copied_settings,
                "parameter_names": list(parameter_names),
                "materiality": materiality_thresholds,
            },
        ),
        fits=fits,
        failures=tuple(failures),
        baseline=baseline_name,
        settings=copied_settings,
        materiality=materiality_thresholds,
    )


def compare(
    fit: FitResult,
    *,
    reference: Any,
    mapping: Mapping[str, str] | None = None,
    conventions: Mapping[str, Any] | None = None,
) -> ComparisonResult:
    """Compare a fit with a reference under matched objective conventions.

    Parameter estimates are always aligned independently. Objective differences
    are calculated only when the reference declares the same method,
    normalization-constant policy, and objective convention archived by the
    PyMixEF fit.
    """

    reference_parameters, reference_objective = _reference_values(reference)
    parameter_mapping = dict(mapping or {})
    current: list[float] = []
    expected: list[float] = []
    source_names: list[str] = []
    target_names: list[str] = []
    issues: list[CompatibilityIssue] = []
    for source_name, source_value in fit.parameters.items():
        target_name = parameter_mapping.get(source_name, source_name)
        if target_name not in reference_parameters:
            issues.append(
                CompatibilityIssue(
                    source_name,
                    "unsupported",
                    f"No aligned reference parameter {target_name!r}.",
                )
            )
            continue
        source_names.append(source_name)
        target_names.append(target_name)
        current.append(float(source_value))
        expected.append(float(reference_parameters[target_name]))
        issues.append(
            CompatibilityIssue(
                source_name,
                "exact" if source_name == target_name else "transformed",
                f"Aligned with {target_name!r}.",
            )
        )
    current_array = np.asarray(current)
    expected_array = np.asarray(expected)
    difference = current_array - expected_array
    relative = np.divide(
        np.abs(difference),
        np.maximum(np.abs(expected_array), np.finfo(float).eps),
    )
    reference_conventions = dict(conventions or {})
    fit_conventions = {
        "method": fit.method.lower(),
        "objective_convention": fit.extra.get("objective_convention"),
        "likelihood_includes_data_constants": fit.extra.get("likelihood_includes_data_constants"),
    }
    required = (
        "method",
        "objective_convention",
        "likelihood_includes_data_constants",
    )
    missing_fit = tuple(key for key in required if fit_conventions.get(key) is None)
    missing_reference = tuple(key for key in required if key not in reference_conventions)
    objective_compatible = True
    if missing_fit:
        objective_compatible = False
        issues.append(
            CompatibilityIssue(
                "PyMixEF objective conventions",
                "unsupported",
                "The fit did not archive calculation-defining fields: "
                + ", ".join(missing_fit)
                + ".",
            )
        )
    if missing_reference:
        objective_compatible = False
        issues.append(
            CompatibilityIssue(
                "reference objective conventions",
                "unsupported",
                "The reference omitted calculation-defining fields: "
                + ", ".join(missing_reference)
                + ".",
            )
        )
    if not missing_fit and not missing_reference:
        mismatches = {
            key: {
                "pymixef": fit_conventions[key],
                "reference": reference_conventions[key],
            }
            for key in required
            if fit_conventions[key] != reference_conventions[key]
        }
        if mismatches:
            objective_compatible = False
            issues.append(
                CompatibilityIssue(
                    "objective conventions",
                    "unsupported",
                    f"Objective conventions differ: {mismatches}.",
                )
            )
        else:
            issues.append(
                CompatibilityIssue(
                    "objective conventions",
                    "exact",
                    "Method, objective scale, and normalization constants match.",
                )
            )
    objective_difference = (
        float(fit.objective - reference_objective)
        if reference_objective is not None and objective_compatible
        else None
    )
    convention_values = {
        "pymixef": fit_conventions,
        "reference": reference_conventions,
    }
    table = DiagnosticTable(
        name="cross_platform_parameter_comparison",
        columns={
            "pymixef_parameter": np.asarray(source_names),
            "reference_parameter": np.asarray(target_names),
            "pymixef_value": current_array,
            "reference_value": expected_array,
            "difference": difference,
            "absolute_difference": np.abs(difference),
            "relative_difference": relative,
        },
        metadata={"conventions": convention_values},
    )
    return ComparisonResult(
        table=table,
        compatibility=CompatibilityReport(
            source_format="PyMixEF result",
            target_format="reference result",
            issues=tuple(issues),
            metadata={"mapping": parameter_mapping},
        ),
        objective_difference=objective_difference,
        conventions=convention_values,
    )
