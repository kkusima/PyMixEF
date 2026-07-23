from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymixef._contracts import WarningRecord
from pymixef._serialization import read_json, stable_hash, write_json
from pymixef.convergence import ConvergenceReport, HessianDiagnostics
from pymixef.diagnostics import residual_table, vpc_table
from pymixef.errors import UnsupportedCapabilityError, ValidationError
from pymixef.provenance import RunManifest
from pymixef.results import FitResult


def _result() -> FitResult:
    manifest = RunManifest.capture(
        model_ir={"schema_version": "1.0.0", "response": "y"},
        data={"y": [1.0, 2.0], "x": [0.0, 1.0]},
        engine="lmm",
        method="reml",
    )
    result = FitResult(
        model_ir={"schema_version": "1.0.0", "response": "y"},
        parameters={"beta[Intercept]": 1.5, "residual_sd": 0.5},
        unconstrained_parameters={"beta[Intercept]": 1.5, "log_residual_sd": -0.693},
        parameter_covariance=np.eye(2) * 0.1,
        fitted_values=np.array([1.5, 1.5]),
        residuals=np.array([-0.5, 0.5]),
        random_effects={},
        objective=2.0,
        log_likelihood=-2.0,
        method="reml",
        engine="lmm",
        convergence=ConvergenceReport(
            status="converged",
            optimizer_terminated=True,
            hessian=HessianDiagnostics(positive_definite=True),
        ),
        manifest=manifest,
        warnings=(WarningRecord("TEST-001", "info", "Test-only warning."),),
        extra={
            "family": "gaussian",
            "residual_scale": 0.5,
            "objective_convention": "negative normalized REML log likelihood",
            "likelihood_includes_data_constants": True,
        },
    )
    result.manifest = manifest.with_outputs(
        {
            "parameters": result.parameters,
            "objective": result.objective,
            "fitted_values": result.fitted_values,
            "residuals": result.residuals,
        }
    )
    return result


def test_result_round_trip(tmp_path: Path) -> None:
    source = _result()
    path = source.save(tmp_path / "fit.json")
    loaded = FitResult.load(path)
    assert loaded.parameters == source.parameters
    np.testing.assert_allclose(loaded.fitted_values, source.fitted_values)
    assert loaded.manifest.model_ir_hash == source.manifest.model_ir_hash
    assert path.with_suffix(".json.sha256").exists()


def test_result_load_verifies_integrity_sidecar(tmp_path: Path) -> None:
    path = _result().save(tmp_path / "fit.json")
    path.write_text(path.read_text().replace('"objective": 2.0', '"objective": 3.0'))
    with pytest.raises(ValidationError, match="integrity sidecar") as captured:
        FitResult.load(path)
    assert captured.value.code == "RESULT-INTEGRITY-001"


def test_result_load_can_require_integrity_sidecar(tmp_path: Path) -> None:
    path = _result().save(tmp_path / "fit.json")
    path.with_suffix(".json.sha256").unlink()
    with pytest.raises(ValidationError) as captured:
        FitResult.load(path, require_sidecar=True)
    assert captured.value.code == "RESULT-INTEGRITY-MISSING-001"


def test_result_load_verifies_manifest_output_hashes(tmp_path: Path) -> None:
    path = _result().save(tmp_path / "fit.json")
    path.write_text(
        path.read_text(encoding="utf-8").replace('"objective": 2.0', '"objective": 3.0'),
        encoding="utf-8",
    )
    path.with_suffix(".json.sha256").write_text(
        stable_hash(read_json(path)) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValidationError) as captured:
        FitResult.load(path)
    assert captured.value.code == "RESULT-MANIFEST-INTEGRITY-001"


def test_result_load_verifies_manifest_model_ir_hash(tmp_path: Path) -> None:
    path = _result().save(tmp_path / "fit.json")
    payload = read_json(path)
    payload["model_ir"]["response"] = "tampered-response"
    write_json(path, payload)
    path.with_suffix(".json.sha256").write_text(
        stable_hash(read_json(path)) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValidationError) as captured:
        FitResult.load(path)

    assert captured.value.code == "RESULT-MODEL-IR-INTEGRITY-001"
    assert captured.value.details["expected"] != captured.value.details["observed"]


def test_diagnostic_tables_and_vpc() -> None:
    residuals = residual_table([1, 3], [2, 2], variance=4.0)
    np.testing.assert_allclose(residuals.columns["pearson_residual"], [-0.5, 0.5])
    sims = np.array([[0.9, 2.9], [1.1, 3.1], [1.0, 3.0]])
    vpc = vpc_table([1, 3], sims, bins=1, seed=42)
    assert vpc.metadata["seed"] == 42
    assert len(vpc) == 3


def test_gaussian_simulation_is_seeded() -> None:
    fit = _result()
    first = fit.simulate(n_replicates=3, seed=123)
    second = fit.simulate(n_replicates=3, seed=123)
    np.testing.assert_array_equal(first, second)


def test_portable_simulation_refuses_unarchived_new_design() -> None:
    with pytest.raises(UnsupportedCapabilityError) as captured:
        _result().simulate(design={"x": [0.0, 1.0]})
    assert captured.value.code == "SIM-DESIGN-UNSUPPORTED-001"


def test_asymptotic_simulation_uses_full_fixed_design() -> None:
    fit = _result()
    fit.extra = {
        **fit.extra,
        "fixed_design": [[1.0, 0.0], [1.0, 2.0]],
        "fixed_effect_covariance": [[0.1, 0.02], [0.02, 0.2]],
    }
    draws = fit.simulate(
        n_replicates=4,
        seed=3,
        parameter_uncertainty="asymptotic",
        residual_error=False,
    )
    assert isinstance(draws, np.ndarray)
    assert draws.shape == (4, 2)
    assert not np.allclose(draws[:, 0] - draws[:, 1], 0.0)
