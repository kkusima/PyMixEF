from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from pymixef.backends import BUILTIN_BACKENDS, Backend
from pymixef.backends.base import BackendNumericalError, validate_payload
from pymixef.backends.glmm import LaplaceGLMMBackend
from pymixef.backends.lmm import GaussianLMMBackend
from pymixef.backends.mmrm import MMRMBackend
from pymixef.families import Bernoulli


@dataclass(frozen=True)
class BackendCase:
    data: dict[str, Any]
    options: dict[str, Any]


def _lmm_case() -> BackendCase:
    rng = np.random.default_rng(101)
    groups = np.repeat(np.arange(8), 4)
    x = rng.normal(size=groups.size)
    random_intercept = rng.normal(0, 0.45, 8)[groups]
    y = 1.2 - 0.3 * x + random_intercept + rng.normal(0, 0.25, groups.size)
    return BackendCase(
        {
            "y": y,
            "X": np.column_stack([np.ones(y.size), x]),
            "fixed_names": ["Intercept", "x"],
            "random_blocks": [{"Z": np.ones((y.size, 1)), "groups": groups, "name": "subject"}],
        },
        {"method": "ML", "compute_hessian": False, "maxiter": 250},
    )


def _glmm_case() -> BackendCase:
    rng = np.random.default_rng(202)
    groups = np.repeat(np.arange(9), 5)
    x = rng.normal(size=groups.size)
    eta = -0.4 + 0.55 * x + rng.normal(0, 0.35, 9)[groups]
    y = rng.binomial(1, expit(eta))
    return BackendCase(
        {
            "y": y,
            "X": np.column_stack([np.ones(y.size), x]),
            "fixed_names": ["Intercept", "x"],
            "family": Bernoulli(),
            "random_blocks": [{"Z": np.ones((y.size, 1)), "groups": groups, "name": "subject"}],
        },
        {"compute_hessian": False, "maxiter": 250},
    )


def _mmrm_case() -> BackendCase:
    rng = np.random.default_rng(303)
    subjects = np.repeat(np.arange(12), 3)
    visits = np.tile(np.arange(3), 12)
    treatment = np.repeat(np.tile([0, 1], 6), 3)
    x = np.column_stack([np.ones(subjects.size), treatment, visits])
    covariance = 0.7 ** np.abs(np.subtract.outer(np.arange(3), np.arange(3)))
    errors = np.concatenate([rng.multivariate_normal(np.zeros(3), covariance) for _ in range(12)])
    y = x @ np.array([1.0, 0.2, 0.3]) + errors
    return BackendCase(
        {
            "y": y,
            "X": x,
            "fixed_names": ["Intercept", "treatment", "visit"],
            "subject": subjects,
            "visit": visits,
        },
        {
            "covariance": "ar1",
            "df_method": "residual",
            "compute_hessian": False,
            "maxiter": 250,
        },
    )


CONFORMANCE_CASES = {
    GaussianLMMBackend: _lmm_case,
    LaplaceGLMMBackend: _glmm_case,
    MMRMBackend: _mmrm_case,
}


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            _assert_nested_equal(actual[name], expected[name])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
        return
    if hasattr(expected, "__dict__"):
        assert type(actual) is type(expected)
        _assert_nested_equal(vars(actual), vars(expected))
        return
    assert actual == expected


def test_every_builtin_backend_has_a_reusable_conformance_case() -> None:
    assert set(BUILTIN_BACKENDS.values()) == set(CONFORMANCE_CASES)


@pytest.mark.parametrize("backend_type", CONFORMANCE_CASES)
def test_builtin_backend_conforms_to_shared_contract(backend_type: type[Any]) -> None:
    case = CONFORMANCE_CASES[backend_type]()
    original = deepcopy(case.data)
    backend = backend_type()
    assert isinstance(backend, Backend)

    first = backend.fit(case.data, **case.options)
    second = backend.fit(case.data, **case.options)
    canonical = validate_payload(first)

    assert tuple(canonical) == (
        "parameters",
        "unconstrained_parameters",
        "parameter_covariance",
        "fitted_values",
        "residuals",
        "random_effects",
        "objective",
        "log_likelihood",
        "method",
        "engine",
        "convergence",
        "diagnostic_data",
        "extra",
    )
    assert len(first["fitted_values"]) == len(np.asarray(case.data["y"]))
    np.testing.assert_allclose(
        np.asarray(case.data["y"]) - first["fitted_values"],
        first["residuals"],
        rtol=1e-8,
        atol=1e-10,
    )
    assert first["parameters"].keys() == second["parameters"].keys()
    np.testing.assert_allclose(
        list(first["parameters"].values()),
        list(second["parameters"].values()),
        rtol=1e-10,
        atol=1e-12,
    )
    assert first["objective"] == pytest.approx(second["objective"], rel=1e-10, abs=1e-12)
    _assert_nested_equal(case.data, original)


def test_payload_validator_rejects_contract_violations() -> None:
    case = _lmm_case()
    payload = GaussianLMMBackend().fit(case.data, **case.options)

    missing = dict(payload)
    del missing["residuals"]
    with pytest.raises(BackendNumericalError, match="omitted payload fields"):
        validate_payload(missing)

    misaligned = dict(payload)
    misaligned["residuals"] = np.asarray(payload["residuals"])[:-1]
    with pytest.raises(BackendNumericalError, match="aligned"):
        validate_payload(misaligned)

    nonsymmetric = dict(payload)
    nonsymmetric["parameter_covariance"] = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(BackendNumericalError, match="symmetric"):
        validate_payload(nonsymmetric)
