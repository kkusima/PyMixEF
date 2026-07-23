from __future__ import annotations

import numpy as np
import pytest

from pymixef.backends.base import convergence_mapping
from pymixef.backends.lmm import fit_lmm
from pymixef.covariance import AR1


def _random_intercept_data(seed: int = 12) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(18), 6)
    x = rng.normal(size=groups.size)
    X = np.column_stack([np.ones(groups.size), x])
    random_intercepts = rng.normal(0, 0.8, 18)
    y = X @ np.array([1.5, -0.4]) + random_intercepts[groups] + rng.normal(0, 0.45, groups.size)
    return {
        "response": y,
        "fixed": X,
        "fixed_names": ("Intercept", "x"),
        "random_blocks": (
            {
                "matrix": np.ones((groups.size, 1)),
                "groups": groups,
                "term_names": ("Intercept",),
                "name": "subject",
            },
        ),
    }


@pytest.mark.parametrize("method", ["ML", "REML"])
def test_dense_lmm_recovers_random_intercept_model(method: str) -> None:
    data = _random_intercept_data()
    result = fit_lmm(data, method=method, maxiter=400)
    assert result["method"] == method
    assert result["extra"]["likelihood_includes_data_constants"]
    assert result["parameters"]["Intercept"] == pytest.approx(1.5, abs=0.3)
    assert result["parameters"]["x"] == pytest.approx(-0.4, abs=0.15)
    assert result["parameters"]["sd(subject:Intercept)"] > 0.2
    assert result["parameters"]["residual_sd"] == pytest.approx(0.45, abs=0.15)
    assert result["random_effects"].shape == (18,)
    assert result["fitted_values"].shape == np.asarray(data["response"]).shape
    assert result["convergence"]["optimizer_terminated"]


def test_multiple_crossed_random_blocks() -> None:
    rng = np.random.default_rng(8)
    subjects = np.repeat(np.arange(10), 8)
    items = np.tile(np.arange(8), 10)
    X = np.ones((subjects.size, 1))
    y = (
        2
        + rng.normal(0, 0.7, 10)[subjects]
        + rng.normal(0, 0.4, 8)[items]
        + rng.normal(0, 0.3, subjects.size)
    )
    result = fit_lmm(
        {
            "y": y,
            "X": X,
            "fixed_names": ["Intercept"],
            "random_blocks": [
                {"Z": np.ones_like(X), "groups": subjects, "name": "subject"},
                {"Z": np.ones_like(X), "groups": items, "name": "item"},
            ],
        },
        method="ML",
        maxiter=500,
    )
    assert result["parameters"]["sd(subject:intercept)"] > 0
    assert result["parameters"]["sd(item:intercept)"] > 0
    assert result["random_effects"].size == 18
    assert np.isfinite(result["log_likelihood"])


def test_rank_deficiency_is_reported_without_crashing() -> None:
    data = _random_intercept_data()
    X = np.asarray(data["fixed"])
    data["fixed"] = np.column_stack([X, X[:, 1]])
    data["fixed_names"] = ("Intercept", "x", "duplicate_x")
    result = fit_lmm(data, method="REML", maxiter=300)
    assert result["convergence"]["fixed_effect_rank"] == 2
    assert result["convergence"]["status"] == "warning"
    assert result["extra"]["aliased_coefficients"]


def test_structured_residual_covariance_object() -> None:
    rng = np.random.default_rng(22)
    subjects = np.repeat(np.arange(16), 3)
    visits = np.tile(np.arange(3), 16)
    X = np.column_stack([np.ones(subjects.size), visits])
    covariance = 0.9**2 * 0.55 ** np.abs(np.subtract.outer(np.arange(3), np.arange(3)))
    errors = np.concatenate([rng.multivariate_normal(np.zeros(3), covariance) for _ in range(16)])
    y = X @ np.array([1.0, 0.25]) + errors
    result = fit_lmm(
        {
            "y": y,
            "X": X,
            "fixed_names": ("Intercept", "visit"),
            "residual_covariance": AR1(dimension=3, index="visit", group="subject"),
            "subjects": subjects,
            "visits": visits,
        },
        method="REML",
    )
    assert result["extra"]["residual_covariance_mode"] == "structured"
    assert result["parameters"]["residual_sd"] > 0
    assert -1 < result["parameters"]["residual_correlation"] < 1
    assert np.all(np.linalg.eigvalsh(result["extra"]["residual_covariance"]) > 0)


def test_numerically_suspect_convergence_has_coded_warnings() -> None:
    report = convergence_mapping(
        success=True,
        message="optimizer stopped",
        iterations=2,
        function_evaluations=4,
        gradient=[0.5],
        hessian_positive_definite=False,
        singular=False,
    )
    assert report["status"] == "warning"
    codes = {item["code"] for item in report["warnings"]}
    assert {"NUM-GRADIENT-LARGE-001", "NUM-HESSIAN-INDEFINITE-001"} <= codes
