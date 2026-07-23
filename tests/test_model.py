from __future__ import annotations

import numpy as np
import pytest

import pymixef
from pymixef.errors import EngineCompatibilityError, ValidationError


def _lmm_data() -> dict[str, np.ndarray]:
    groups = np.repeat(np.arange(8), 4)
    x = np.tile(np.arange(4, dtype=float), 8)
    group_effect = np.linspace(-1.0, 1.0, 8)
    residual = np.tile(np.array([-0.1, 0.05, 0.1, -0.05]), 8)
    y = 2.0 + 0.7 * x + group_effect[groups] + residual
    return {"y": y, "x": x, "group": groups}


def test_formula_compile_and_lmm_fit() -> None:
    model = pymixef.Model.from_formula("y ~ x + (1 | group)")
    plan = model.compile(_lmm_data(), method="reml")
    assert plan.engine == "lmm"
    assert plan.model_ir.response == "y"
    assert "Data audit: 32 input, 32 analysis" in plan.explain()
    fit = plan.fit()
    assert fit.n_observations == 32
    assert fit.engine == "lmm"
    assert fit.manifest.model_ir_hash.startswith("sha256:")
    assert "Intercept" in fit.parameters
    assert fit.extra["data_audit"]["excluded_rows"] == 0
    assert fit.manifest.settings["maxiter"] == 1_000
    assert fit.manifest.settings["tolerance"] == 1e-8
    assert fit.manifest.settings["optimizer_sequence"]
    assert fit.manifest.convergence["status"] == fit.convergence.status


def test_structured_builder_uses_same_formula_semantics() -> None:
    model = pymixef.Model(
        response=pymixef.Response("y"),
        fixed=pymixef.Fixed("x"),
        random=[pymixef.Random("1", group="group")],
    )
    assert model.specification.response == "y"
    assert model.specification.random_terms[0].group == "group"


def test_unsupported_aghq_fails_before_optimization() -> None:
    model = pymixef.Model.from_formula("y ~ x + (1 | group)", family=pymixef.families.Bernoulli())
    with pytest.raises(EngineCompatibilityError, match="Laplace"):
        model.compile(_lmm_data(), engine="aghq")


@pytest.mark.parametrize("component", ["zero_inflation", "dispersion", "shape"])
def test_distributional_predictors_fail_before_optimization(component: str) -> None:
    model = pymixef.Model.from_formula(
        "y ~ x + (1 | group)",
        family=pymixef.families.Poisson(),
        **{component: "~ x"},
    )
    report = model.validate()
    assert not report.valid
    with pytest.raises(EngineCompatibilityError) as captured:
        model.compile(_lmm_data())
    assert "ENGINE-DISTRIBUTIONAL-PREDICTOR-001" in str(captured.value)


def test_unknown_fit_setting_is_not_silently_ignored() -> None:
    model = pymixef.Model.from_formula("y ~ x + (1 | group)")
    with pytest.raises(ValidationError) as captured:
        model.compile(_lmm_data(), tolernace=1e-6)
    assert captured.value.code == "ENGINE-SETTING-UNKNOWN-001"


def test_missing_mmrm_structural_key_fails_explicitly() -> None:
    data = {
        "y": [1.0, 2.0, 1.5],
        "visit": [1, 2, 1],
        "subject": ["A", None, "B"],
    }
    model = pymixef.Model.from_formula(
        "y ~ visit",
        residual=pymixef.covariance.Unstructured(index="visit", group="subject"),
    )
    with pytest.raises(ValidationError, match="missing values"):
        model.compile(data)
