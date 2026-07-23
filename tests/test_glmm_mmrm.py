from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from pymixef.backends.base import BackendInputError, BackendUnsupportedError
from pymixef.backends.glmm import fit_glmm
from pymixef.backends.mmrm import fit_mmrm
from pymixef.families import (
    Bernoulli,
    Binomial,
    NegativeBinomial2,
    Poisson,
    ZeroInflated,
)


@pytest.mark.parametrize(
    "family",
    [Bernoulli(), Binomial(), Poisson(), NegativeBinomial2(3.0)],
)
def test_laplace_glmm_supported_families(family: object) -> None:
    rng = np.random.default_rng(30)
    groups = np.repeat(np.arange(10), 7)
    x = rng.normal(size=groups.size)
    X = np.column_stack([np.ones(groups.size), x])
    eta = -0.2 + 0.45 * x + rng.normal(0, 0.5, 10)[groups]
    data: dict[str, object] = {
        "X": X,
        "fixed_names": ["Intercept", "x"],
        "family": family,
        "random_blocks": [{"Z": np.ones((groups.size, 1)), "groups": groups, "name": "subject"}],
    }
    if isinstance(family, Bernoulli):
        data["y"] = rng.binomial(1, expit(eta))
    elif isinstance(family, Binomial):
        data["trials"] = np.full(groups.size, 4)
        data["y"] = rng.binomial(4, expit(eta))
    elif isinstance(family, Poisson):
        data["y"] = rng.poisson(np.exp(eta))
    else:
        mu = np.exp(eta)
        data["y"] = rng.negative_binomial(3, 3 / (3 + mu))
    result = fit_glmm(data, maxiter=250, compute_hessian=False)
    assert result["method"] == "Laplace"
    assert result["extra"]["quadrature_order"] == 1
    assert result["extra"]["likelihood_includes_data_constants"]
    assert np.isfinite(result["log_likelihood"])
    assert result["random_effects"].size == 10
    assert result["convergence"]["conditional_mode_gradient_inf_norm"] < 1e-4
    assert result["convergence"]["fixed_effect_rank"] == 2
    assert result["extra"]["fixed_effect_rank"] == 2


def test_laplace_rejects_unimplemented_extended_family() -> None:
    data = {
        "y": np.array([0, 1, 0, 2]),
        "X": np.ones((4, 1)),
        "family": ZeroInflated(Poisson()),
        "random_blocks": [{"Z": np.ones((4, 1)), "groups": [0, 0, 1, 1]}],
    }
    with pytest.raises(BackendUnsupportedError) as error:
        fit_glmm(data)
    assert error.value.code == "ENGINE-UNSUPPORTED-001"


def _mmrm_data(seed: int = 14) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    subjects = np.repeat(np.arange(20), 3)
    visits = np.tile(np.arange(3), 20)
    treatment_subject = rng.integers(0, 2, 20)
    treatment = treatment_subject[subjects]
    X = np.column_stack(
        [
            np.ones(subjects.size),
            treatment,
            visits,
            treatment * visits,
        ]
    )
    beta = np.array([1.0, 0.25, 0.3, 0.12])
    visit_covariance = np.array([[1.0, 0.45, 0.2], [0.45, 1.2, 0.5], [0.2, 0.5, 1.5]])
    errors = np.concatenate(
        [rng.multivariate_normal(np.zeros(3), visit_covariance) for _ in range(20)]
    )
    return {
        "y": X @ beta + errors,
        "X": X,
        "fixed_names": ["Intercept", "treatment", "visit", "treatment:visit"],
        "subject": subjects,
        "visit": visits,
        "emm_matrix": {
            "matrix": [[1, 0, 2, 0], [1, 1, 2, 2]],
            "names": ["control_visit_2", "active_visit_2"],
        },
        "contrasts": {"active-control_visit_2": [0, 1, 0, 2]},
    }


@pytest.mark.parametrize(
    "structure",
    [
        "unstructured",
        "ar1",
        "heterogeneous-ar1",
        "compound-symmetry",
        "toeplitz",
        "ante-dependence",
        "spatial-power",
    ],
)
def test_mmrm_visit_covariance_structures_are_positive_definite(structure: str) -> None:
    result = fit_mmrm(
        _mmrm_data(),
        covariance=structure,
        df_method="satterthwaite",
        maxiter=400,
        compute_hessian=structure != "unstructured",
    )
    visit_covariance = result["extra"]["visit_covariance"]
    assert visit_covariance.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(visit_covariance) > 0)
    assert np.isfinite(result["log_likelihood"])
    assert result["extra"]["degrees_of_freedom_method"] == "Satterthwaite delta-method"
    assert result["extra"]["estimated_marginal_means"]["name"] == [
        "control_visit_2",
        "active_visit_2",
    ]
    contrast = result["extra"]["contrasts"]
    assert contrast["name"] == ["active-control_visit_2"]
    assert contrast["degrees_of_freedom"][0] >= 1


def test_mmrm_exact_kenward_roger_is_not_faked() -> None:
    with pytest.raises(BackendUnsupportedError, match="exact Kenward-Roger"):
        fit_mmrm(_mmrm_data(), covariance="ar1", df_method="kenward-roger")


def test_mmrm_kr_inspired_option_is_explicitly_labeled() -> None:
    result = fit_mmrm(
        _mmrm_data(),
        covariance="ar1",
        df_method="kenward-roger-approximate",
        maxiter=300,
    )
    assert "not exact Kenward-Roger" in result["extra"]["degrees_of_freedom_method"]
    warning_codes = {item["code"] for item in result["convergence"]["warnings"]}
    assert "MMRM-KR-APPROX-001" in warning_codes


def _permute_mmrm_rows(data: dict[str, object], order: np.ndarray) -> dict[str, object]:
    permuted = dict(data)
    for name in ("y", "X", "subject", "visit", "visit_times"):
        if name in permuted:
            permuted[name] = np.asarray(permuted[name])[order]
    return permuted


@pytest.mark.parametrize(
    "structure",
    [
        "ar1",
        "heterogeneous-ar1",
        "toeplitz",
        "heterogeneous-toeplitz",
        "ante-dependence",
    ],
)
def test_ordered_mmrm_covariance_is_invariant_to_input_row_permutation(
    structure: str,
) -> None:
    data = _mmrm_data()
    rng = np.random.default_rng(314)
    permuted = _permute_mmrm_rows(data, rng.permutation(len(np.asarray(data["y"]))))

    fit = fit_mmrm(
        data,
        covariance=structure,
        df_method="residual",
        compute_hessian=False,
        maxiter=400,
    )
    permuted_fit = fit_mmrm(
        permuted,
        covariance=structure,
        df_method="residual",
        compute_hessian=False,
        maxiter=400,
    )

    assert fit["extra"]["visit_levels"] == [0, 1, 2]
    assert permuted_fit["extra"]["visit_levels"] == [0, 1, 2]
    assert fit["extra"]["visit_order_source"] == "ascending-numeric-visit-labels"
    np.testing.assert_allclose(
        list(fit["parameters"].values()),
        list(permuted_fit["parameters"].values()),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        fit["extra"]["visit_covariance"],
        permuted_fit["extra"]["visit_covariance"],
        rtol=2e-5,
        atol=2e-6,
    )
    assert fit["log_likelihood"] == pytest.approx(
        permuted_fit["log_likelihood"], rel=2e-7, abs=2e-7
    )


def test_numeric_visit_times_define_string_visit_order_and_are_archived() -> None:
    data = _mmrm_data()
    numeric_visit = np.asarray(data["visit"])
    labels = np.asarray(["week 8", "baseline", "week 4"], dtype=object)[numeric_visit]
    times = np.asarray([8.0, 0.0, 4.0])[numeric_visit]
    data["visit"] = labels
    data["visit_times"] = times

    result = fit_mmrm(
        data,
        covariance="ar1",
        df_method="residual",
        compute_hessian=False,
    )

    assert result["extra"]["visit_levels"] == ["baseline", "week 4", "week 8"]
    assert result["extra"]["visit_times"] == [0.0, 4.0, 8.0]
    assert result["extra"]["visit_order_source"] == "ascending-explicit-visit-times"


def test_visit_time_mapping_defines_per_level_spatial_distances() -> None:
    data = _mmrm_data()
    numeric_visit = np.asarray(data["visit"])
    data["visit"] = np.asarray(["late", "early", "middle"], dtype=object)[numeric_visit]
    data["visit_times"] = {"early": 0.0, "middle": 3.0, "late": 10.0}

    result = fit_mmrm(
        data,
        covariance="spatial-power",
        df_method="residual",
        compute_hessian=False,
    )

    assert result["extra"]["visit_levels"] == ["early", "middle", "late"]
    assert result["extra"]["visit_times"] == [0.0, 3.0, 10.0]
    assert result["extra"]["visit_order_source"] == "ascending-explicit-visit-times"


def test_explicit_visit_order_defines_toeplitz_adjacency() -> None:
    data = _mmrm_data()
    numeric_visit = np.asarray(data["visit"])
    data["visit"] = np.asarray(["late", "early", "middle"], dtype=object)[numeric_visit]
    data["visit_order"] = ["early", "middle", "late"]

    result = fit_mmrm(
        data,
        covariance="toeplitz",
        df_method="residual",
        compute_hessian=False,
    )

    assert result["extra"]["visit_levels"] == ["early", "middle", "late"]
    assert result["extra"]["visit_order_source"] == "explicit-visit-order"
    assert result["extra"]["declared_visit_levels"] == ["early", "middle", "late"]


def test_ordered_categorical_visit_levels_are_preserved() -> None:
    pd = pytest.importorskip("pandas")
    data = _mmrm_data()
    numeric_visit = np.asarray(data["visit"])
    labels = np.asarray(["week 8", "baseline", "week 4"], dtype=object)[numeric_visit]
    data["visit"] = pd.Categorical(
        labels,
        categories=["baseline", "week 4", "week 8"],
        ordered=True,
    )

    result = fit_mmrm(
        data,
        covariance="ante-dependence",
        df_method="residual",
        compute_hessian=False,
    )

    assert result["extra"]["visit_levels"] == ["baseline", "week 4", "week 8"]
    assert result["extra"]["visit_order_source"] == "ordered-categorical-levels"


def test_order_dependent_covariance_refuses_ambiguous_visit_labels() -> None:
    data = _mmrm_data()
    numeric_visit = np.asarray(data["visit"])
    data["visit"] = np.asarray(["late", "early", "middle"], dtype=object)[numeric_visit]

    with pytest.raises(BackendInputError, match="requires an explicit visit order"):
        fit_mmrm(data, covariance="ar1", compute_hessian=False)
