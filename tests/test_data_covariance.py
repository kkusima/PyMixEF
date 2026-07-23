from __future__ import annotations

import numpy as np
import pytest

from pymixef.covariance import (
    AR1,
    AnteDependence,
    CompoundSymmetry,
    Diagonal,
    HeterogeneousAR1,
    HeterogeneousToeplitz,
    SpatialPower,
    Toeplitz,
    Unstructured,
    singularity_report,
    validate_covariance,
)
from pymixef.data import (
    MissingnessKind,
    adapt_data,
    audit_data,
    find_duplicate_keys,
    stable_sort,
    validate_monotonic_time,
)
from pymixef.errors import DataError
from pymixef.plugins import Registry
from pymixef.transforms import (
    BoundedTransform,
    CholeskyCovarianceTransform,
    LogTransform,
    OrderedTransform,
    SimplexTransform,
    SoftplusTransform,
)


def test_mapping_adapter_is_immutable_and_row_ids_are_stable() -> None:
    source = {"y": [1.0, 2.0], "x": [3.0, 4.0]}
    table = adapt_data(source)
    same = adapt_data(source)
    assert table.row_ids == same.row_ids
    assert table.fingerprint == same.fingerprint
    with pytest.raises(ValueError):
        table["x"][0] = 99
    source["x"][0] = 99
    assert table["x"][0] == 3


def test_missingness_contract_distinguishes_record_types() -> None:
    source = {
        "y": [1.0, np.nan, np.nan, np.nan, 5.0],
        "x": [1.0, 2.0, 3.0, np.nan, 5.0],
        "censored": [False, True, False, False, False],
        "structural": [False, False, True, False, False],
        "invalid": [False, False, False, False, True],
    }
    result = audit_data(
        source,
        response="y",
        covariates=("x",),
        censored="censored",
        structurally_absent="structural",
        invalid="invalid",
    )
    assert result.audit.input_rows == 5
    assert result.audit.analysis_rows == 2
    kinds = [record.missingness for record in result.audit.records]
    assert kinds == [
        None,
        MissingnessKind.CENSORED_RESPONSE,
        MissingnessKind.STRUCTURALLY_ABSENT_ENDPOINT,
        MissingnessKind.MISSING_COVARIATE,
        MissingnessKind.INVALID_RECORD,
    ]


def test_longitudinal_helpers() -> None:
    data = {"id": ["b", "a", "a"], "time": [1, 2, 1]}
    assert find_duplicate_keys({"id": ["a", "a"], "time": [1, 1]}, ("id", "time")) == (("a", 1),)
    with pytest.raises(DataError, match="decreases"):
        validate_monotonic_time(data, group="id", time="time")
    sorted_data, order = stable_sort(data, ("id", "time"))
    assert order == (2, 1, 0)
    assert sorted_data["id"].tolist() == ["a", "a", "b"]


@pytest.mark.parametrize(
    ("structure", "size"),
    [
        (Diagonal, 4),
        (Unstructured, 4),
        (CompoundSymmetry, 4),
        (AR1, 4),
        (HeterogeneousAR1, 4),
        (Toeplitz, 4),
        (HeterogeneousToeplitz, 4),
        (AnteDependence, 4),
        (SpatialPower, 4),
    ],
)
def test_covariance_structures_are_positive_definite(structure: type, size: int) -> None:
    kernel = structure(size)
    parameters = np.random.default_rng(20260723).normal(scale=0.7, size=kernel.parameter_count())
    matrix = kernel.covariance(parameters)
    report = validate_covariance(matrix)
    assert report.positive_definite
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix)[0] > 0
    derivatives = kernel.derivatives(parameters)
    assert derivatives.shape == (parameters.size, size, size)
    draws = kernel.simulate(parameters, rng=4, draws=3)
    assert draws.shape == (3, size)


def test_ar1_unequal_spacing_and_spatial_coordinates() -> None:
    ar = AR1(3)
    matrix = ar.covariance([0.0, 0.5], index=[0.0, 0.5, 2.0])
    assert np.linalg.eigvalsh(matrix)[0] > 0
    spatial = SpatialPower()
    matrix = spatial.covariance([0.0, 0.0], coordinates=[[0, 0], [1, 0], [1, 2]])
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[0, 2] == pytest.approx(0.5 ** np.sqrt(5))


def test_singularity_report_does_not_crash_on_boundary() -> None:
    report = singularity_report([[1.0, 1.0], [1.0, 1.0]])
    assert report["singular"]
    assert report["effective_rank"] == 1
    assert report["near_perfect_correlations"] == [(0, 1, 1.0)]


@pytest.mark.parametrize(
    ("transform", "unconstrained"),
    [
        (LogTransform(), np.array([-5.0, 0.0, 2.0])),
        (SoftplusTransform(), np.array([-5.0, 0.0, 2.0])),
        (BoundedTransform(-2.0, 3.0), np.array([-5.0, 0.0, 2.0])),
        (SimplexTransform(), np.array([-2.0, 1.0, 0.5])),
        (OrderedTransform(), np.array([-2.0, 1.0, 0.5])),
        (CholeskyCovarianceTransform(2), np.array([-1.0, 0.3, 0.2])),
    ],
)
def test_canonical_transforms_round_trip(transform: object, unconstrained: np.ndarray) -> None:
    natural = transform.forward(unconstrained)
    np.testing.assert_allclose(transform.inverse(natural), unconstrained, atol=1e-10)
    assert np.isfinite(transform.log_abs_det_jacobian(unconstrained))


def test_plugin_registry_is_deterministic_and_rejects_duplicates() -> None:
    registry: Registry[object] = Registry("test")
    alpha = object()
    beta = object()
    registry.register("Beta_Value", beta)
    registry.register("alpha", alpha)
    assert registry.names() == ("alpha", "beta-value")
    assert registry.get("BETA_VALUE") is beta
    with pytest.raises(Exception, match="already registered"):
        registry.register("alpha", object())
