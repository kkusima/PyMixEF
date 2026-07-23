from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from pymixef.compare import approximation_sensitivity
from pymixef.data import adapt_data, pattern_mixture_adjust
from pymixef.diagnostics import group_influence
from pymixef.errors import DataError

from .test_results import _result


def test_pattern_mixture_adjustment_preserves_observed_values_and_row_identity() -> None:
    source = {
        "response": np.array([11.0, 12.0, 13.0, 14.0]),
        "imputed": np.array([False, True, False, True]),
        "treatment": np.array(["control", "control", "active", "active"]),
        "visit": np.array([1, 2, 1, 2]),
    }
    result = pattern_mixture_adjust(
        source,
        response="response",
        imputed="imputed",
        delta={("control", 2): -1.5, ("active", 2): -3.0},
        by=("treatment", "visit"),
    )

    np.testing.assert_array_equal(result.data["response"], [11.0, 10.5, 13.0, 11.0])
    np.testing.assert_array_equal(result.data["treatment"], source["treatment"])
    assert result.data.row_ids == adapt_data(source).row_ids
    assert [record.input_position for record in result.records] == [1, 3]
    assert [record.delta for record in result.records] == [-1.5, -3.0]
    assert result.adjusted_rows == 2
    assert result.to_dict()["adjusted_fingerprint"] == result.data.fingerprint


def test_pattern_mixture_adjustment_validates_mask_completion_and_strata() -> None:
    with pytest.raises(DataError, match="Boolean"):
        pattern_mixture_adjust(
            {"y": [1.0, 2.0], "imputed": [0, 1]},
            response="y",
            imputed="imputed",
            delta=-1.0,
        )
    with pytest.raises(DataError, match="completed and finite"):
        pattern_mixture_adjust(
            {"y": [1.0, np.nan], "imputed": [False, True]},
            response="y",
            imputed="imputed",
            delta=-1.0,
        )
    with pytest.raises(DataError, match="No delta"):
        pattern_mixture_adjust(
            {
                "y": [1.0, 2.0],
                "imputed": [False, True],
                "arm": ["control", "active"],
            },
            response="y",
            imputed="imputed",
            delta={"control": -1.0},
            by="arm",
        )


def test_pattern_mixture_preserves_large_unselected_integers_and_rejects_overflow() -> None:
    large = 2**60 + 1
    result = pattern_mixture_adjust(
        {
            "y": np.asarray([large, 5], dtype=np.int64),
            "imputed": np.asarray([False, True]),
        },
        response="y",
        imputed="imputed",
        delta=-1.5,
    )
    assert result.data["y"][0] == large
    assert isinstance(result.data["y"][0], (int, np.integer))
    assert result.data["y"][1] == 3.5

    with pytest.raises(DataError, match="overflowed"):
        pattern_mixture_adjust(
            {
                "y": [1.0, np.finfo(float).max],
                "imputed": [False, True],
            },
            response="y",
            imputed="imputed",
            delta=np.finfo(float).max,
        )


def test_pattern_mixture_rejects_ambiguous_or_non_numeric_contracts() -> None:
    with pytest.raises(DataError, match="separate"):
        pattern_mixture_adjust(
            {"y": [True, False]},
            response="y",
            imputed="y",
            delta=-1.0,
        )
    with pytest.raises(DataError, match="non-Boolean"):
        pattern_mixture_adjust(
            {"y": ["1.0", "2.0"], "imputed": [False, True]},
            response="y",
            imputed="imputed",
            delta=-1.0,
        )
    with pytest.raises(DataError, match="must be numeric"):
        pattern_mixture_adjust(
            {"y": [1.0, 2.0], "imputed": [False, True]},
            response="y",
            imputed="imputed",
            delta=object(),
        )


def _mean_fit(data: Mapping[str, np.ndarray], *, rank: int = 2):
    fit = _result()
    mean = float(np.mean(data["y"]))
    fit.parameters = {
        "beta[Intercept]": mean,
        "residual_sd": float(np.std(data["y"], ddof=0)),
    }
    fit.objective = float(np.sum(np.square(data["y"] - mean)))
    fit.convergence = replace(
        fit.convergence,
        engine_metrics={"fixed_effect_rank": rank},
    )
    return fit


def test_group_influence_deletes_complete_groups_and_compares_approximation() -> None:
    data = {
        "y": np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0]),
        "site": np.repeat(["north", "south", "west"], 2),
    }
    observed_samples: list[tuple[str, ...]] = []

    def fit_function(sample: Mapping[str, np.ndarray]):
        labels, counts = np.unique(sample["site"], return_counts=True)
        assert np.all(counts == 2)
        observed_samples.append(tuple(labels.tolist()))
        return _mean_fit(sample)

    def exact_approximation(level: object, baseline: object) -> Mapping[str, float]:
        keep = data["site"] != level
        return _mean_fit({name: values[keep] for name, values in data.items()}).parameters

    result = group_influence(
        fit_function,
        data,
        group="site",
        approximation=exact_approximation,
    )

    assert observed_samples[0] == ("north", "south", "west")
    assert all(len(labels) == 2 for labels in observed_samples[1:])
    assert result.requested_groups == 3
    assert result.successful_groups == 3
    assert result.failed_groups == 0
    assert len(result.table) == 6
    np.testing.assert_allclose(result.table.columns["approximation_error"], 0.0)
    assert np.all(np.isfinite(result.table.columns["cook_distance"]))
    assert set(result.table.columns["remaining_rows"]) == {4}
    assert not np.any(result.table.columns["rank_changed"])


def test_group_influence_accounts_for_refit_failures() -> None:
    data = {
        "y": np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0]),
        "site": np.repeat([0, 1, 2], 2),
    }

    def fit_function(sample: Mapping[str, np.ndarray]):
        if len(sample["y"]) < 6 and 1 not in sample["site"]:
            raise RuntimeError("deliberate site-deletion failure")
        return _mean_fit(sample)

    result = group_influence(fit_function, data, group="site")
    assert result.requested_groups == 3
    assert result.successful_groups == 2
    assert result.failed_groups == 1
    assert result.failures[0]["phase"] == "full-refit"
    assert "deliberate" in str(result.failures[0]["message"])


def test_group_influence_accounts_for_nonfinite_refit_objectives() -> None:
    data = {
        "y": np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0]),
        "site": np.repeat([0, 1, 2], 2),
    }

    def fit_function(sample: Mapping[str, np.ndarray]):
        fit = _mean_fit(sample)
        if len(sample["y"]) < 6 and 1 not in sample["site"]:
            fit.objective = np.nan
        return fit

    result = group_influence(fit_function, data, group="site")
    assert result.failed_groups == 1
    assert "non-finite objective" in str(result.failures[0]["message"])


def test_group_influence_surfaces_rank_changes_and_rejects_indefinite_covariance() -> None:
    data = {
        "y": np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0]),
        "site": np.repeat([0, 1, 2], 2),
    }

    def fit_function(sample: Mapping[str, np.ndarray]):
        rank = 1 if len(sample["y"]) < 6 and 2 not in sample["site"] else 2
        return _mean_fit(sample, rank=rank)

    result = group_influence(fit_function, data, group="site")
    changed = result.table.columns["rank_changed"]
    assert np.any(changed)
    assert set(result.table.columns["baseline_model_rank"]) == {2}
    assert set(result.table.columns["full_refit_model_rank"][changed]) == {1}

    baseline = _mean_fit(data)
    baseline.parameter_covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        group_influence(fit_function, data, group="site", baseline=baseline)

    baseline = _mean_fit(data)
    baseline.parameters = {**baseline.parameters, "beta[Intercept]": np.nan}
    with pytest.raises(RuntimeError, match="non-finite parameter"):
        group_influence(fit_function, data, group="site", baseline=baseline)


def test_approximation_sensitivity_refits_and_retains_failures() -> None:
    def fit_function(settings: Mapping[str, object]):
        if settings.get("fail"):
            raise RuntimeError("deliberate sensitivity failure")
        fit = _result()
        offset = float(settings["offset"])
        fit.parameters = {name: value + offset for name, value in fit.parameters.items()}
        fit.parameter_covariance = np.eye(2) * (0.1 + offset)
        fit.objective += 2.0 * offset
        return fit

    result = approximation_sensitivity(
        fit_function,
        {
            "reference": {"offset": 0.0, "quadrature_order": 1},
            "tighter": {"offset": 0.1, "quadrature_order": 5},
            "failed": {"offset": 0.0, "fail": True},
        },
        materiality={
            "parameter_relative": 0.2,
            "standard_error_relative": 0.2,
            "objective_absolute": 0.5,
        },
        baseline="reference",
    )

    assert result.baseline == "reference"
    assert result.successful_scenarios == 2
    assert result.failed_scenarios == 1
    assert len(result.table) == 4
    tighter = result.table.columns["scenario"] == "tighter"
    np.testing.assert_allclose(result.table.columns["difference"][tighter], 0.1)
    np.testing.assert_allclose(result.table.columns["objective_difference"][tighter], 0.2)
    assert np.all(result.table.columns["material_standard_error"][tighter])
    assert not np.any(result.table.columns["material_parameter"][tighter])
    assert not np.any(result.table.columns["material_objective"][tighter])
    assert np.all(result.table.columns["material"][tighter])
    assert result.settings["tighter"]["quadrature_order"] == 5
    assert result.failures[0]["scenario"] == "failed"


def test_approximation_sensitivity_requires_a_successful_baseline() -> None:
    def fail(_: Mapping[str, object]):
        raise RuntimeError("no fit")

    with pytest.raises(RuntimeError, match="Baseline scenario"):
        approximation_sensitivity(
            fail,
            {"reference": {}, "alternative": {}},
            materiality={
                "parameter_relative": 0.1,
                "standard_error_relative": 0.1,
                "objective_absolute": 1.0,
            },
            baseline="reference",
        )


def test_approximation_sensitivity_deep_copies_settings_and_requires_covariance() -> None:
    scenarios = {
        "reference": {"offset": 0.0, "solver": {"orders": [1]}},
        "alternative": {"offset": 0.1, "solver": {"orders": [3]}},
    }

    def mutating_fit(settings: Mapping[str, object]):
        solver = settings["solver"]
        assert isinstance(solver, dict)
        orders = solver["orders"]
        assert isinstance(orders, list)
        orders.append(99)
        fit = _result()
        offset = float(settings["offset"])
        fit.parameters = {name: value + offset for name, value in fit.parameters.items()}
        if offset:
            fit.parameter_covariance = None
        return fit

    result = approximation_sensitivity(
        mutating_fit,
        scenarios,
        materiality={
            "parameter_relative": 0.1,
            "standard_error_relative": 0.1,
            "objective_absolute": 1.0,
        },
        baseline="reference",
    )
    assert scenarios["reference"]["solver"]["orders"] == [1]
    assert scenarios["alternative"]["solver"]["orders"] == [3]
    assert result.settings["reference"]["solver"]["orders"] == [1]
    assert result.failed_scenarios == 1
    assert result.failures[0]["scenario"] == "alternative"
    assert "covariance" in str(result.failures[0]["message"])


def test_approximation_sensitivity_rejects_indefinite_covariance() -> None:
    def fit_function(settings: Mapping[str, object]):
        fit = _result()
        if settings["indefinite"]:
            fit.parameter_covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
        return fit

    result = approximation_sensitivity(
        fit_function,
        {
            "reference": {"indefinite": False},
            "invalid": {"indefinite": True},
        },
        materiality={
            "parameter_relative": 0.1,
            "standard_error_relative": 0.1,
            "objective_absolute": 1.0,
        },
        baseline="reference",
    )
    assert result.failed_scenarios == 1
    assert "positive semidefinite" in str(result.failures[0]["message"])
