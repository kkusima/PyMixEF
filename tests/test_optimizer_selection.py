from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymixef.backends.base import optimizer_covariance, select_optimizer_result


def _result(*, success: bool, objective: float) -> SimpleNamespace:
    return SimpleNamespace(success=success, fun=objective)


def test_successful_rescue_is_retained_over_failed_refinement() -> None:
    rescue = _result(success=True, objective=12.0)
    failed_refinement = _result(success=False, objective=12.0 - 1e-10)

    selected = select_optimizer_result(
        failed_refinement,
        rescue,
        objective_tolerance=1e-8,
    )

    assert selected is rescue


def test_successful_refinement_is_preferred_within_objective_tolerance() -> None:
    rescue = _result(success=True, objective=12.0)
    refined = _result(success=True, objective=12.0 + 1e-10)

    selected = select_optimizer_result(
        refined,
        rescue,
        objective_tolerance=1e-8,
    )

    assert selected is refined


def test_materially_better_failed_refinement_is_not_called_converged() -> None:
    rescue = _result(success=True, objective=12.0)
    failed_refinement = _result(success=False, objective=10.0)

    selected = select_optimizer_result(
        failed_refinement,
        rescue,
        objective_tolerance=1e-8,
    )

    assert selected is failed_refinement


def test_finite_result_is_preferred_when_neither_optimizer_converges() -> None:
    nonfinite_rescue = _result(success=False, objective=np.inf)
    finite_refinement = _result(success=False, objective=12.0)

    selected = select_optimizer_result(
        finite_refinement,
        nonfinite_rescue,
        objective_tolerance=1e-8,
    )

    assert selected is finite_refinement


def test_covariance_falls_back_to_observed_hessian_at_selected_point() -> None:
    selected_powell = _result(success=True, objective=4.0)

    covariance, positive, source = optimizer_covariance(
        lambda point: float(2.0 * (point[0] - 3.0) ** 2),
        np.array([3.0]),
        selected_powell,
        compute_hessian=False,
        finite_difference_limit=5,
    )

    np.testing.assert_allclose(covariance, [[0.25]], rtol=1e-4)
    assert positive is True
    assert source == "observed-finite-difference-fallback"
