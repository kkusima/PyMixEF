import numpy as np
import pytest

from pymixef.pharmacometrics import (
    ODESimulationError,
    UnsupportedEventSemantics,
    additive,
    combined,
    finite_difference_sensitivities,
    left_censored_loglikelihood,
    one_compartment_infusion,
    one_compartment_iv_bolus,
    one_compartment_oral,
    proportional,
    simulate_ode,
    simulate_subjects,
    two_compartment_iv_bolus,
)


def test_iv_bolus_ode_matches_one_compartment_closed_form() -> None:
    events = [
        {"ID": 1, "TIME": 0, "EVID": 1, "AMT": 100, "CMT": 1},
        {"ID": 1, "TIME": 0, "EVID": 0},
        {"ID": 1, "TIME": 1, "EVID": 0},
        {"ID": 1, "TIME": 4, "EVID": 0},
    ]

    def rhs(t, y, parameters):
        del t
        return [-parameters.parameters["CL"] / parameters.parameters["V"] * y[0]]

    times = np.array([0.0, 1.0, 4.0])
    result = simulate_ode(
        rhs,
        [0.0],
        events,
        t_eval=times,
        parameters={"CL": 5.0, "V": 20.0},
        state_names=["central"],
    )
    expected = one_compartment_iv_bolus(times, dose=100, clearance=5, volume=20)
    np.testing.assert_allclose(result.state("central") / 20, expected, rtol=2e-8)
    assert result.observations[0].state[0] == pytest.approx(100)
    assert result.metadata.event_actions == 4
    assert result.metadata.success
    with pytest.raises(ValueError):
        result.states[0, 0] = 1


def test_same_time_reset_covariate_dose_observation_semantics() -> None:
    events = [
        {"ID": 1, "TIME": 0, "EVID": 1, "AMT": 4},
        {"ID": 1, "TIME": 2, "EVID": 0},
        {"ID": 1, "TIME": 2, "EVID": 1, "AMT": 5},
        {"ID": 1, "TIME": 2, "EVID": 3},
        {
            "ID": 1,
            "TIME": 2,
            "EVID": "covariate",
            "COVARIATES": {"slope": 3.0},
        },
        {"ID": 1, "TIME": 3, "EVID": 0},
    ]

    def rhs(t, y, context):
        del t, y
        return [context.covariates.get("slope", 0.0)]

    result = simulate_ode(rhs, [0], events, initial_covariates={"slope": 1.0})
    # Reset -> covariate update -> dose -> observation.
    assert result.observations[0].state[0] == pytest.approx(5)
    assert result.observations[1].state[0] == pytest.approx(8)


def test_finite_infusion_and_overlapping_rates_match_closed_form() -> None:
    events = [
        {"ID": 1, "TIME": 0, "EVID": 1, "AMT": 100, "RATE": 50, "CMT": 1},
        {"ID": 1, "TIME": 1, "EVID": 1, "AMT": 50, "RATE": 50, "CMT": 1},
    ]
    times = np.array([0.5, 1.0, 1.5, 2.0, 3.0])

    def rhs(t, y):
        del t
        return [-0.2 * y[0]]

    result = simulate_ode(rhs, [0], events, t_eval=times)
    expected = one_compartment_infusion(times, dose=100, rate=50, clearance=0.2, volume=1)
    expected += one_compartment_infusion(times, dose=50, rate=50, clearance=0.2, volume=1, start=1)
    np.testing.assert_allclose(result.states[:, 0], expected, rtol=2e-8, atol=1e-9)
    assert result.metadata.generated_infusion_stops == 2


def test_addl_is_applied_and_metadata_retained() -> None:
    events = [
        {"ID": 1, "TIME": 0, "EVID": 1, "AMT": 2, "ADDL": 2, "II": 1},
    ]
    result = simulate_ode(lambda t, y: np.zeros_like(y), [0], events, t_eval=[0, 1, 2])
    np.testing.assert_allclose(result.states[:, 0], [2, 4, 6])
    assert result.metadata.generated_additional_doses == 2


def test_reset_and_dose_are_split_around_covariate_priority() -> None:
    events = [
        {"ID": 1, "TIME": 0, "EVID": 1, "AMT": 100},
        {"ID": 1, "TIME": 1, "EVID": 4, "AMT": 7},
        {"ID": 1, "TIME": 1, "EVID": 0},
    ]
    result = simulate_ode(lambda t, y: [0], [0], events)
    assert result.observations[0].state[0] == pytest.approx(7)


def test_covariates_and_compartment_mapping() -> None:
    events = [
        {"ID": "S", "TIME": 0, "EVID": 1, "AMT": 3, "CMT": "CENT"},
        {
            "ID": "S",
            "TIME": 0,
            "EVID": 5,
            "COVARIATES": {"input": 2.0},
        },
    ]

    def rhs(t, y, parameters, covariates):
        del t, y
        return [parameters["base"] + covariates["input"]]

    result = simulate_ode(
        rhs,
        [0],
        events,
        t_eval=[0, 1],
        parameters={"base": 1},
        state_names=["central"],
        compartment_map={"CENT": "central"},
    )
    np.testing.assert_allclose(result.states[:, 0], [3, 6], rtol=1e-8)


def test_forward_sensitivity_matches_exponential_decay_derivative() -> None:
    def rhs(t, y, context):
        del t
        return [-context.parameters["k"] * y[0]]

    times = np.array([0.0, 0.5, 2.0])
    result = simulate_ode(
        rhs,
        [1],
        t_eval=times,
        parameters={"k": 0.4},
        sensitivity_parameters=["k"],
    )
    expected = -times * np.exp(-0.4 * times)
    np.testing.assert_allclose(result.sensitivity(0, "k"), expected, rtol=2e-5, atol=1e-7)
    assert result.metadata.sensitivity_method == "forward-finite-difference"
    check = finite_difference_sensitivities(
        rhs,
        [1],
        None,
        parameters={"k": 0.4},
        parameter_names=["k"],
        t_eval=times,
        compare_central=True,
    )
    assert check.maximum_scaled_difference is not None
    assert check.maximum_scaled_difference < 1e-4


def test_steady_state_fails_with_stable_unsupported_error() -> None:
    with pytest.raises(UnsupportedEventSemantics) as caught:
        simulate_ode(
            lambda t, y: [0],
            [0],
            [{"ID": 1, "TIME": 0, "EVID": 1, "AMT": 1, "SS": 1}],
        )
    assert caught.value.code == "ODE-EVENT-UNSUPPORTED-001"


def test_solver_failure_has_structured_context() -> None:
    def bad_rhs(t, y):
        del y
        if t > 0:
            return [np.nan]
        return [0]

    with pytest.raises(ODESimulationError) as caught:
        simulate_ode(bad_rhs, [0], t_eval=[1], subject_id="bad")
    assert caught.value.subject_id == "bad"
    assert caught.value.code == "ODE-SIMULATION-FAILED-001"


def test_simulate_subjects_is_deterministic() -> None:
    events = [
        {"ID": "B", "TIME": 0, "EVID": 1, "AMT": 2},
        {"ID": "A", "TIME": 0, "EVID": 1, "AMT": 1},
    ]
    results = simulate_subjects(lambda t, y: [0], [0], events, t_eval=[0], state_names=["central"])
    assert tuple(results) == ("A", "B")
    assert results["A"].states[0, 0] == 1
    assert results["B"].states[0, 0] == 2


def test_closed_form_pk_limits_and_two_compartment_initial_value() -> None:
    oral = one_compartment_oral(
        [0, 1],
        dose=100,
        clearance=10,
        volume=20,
        absorption_rate=0.5,  # equal to elimination rate
    )
    assert oral[0] == pytest.approx(0)
    assert oral[1] == pytest.approx(5 * 0.5 * np.exp(-0.5))
    central = two_compartment_iv_bolus(
        [0, 1],
        dose=100,
        clearance=5,
        central_volume=20,
        intercompartmental_clearance=4,
        peripheral_volume=30,
    )
    assert central[0] == pytest.approx(5)
    assert central[1] < central[0]


def test_observation_error_models_and_stable_bql_likelihood() -> None:
    prediction = np.array([0.0, 2.0])
    error = additive(1.0) + proportional(0.5)
    assert error.variance(prediction).tolist() == pytest.approx([1.0, 2.0])
    explicit = combined(1.0, 0.5)
    np.testing.assert_allclose(error.variance(prediction), explicit.variance(prediction))
    logcdf = left_censored_loglikelihood([-1000], [0], additive(1), parameters=None)
    assert np.isfinite(logcdf[0])
