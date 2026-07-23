from types import SimpleNamespace

import numpy as np
import pytest

import pymixef.pharmacometrics.estimation as estimation_module
from pymixef.pharmacometrics import (
    ConditionalModeError,
    ConditionalObjective,
    Dose,
    DSLValidationError,
    Eta,
    Param,
    SAEMControl,
    SAEMProblem,
    State,
    UnsupportedEstimatorError,
    additive,
    apply_random_effects,
    compiled_model,
    covariate,
    d,
    eta_shrinkage,
    exp,
    experimental_saem,
    find_conditional_mode,
    fit_focei,
    laplace_population_objective,
    model,
    observe,
    omega_from_standard_deviations,
    proportional,
)


def test_typed_model_decorator_compiles_and_explains() -> None:
    @model
    def one_compartment():
        tvcl = Param.positive("tvcl", init=5, unit="L/h")
        tvv = Param.positive("tvv", init=30, unit="L")
        sigma_add = Param.positive("sigma_add", init=0.1)
        (eta_cl,) = Eta.independent("eta_cl")
        weight = covariate("weight", unit="kg", reference=70)
        clearance = tvcl * (weight / 70) ** 0.75 * exp(eta_cl)
        central = State("central", unit="mg")
        Dose.into(central, amount="AMT", rate="RATE")
        d(central, -(clearance / tvv) * central)
        observe("DV", mean=central / tvv, error=additive(sigma_add), censored_below="LLOQ")

    compiled = one_compartment()
    assert compiled.name == "one_compartment"
    assert [parameter.name for parameter in compiled.parameters] == [
        "tvcl",
        "tvv",
        "sigma_add",
    ]
    assert compiled.validate().valid
    assert compiled.validate().estimator_compatibility["focei_fit"] is False
    specification = compiled.explain()
    assert "PyMixEF pharmacometric model" in specification
    assert "d(central)/dt" in specification
    assert "L/h" in specification
    serialized = compiled.to_dict()
    assert serialized["schema_version"] == "1.0"
    assert serialized["authoring_mode"] == "executed-python-declarations"
    assert one_compartment.declaration_signature == "()"


def test_expression_tree_is_data_only_and_explicitly_evaluable() -> None:
    parameter = Param.positive("clearance", init=1)
    expression = exp(parameter) / 2
    assert expression.to_dict()["operation"] == "divide"
    assert expression.evaluate({"clearance": np.log(4)}) == pytest.approx(2)
    assert "exp(clearance)" in expression.format()


def test_direct_compilation_does_not_execute_a_function() -> None:
    parameter = Param.positive("k", init=0.2)
    state = State("amount")
    equation = state.derivative(-parameter * state)
    endpoint = observe("DV", mean=state, error=additive(1.0))
    direct = compiled_model(
        "direct",
        parameters=[parameter],
        states=[state],
        equations=[equation],
        observations=[endpoint],
    )
    assert direct.authoring_mode == "direct-data-declarations"
    assert direct.validate().valid


def test_model_validation_rejects_missing_derivative_and_duplicates() -> None:
    @model
    def invalid():
        Param.real("x")
        Param.real("x")
        State("central")
        observe("DV", mean=0, error=additive(1))

    with pytest.raises(DSLValidationError, match="duplicate parameter"):
        invalid()


def test_parameter_constraints_are_checked() -> None:
    with pytest.raises(DSLValidationError):
        Param.positive("bad", init=0)
    with pytest.raises(DSLValidationError):
        Param.bounded("probability", init=2, lower=0, upper=1)
    parameter = Param.bounded("probability", init=0.5, lower=0, upper=1)
    assert parameter.constraint == "bounded"


def test_error_composition_serializes_in_model() -> None:
    @model
    def combined_error_model():
        state = State("amount")
        d(state, -state)
        observe("DV", mean=state, error=additive("sa") + proportional("sp"))

    error = combined_error_model().to_dict()["observations"][0]["error"]
    assert error == {
        "type": "combined",
        "additive_sigma": "sa",
        "proportional_sigma": "sp",
        "power": 1.0,
    }


def test_population_helpers() -> None:
    individual = apply_random_effects(
        {"CL": 5.0, "F": 0.4},
        {"CL": np.log(2), "F": 0.0},
        relationships={"F": "logit"},
    )
    assert individual["CL"] == pytest.approx(10)
    assert individual["F"] == pytest.approx(0.4)
    omega = omega_from_standard_deviations([0.2, 0.3], [[1.0, 0.25], [0.25, 1.0]])
    assert np.all(np.linalg.eigvalsh(omega) > 0)
    shrinkage = eta_shrinkage([[-0.1, -0.15], [0.1, 0.15], [0.0, 0.0]], np.diag([0.04, 0.09]))
    assert np.all(shrinkage > 0)


def test_conditional_mode_objective_retains_residual_interaction() -> None:
    observations = np.array([2.0, 2.2, 1.8])

    def predict(eta):
        return np.full(observations.shape, 2.0 * np.exp(eta[0]))

    objective = ConditionalObjective(
        observations,
        predict,
        np.array([[0.5**2]]),
        proportional(0.1),
    )
    variance_at_zero = objective.components([0]).variances
    variance_at_one = objective.components([1]).variances
    assert np.all(variance_at_one > variance_at_zero)
    mode = find_conditional_mode(objective, require_success=True)
    assert abs(mode.eta[0]) < 0.05
    assert mode.hessian_positive_definite
    assert mode.gradient_norm < 1e-4
    population = laplace_population_objective([objective])
    assert np.isfinite(population.objective)
    assert len(population.modes) == 1


def _quadratic_conditional_objective() -> ConditionalObjective:
    observations = np.array([0.0])
    return ConditionalObjective(
        observations,
        lambda eta: np.array([eta[0]]),
        np.array([[1.0]]),
        additive(1.0),
    )


def test_conditional_mode_accepts_independently_verified_precision_loss(monkeypatch) -> None:
    objective = _quadratic_conditional_objective()

    def report_precision_loss(*args, **kwargs):
        return SimpleNamespace(
            x=np.array([0.0]),
            success=False,
            message="Desired error not necessarily achieved due to precision loss.",
            nit=4,
            nfev=30,
        )

    monkeypatch.setattr(estimation_module, "minimize", report_precision_loss)
    mode = find_conditional_mode(objective, require_success=True)

    assert mode.success
    assert mode.hessian_positive_definite
    assert mode.gradient_norm <= 1e-4
    assert "NLME-CONDITIONAL-OPTIMIZER-001" in mode.warning_codes


def test_conditional_mode_rejects_unverified_optimizer_failure(monkeypatch) -> None:
    objective = _quadratic_conditional_objective()

    def report_nonstationary_failure(*args, **kwargs):
        return SimpleNamespace(
            x=np.array([1.0]),
            success=False,
            message="Desired error not necessarily achieved due to precision loss.",
            nit=0,
            nfev=1,
        )

    monkeypatch.setattr(estimation_module, "minimize", report_nonstationary_failure)
    mode = find_conditional_mode(objective, tolerance=1e-2)
    assert not mode.success
    assert "NLME-CONDITIONAL-OPTIMIZER-001" in mode.warning_codes
    assert "NLME-CONDITIONAL-GRADIENT-001" in mode.warning_codes

    with pytest.raises(ConditionalModeError, match="conditional mode failed"):
        find_conditional_mode(objective, tolerance=1e-2, require_success=True)


def test_conditional_mode_require_success_checks_raw_success_result(monkeypatch) -> None:
    objective = _quadratic_conditional_objective()

    def report_nonstationary_success(*args, **kwargs):
        return SimpleNamespace(
            x=np.array([1.0]),
            success=True,
            message="Optimization terminated successfully.",
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr(estimation_module, "minimize", report_nonstationary_success)
    with pytest.raises(ConditionalModeError, match="NLME-CONDITIONAL-GRADIENT-001"):
        find_conditional_mode(objective, require_success=True)


def test_conditional_objective_supports_bql_and_missing_values() -> None:
    objective = ConditionalObjective(
        np.array([np.nan, 1.0, np.nan]),
        lambda eta: np.full(3, eta[0]),
        np.array([[1.0]]),
        additive(0.5),
        censored=np.array([True, False, False]),
        lower_limits=np.array([0.2, np.nan, np.nan]),
    )
    assert np.isfinite(objective([0.5]))


def test_production_focei_has_stable_unsupported_error() -> None:
    with pytest.raises(UnsupportedEstimatorError) as caught:
        fit_focei()
    assert caught.value.code == "ENGINE-UNSUPPORTED-001"
    assert "ConditionalObjective" in caught.value.compatible


def test_experimental_saem_is_seeded_transparent_and_callback_based() -> None:
    # A deliberately simple latent-normal toy problem.  Its sufficient
    # statistic is E[z], and the M-step maps that statistic directly to theta.
    problem = SAEMProblem(
        initial_parameters=np.array([0.0]),
        initial_latent=np.array([0.0]),
        log_joint=lambda theta, latent: (
            -0.5 * float(np.square(latent - theta).sum() + np.square(latent - 1.0).sum())
        ),
        sufficient_statistics=lambda theta, latent: latent.copy(),
        m_step=lambda statistics, current: np.array([statistics.mean()]),
        parameter_names=("location",),
    )
    controls = SAEMControl(
        iterations=80,
        burn_in=20,
        mcmc_steps=3,
        proposal_scale=0.4,
        seed=42,
        keep_latent_trace=True,
    )
    first = experimental_saem(problem, controls)
    second = experimental_saem(problem, controls)
    np.testing.assert_array_equal(first.parameter_trace, second.parameter_trace)
    assert first.experimental
    assert first.reproducibility_class == "stochastic-with-monte-carlo-error"
    assert first.proposals == controls.iterations * controls.mcmc_steps
    assert 0 <= first.acceptance_rate <= 1
    assert first.step_sizes[0] == 1
    assert first.step_sizes[-1] < 1
    assert first.latent_trace is not None
