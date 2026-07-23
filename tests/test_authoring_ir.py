from __future__ import annotations

import pytest

from pymixef import Fixed, Model, ModelIR, Random, Response
from pymixef.formula import parse_formula
from pymixef.ir import MODEL_IR_SCHEMA_VERSION, PriorIR
from pymixef.pharmacometrics import (
    Dose,
    DSLValidationError,
    Eta,
    Expr,
    Param,
    State,
    additive,
    compiled_model,
    covariate,
    d,
    exp,
    model,
    observe,
)


def _assert_versioned_round_trip(model_ir: ModelIR) -> None:
    assert model_ir.schema_version == MODEL_IR_SCHEMA_VERSION
    assert ModelIR.from_json(model_ir.to_json(indent=2)) == model_ir
    assert ModelIR.from_json(model_ir.canonical_json()).semantic_hash == model_ir.semantic_hash


def test_all_authoring_surfaces_produce_common_versioned_ir() -> None:
    formula_ir = parse_formula("y ~ time + (time | subject)").to_ir()

    builder_ir = Model(
        response=Response("y"),
        fixed=Fixed("time"),
        random=(Random("time", group="subject"),),
        priors={
            "time": {
                "distribution": "normal",
                "parameters": {"mean": 0.0, "sd": 2.0},
            },
            "Intercept": PriorIR(
                target="Intercept",
                distribution="normal",
                parameters={"mean": 0.0, "sd": 10.0},
            ),
        },
    ).to_ir()

    @model
    def one_compartment():
        clearance = Param.positive("clearance", init=5.0, unit="L/h")
        volume = Param.positive("volume", init=30.0, unit="L")
        sigma = Param.positive("sigma", init=0.1)
        (eta_clearance,) = Eta.independent("eta_clearance", block="omega_cl")
        weight = covariate("weight", unit="kg", reference=70)
        central = State("central", unit="mg")
        Dose.into(central, amount="AMT", rate="RATE")
        d(
            central,
            -(clearance * (weight / 70) ** 0.75 * exp(eta_clearance) / volume) * central,
        )
        observe("DV", mean=central / volume, error=additive(sigma))

    pharmacometrics_ir = one_compartment.to_ir()

    for model_ir in (formula_ir, builder_ir, pharmacometrics_ir):
        _assert_versioned_round_trip(model_ir)

    assert formula_ir.source == "formula"
    assert builder_ir.metadata["authoring_surface"] == "structured-builder"
    assert [prior.target for prior in builder_ir.priors] == ["time", "Intercept"]
    assert "priors" not in builder_ir.metadata

    assert pharmacometrics_ir.source == "pharmacometrics-dsl"
    assert [parameter.name for parameter in pharmacometrics_ir.parameters] == [
        "clearance",
        "volume",
        "sigma",
    ]
    assert all(parameter.transform == "log" for parameter in pharmacometrics_ir.parameters)
    assert pharmacometrics_ir.random_effects[0].terms == ("eta_clearance",)
    assert pharmacometrics_ir.predictors[0].name == "weight"
    assert pharmacometrics_ir.state_equations[0].state == "central"
    assert pharmacometrics_ir.events[0].event_type == "dose-mapping"
    assert pharmacometrics_ir.likelihoods[0].formulas["error"]["type"] == "additive"
    assert pharmacometrics_ir.outputs[0].name == "DV_prediction"


def test_pharmacometrics_ir_infers_declared_symbolic_error_parameters() -> None:
    state = State("amount")
    equation = d(state, -state)
    endpoint = observe("DV", mean=state, error=additive("sigma_deferred"))
    model_ir = compiled_model(
        "deferred-error",
        states=(state,),
        equations=(equation,),
        observations=(endpoint,),
    ).to_ir()

    sigma = model_ir.parameters[0]
    assert sigma.name == "sigma_deferred"
    assert sigma.initial is None
    assert sigma.role == "observation-error-parameter"
    assert model_ir.likelihoods[0].dependencies == ("amount", "sigma_deferred")
    _assert_versioned_round_trip(model_ir)


@pytest.mark.parametrize(
    "mean,error,code",
    [
        (Expr("unsupported-operation"), additive(1.0), "DSL-IR-EXPRESSION-UNSUPPORTED-001"),
        (
            Expr("constant", value=1.0),
            type(
                "CustomError",
                (),
                {"to_dict": lambda self: {"type": "custom-distribution"}},
            )(),
            "DSL-IR-ERROR-UNSUPPORTED-001",
        ),
    ],
)
def test_pharmacometrics_ir_rejects_opaque_semantics(
    mean: Expr,
    error: object,
    code: str,
) -> None:
    state = State("amount")
    equation = d(state, -state)
    endpoint = observe("DV", mean=mean, error=error)
    compiled = compiled_model(
        "unsupported",
        states=(state,),
        equations=(equation,),
        observations=(endpoint,),
    )

    with pytest.raises(DSLValidationError, match=code):
        compiled.to_ir()
