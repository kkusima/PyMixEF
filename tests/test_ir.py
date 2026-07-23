from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from pymixef.errors import IRValidationError, IRVersionError
from pymixef.ir import (
    CovarianceIR,
    EventIR,
    FixedEffectIR,
    LikelihoodIR,
    ModelIR,
    OutputIR,
    ParameterIR,
    PredictorIR,
    PriorIR,
    RandomEffectIR,
    StateEquationIR,
    TransformIR,
    diff_models,
    migrate_ir,
)


def example_ir() -> ModelIR:
    return ModelIR(
        name="random slope",
        source="formula",
        formula="y ~ time + (time | subject)",
        response="y",
        fixed_effects=(
            FixedEffectIR(name="Intercept", expression="1", columns=("Intercept",)),
            FixedEffectIR(name="time", expression="time", columns=("time",)),
        ),
        random_effects=(
            RandomEffectIR(
                terms=("Intercept", "time"),
                group="subject",
                covariance="unstructured",
            ),
        ),
        likelihoods=(LikelihoodIR(response="y", family="gaussian"),),
        covariance_structures=(
            CovarianceIR(structure="unstructured", target="subject", dimension=2),
        ),
        parameters=(
            ParameterIR(name="beta_time", initial=0.0),
            ParameterIR(name="sigma", initial=1.0, transform="log", support="positive"),
        ),
        data_schema={"columns": {"time": {"dtype": "float64"}}},
        metadata={"purpose": "unit-test"},
    )


def test_model_ir_round_trip_semantic_equality_and_hash() -> None:
    model = example_ir()
    restored = ModelIR.from_json(model.to_json(indent=2))
    assert restored == model
    assert hash(restored) == hash(model)
    assert restored.semantic_hash == model.semantic_hash
    assert json.loads(model.canonical_json()) == model.to_dict()


def test_ir_is_deeply_immutable() -> None:
    model = example_ir()
    with pytest.raises(TypeError):
        model.metadata["purpose"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        model.name = "changed"  # type: ignore[misc]


def test_ir_migration_is_explicit_and_does_not_mutate_input() -> None:
    legacy = {
        "schema_version": "0.1.0",
        "response": "y",
        "formula": "y ~ 1",
        "fixed": [],
    }
    migrated = migrate_ir(legacy)
    assert legacy["schema_version"] == "0.1.0"
    assert migrated["schema_version"] == "1.0.0"
    assert migrated["fixed_effects"] == []
    assert ModelIR.from_dict(legacy).response == "y"


def test_ir_rejects_missing_future_and_unknown_fields() -> None:
    with pytest.raises(IRVersionError, match="schema_version"):
        ModelIR.from_dict({})
    with pytest.raises(IRVersionError, match="migration path"):
        ModelIR.from_dict({"schema_version": "99.0.0"})
    document = example_ir().to_dict()
    document["secret_backend_syntax"] = "not semantic"
    with pytest.raises(IRValidationError, match="Unknown"):
        ModelIR.from_dict(document)


def test_model_diff_is_deterministic_and_classified() -> None:
    before = example_ir()
    document = before.to_dict()
    document["formula"] = "y ~ treatment + time + (time | subject)"
    document["fixed_effects"].append(
        FixedEffectIR(name="treatment", expression="treatment").to_dict()
    )
    document["family"] = "student-t"
    document["likelihoods"][0]["family"] = "student-t"
    document["covariance_structures"][0]["structure"] = "diagonal"
    after = ModelIR.from_dict(document)
    difference = diff_models(before, after)
    assert not difference.equal
    assert {"formulas", "families", "covariance"} <= set(difference.categories)
    assert difference.to_json() == before.diff(after).to_json()
    assert [entry.path for entry in difference.entries] == sorted(
        entry.path for entry in difference.entries
    )


def test_ir_rejects_duplicate_parameter_names() -> None:
    with pytest.raises(IRValidationError, match="unique"):
        ModelIR(
            parameters=(
                ParameterIR(name="theta"),
                ParameterIR(name="theta"),
            )
        )


def test_every_v1_node_type_round_trips_through_schema_validation() -> None:
    model = ModelIR(
        predictors=(PredictorIR(name="eta", expression="beta * x"),),
        state_equations=(StateEquationIR(state="central", rhs="-k * central"),),
        events=(EventIR(event_type="dose", target="central", fields={"amount": 100.0}),),
        transforms=(TransformIR(name="positive", kind="log", options={"offset": 0.0}),),
        priors=(
            PriorIR(
                target="beta",
                distribution="normal",
                parameters={"mean": 0.0, "sd": 1.0},
            ),
        ),
        outputs=(OutputIR(name="concentration", expression="central / volume"),),
        metadata={"nested": {"valid": [None, True, 1, 2.5, "text"]}},
    )

    payload = model.to_dict()
    restored = ModelIR.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored == model


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("family", 123),
        ("family", ""),
        ("name", []),
        ("fixed_effects", {}),
        ("random_effects", ()),
        ("data_schema", []),
        ("estimator", "ml"),
        ("metadata", []),
    ],
)
def test_from_dict_rejects_wrong_top_level_schema_types(
    field_name: str,
    invalid_value: object,
) -> None:
    document = example_ir().to_dict()
    document[field_name] = invalid_value

    with pytest.raises(IRValidationError):
        ModelIR.from_dict(document)


@pytest.mark.parametrize("field_name", ["family", "parameters", "metadata"])
def test_from_dict_rejects_missing_required_v1_fields(field_name: str) -> None:
    document = example_ir().to_dict()
    del document[field_name]

    with pytest.raises(IRValidationError, match="missing required"):
        ModelIR.from_dict(document)


def test_from_dict_rejects_non_object_root_and_non_string_version() -> None:
    with pytest.raises(IRValidationError, match="JSON object"):
        ModelIR.from_dict([])  # type: ignore[arg-type]

    document = example_ir().to_dict()
    document["schema_version"] = 1
    with pytest.raises(IRVersionError, match="string schema_version"):
        ModelIR.from_dict(document)


def test_from_dict_rejects_invalid_node_structures_without_coercion() -> None:
    document = example_ir().to_dict()
    document["fixed_effects"][0]["columns"] = "Intercept"
    with pytest.raises(IRValidationError, match="JSON array"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    document["fixed_effects"][0]["node_type"] = 123
    with pytest.raises(IRValidationError, match="string"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    del document["fixed_effects"][0]["annotations"]
    with pytest.raises(IRValidationError, match="missing required"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    document["fixed_effects"][0]["backend_hint"] = "unsafe"
    with pytest.raises(IRValidationError, match="Unknown"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    document["fixed_effects"] = [42]
    with pytest.raises(IRValidationError, match="JSON object"):
        ModelIR.from_dict(document)


def test_from_dict_rejects_boolean_numbers_and_invalid_nested_json() -> None:
    document = example_ir().to_dict()
    document["parameters"][0]["fixed"] = 1
    with pytest.raises(IRValidationError, match="boolean"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    document["parameters"][0]["initial"] = True
    with pytest.raises(IRValidationError, match="finite number"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    document["covariance_structures"][0]["dimension"] = True
    with pytest.raises(IRValidationError, match="integer"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    document["metadata"] = {"invalid": float("nan")}
    with pytest.raises(IRValidationError, match="finite JSON number"):
        ModelIR.from_dict(document)

    document = example_ir().to_dict()
    document["metadata"] = {1: "invalid-key"}
    with pytest.raises(IRValidationError, match="object key"):
        ModelIR.from_dict(document)
