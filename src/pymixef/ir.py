"""Versioned, immutable intermediate representation for scientific models.

The IR records mathematical meaning independently of a formula parser or
estimation backend.  JSON is canonicalized for stable equality, hashing, and
change-impact reports; pickle is intentionally not part of the persistence
contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar, cast

from .errors import IRValidationError, IRVersionError

MODEL_IR_SCHEMA_VERSION = "1.0.0"
IR_SCHEMA_VERSION = MODEL_IR_SCHEMA_VERSION
LEGACY_IR_SCHEMA_VERSION = "0.1.0"
SUPPORTED_IR_SCHEMA_VERSIONS = (LEGACY_IR_SCHEMA_VERSION, MODEL_IR_SCHEMA_VERSION)

JSONScalar = None | bool | int | float | str
FrozenJSON = JSONScalar | tuple["FrozenJSON", ...] | Mapping[str, "FrozenJSON"]


def _freeze(value: Any) -> FrozenJSON:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (value != value or abs(value) == float("inf")):
            raise IRValidationError("IR values must be finite JSON numbers.")
        return value
    if isinstance(value, Mapping):
        invalid_keys = [key for key in value if not isinstance(key, str)]
        if invalid_keys:
            raise IRValidationError(
                "IR JSON object keys must be strings.",
                code="IR-SCHEMA-TYPE-001",
                details={"actual_type": type(invalid_keys[0]).__name__},
            )
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "to_dict"):
        return _freeze(value.to_dict())
    raise IRValidationError(
        f"IR value of type {type(value).__name__!r} is not JSON serializable.",
        code="IR-SERIALIZATION-001",
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _thaw(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class IRNode:
    """Metadata common to all mathematical IR nodes."""

    dimensions: tuple[int | str, ...] = ()
    support: str = "real"
    transform: str = "identity"
    unit: str | None = None
    differentiability: str = "differentiable"
    dependencies: tuple[str, ...] = ()
    source_location: str | None = None
    annotations: Mapping[str, FrozenJSON] = field(default_factory=dict)

    node_type: ClassVar[str] = "node"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "dependencies", tuple(str(item) for item in self.dependencies))
        object.__setattr__(self, "annotations", _freeze(self.annotations))
        if self.differentiability not in {
            "differentiable",
            "piecewise-differentiable",
            "non-differentiable",
            "unknown",
        }:
            raise IRValidationError(
                f"Invalid differentiability status {self.differentiability!r}.",
                code="IR-DIFFERENTIABILITY-001",
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation including the node tag."""

        result: dict[str, Any] = {"node_type": self.node_type}
        for item in fields(self):
            result[item.name] = _thaw(getattr(self, item.name))
        return result


@dataclass(frozen=True, slots=True)
class ParameterIR(IRNode):
    """One parameter and its explicit optimizer-to-natural transform."""

    name: str
    initial: float | tuple[float, ...] | None = None
    bounds: tuple[float | None, float | None] | None = None
    fixed: bool = False
    role: str = "parameter"

    node_type: ClassVar[str] = "parameter"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        if not self.name:
            raise IRValidationError("Parameter names cannot be empty.")
        if isinstance(self.initial, (list, tuple)):
            object.__setattr__(self, "initial", tuple(float(item) for item in self.initial))
        if self.bounds is not None:
            if len(self.bounds) != 2:
                raise IRValidationError("Parameter bounds require lower and upper values.")
            lower, upper = self.bounds
            if lower is not None and upper is not None and lower >= upper:
                raise IRValidationError("Parameter bounds require lower < upper.")
            object.__setattr__(self, "bounds", (lower, upper))


@dataclass(frozen=True, slots=True)
class FixedEffectIR(IRNode):
    """A resolved fixed-effect term and its generated model-matrix columns."""

    name: str
    expression: str
    columns: tuple[str, ...] = ()

    node_type: ClassVar[str] = "fixed_effect"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        object.__setattr__(self, "columns", tuple(self.columns))


@dataclass(frozen=True, slots=True)
class RandomEffectIR(IRNode):
    """A random-effect block with explicit grouping and correlation semantics."""

    terms: tuple[str, ...]
    group: str
    correlated: bool = True
    covariance: str = "unstructured"
    known_matrix: tuple[tuple[float, ...], ...] | None = None

    node_type: ClassVar[str] = "random_effect"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.group or not self.terms:
            raise IRValidationError("Random effects require terms and a grouping expression.")
        if self.known_matrix is not None:
            object.__setattr__(
                self,
                "known_matrix",
                tuple(tuple(float(value) for value in row) for row in self.known_matrix),
            )


@dataclass(frozen=True, slots=True)
class PredictorIR(IRNode):
    """A safe, resolved predictor expression."""

    name: str
    expression: str
    kind: str = "derived"

    node_type: ClassVar[str] = "predictor"


@dataclass(frozen=True, slots=True)
class LikelihoodIR(IRNode):
    """An observation likelihood component and its distributional predictors."""

    response: str
    family: str = "gaussian"
    link: str = "identity"
    component: str = "conditional"
    formulas: Mapping[str, FrozenJSON] = field(default_factory=dict)

    node_type: ClassVar[str] = "likelihood"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        object.__setattr__(self, "formulas", _freeze(self.formulas))


@dataclass(frozen=True, slots=True)
class CovarianceIR(IRNode):
    """A covariance block attached to random effects or observations."""

    structure: str
    target: str = "random-effects"
    dimension: int | None = None
    index: str | None = None
    group: str | None = None
    options: Mapping[str, FrozenJSON] = field(default_factory=dict)

    node_type: ClassVar[str] = "covariance"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        object.__setattr__(self, "options", _freeze(self.options))
        if self.dimension is not None and self.dimension < 1:
            raise IRValidationError("Covariance dimensions must be positive.")


@dataclass(frozen=True, slots=True)
class TransformIR(IRNode):
    """A named transform node used by one or more parameters."""

    name: str
    kind: str
    options: Mapping[str, FrozenJSON] = field(default_factory=dict)

    node_type: ClassVar[str] = "transform"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        object.__setattr__(self, "options", _freeze(self.options))


@dataclass(frozen=True, slots=True)
class PriorIR(IRNode):
    """A prior distribution explicitly attached to a parameter."""

    target: str
    distribution: str
    parameters: Mapping[str, FrozenJSON] = field(default_factory=dict)

    node_type: ClassVar[str] = "prior"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        object.__setattr__(self, "parameters", _freeze(self.parameters))


@dataclass(frozen=True, slots=True)
class StateEquationIR(IRNode):
    """A differential or algebraic state equation."""

    state: str
    rhs: str
    equation_kind: str = "ode"
    initial: str | float = 0.0

    node_type: ClassVar[str] = "state_equation"


@dataclass(frozen=True, slots=True)
class EventIR(IRNode):
    """A canonical event declaration."""

    event_type: str
    target: str | None = None
    fields: Mapping[str, FrozenJSON] = field(default_factory=dict)

    node_type: ClassVar[str] = "event"

    def __post_init__(self) -> None:
        IRNode.__post_init__(self)
        object.__setattr__(self, "fields", _freeze(self.fields))


@dataclass(frozen=True, slots=True)
class OutputIR(IRNode):
    """A named model output or derived quantity."""

    name: str
    expression: str
    output_kind: str = "prediction"

    node_type: ClassVar[str] = "output"


_NODE_TYPES: dict[str, type[IRNode]] = {
    node.node_type: node
    for node in (
        ParameterIR,
        FixedEffectIR,
        RandomEffectIR,
        PredictorIR,
        LikelihoodIR,
        CovarianceIR,
        TransformIR,
        PriorIR,
        StateEquationIR,
        EventIR,
        OutputIR,
    )
}

_TOP_LEVEL_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "family",
        "fixed_effects",
        "random_effects",
        "predictors",
        "likelihoods",
        "covariance_structures",
        "state_equations",
        "events",
        "parameters",
        "transforms",
        "priors",
        "outputs",
        "data_schema",
        "estimator",
        "metadata",
    }
)
_TOP_LEVEL_FIELDS = _TOP_LEVEL_REQUIRED_FIELDS | {
    "name",
    "source",
    "formula",
    "response",
}
_NODE_METADATA_FIELDS = frozenset(
    {
        "dimensions",
        "support",
        "transform",
        "unit",
        "differentiability",
        "dependencies",
        "source_location",
        "annotations",
    }
)
_NODE_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "fixed_effect": frozenset({"node_type", "name", "expression", "columns"}),
    "random_effect": frozenset({"node_type", "terms", "group", "correlated", "covariance"}),
    "predictor": frozenset({"node_type", "name", "expression", "kind"}),
    "likelihood": frozenset({"node_type", "response", "family", "link", "component", "formulas"}),
    "covariance": frozenset(
        {"node_type", "structure", "target", "dimension", "index", "group", "options"}
    ),
    "state_equation": frozenset({"node_type", "state", "rhs", "equation_kind", "initial"}),
    "event": frozenset({"node_type", "event_type", "target", "fields"}),
    "parameter": frozenset({"node_type", "name", "initial", "bounds", "fixed", "role"}),
    "transform": frozenset({"node_type", "name", "kind", "options"}),
    "prior": frozenset({"node_type", "target", "distribution", "parameters"}),
    "output": frozenset({"node_type", "name", "expression", "output_kind"}),
}
_NODE_OPTIONAL_FIELDS: dict[str, frozenset[str]] = {
    "random_effect": frozenset({"known_matrix"}),
}
_DIFFERENTIABILITY_VALUES = frozenset(
    {
        "differentiable",
        "piecewise-differentiable",
        "non-differentiable",
        "unknown",
    }
)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _schema_type_error(path: str, expected: str, value: Any) -> None:
    raise IRValidationError(
        f"{path} must be {expected}; got {_json_type_name(value)}.",
        code="IR-SCHEMA-TYPE-001",
        details={
            "path": path,
            "expected": expected,
            "actual_type": _json_type_name(value),
        },
    )


def _expect_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _schema_type_error(path, "a JSON object", value)
    invalid_keys = [key for key in value if not isinstance(key, str)]
    if invalid_keys:
        _schema_type_error(f"{path} object key", "a string", invalid_keys[0])
    return cast(Mapping[str, Any], value)


def _expect_array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _schema_type_error(path, "a JSON array", value)
    return cast(list[Any], value)


def _expect_string(value: Any, path: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, str):
        _schema_type_error(path, "a string", value)
    if nonempty and not value:
        raise IRValidationError(
            f"{path} must not be empty.",
            code="IR-SCHEMA-VALUE-001",
            details={"path": path, "constraint": "minLength", "minimum": 1},
        )


def _expect_optional_string(value: Any, path: str) -> None:
    if value is not None:
        _expect_string(value, path)


def _expect_boolean(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        _schema_type_error(path, "a boolean", value)


def _expect_number(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _schema_type_error(path, "a finite number", value)
    if isinstance(value, float) and not math.isfinite(value):
        raise IRValidationError(
            f"{path} must be a finite JSON number.",
            code="IR-SCHEMA-VALUE-001",
            details={"path": path, "constraint": "finite"},
        )


def _expect_integer(value: Any, path: str, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _schema_type_error(path, "an integer", value)
    if minimum is not None and value < minimum:
        raise IRValidationError(
            f"{path} must be at least {minimum}.",
            code="IR-SCHEMA-VALUE-001",
            details={"path": path, "constraint": "minimum", "minimum": minimum},
        )


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        _expect_number(value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        mapping = _expect_object(value, path)
        for key, item in mapping.items():
            _validate_json_value(item, f"{path}.{key}")
        return
    _schema_type_error(path, "a JSON value", value)


def _expect_json_object(value: Any, path: str) -> Mapping[str, Any]:
    mapping = _expect_object(value, path)
    for key, item in mapping.items():
        _validate_json_value(item, f"{path}.{key}")
    return mapping


def _validate_fields(
    value: Mapping[str, Any],
    path: str,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise IRValidationError(
            f"{path} is missing required fields: {', '.join(missing)}.",
            code="IR-SCHEMA-REQUIRED-001",
            details={"path": path, "fields": missing},
        )
    unknown = sorted(value.keys() - allowed)
    if unknown:
        raise IRValidationError(
            f"Unknown fields at {path}: {', '.join(unknown)}.",
            code="IR-UNKNOWN-FIELD-001",
            details={"path": path, "fields": unknown},
        )


def _validate_string_array(value: Any, path: str, *, nonempty: bool = False) -> None:
    items = _expect_array(value, path)
    if nonempty and not items:
        raise IRValidationError(
            f"{path} must contain at least one item.",
            code="IR-SCHEMA-VALUE-001",
            details={"path": path, "constraint": "minItems", "minimum": 1},
        )
    for index, item in enumerate(items):
        _expect_string(item, f"{path}[{index}]")


def _validate_node_metadata(value: Mapping[str, Any], path: str) -> None:
    dimensions = _expect_array(value["dimensions"], f"{path}.dimensions")
    for index, item in enumerate(dimensions):
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            _schema_type_error(
                f"{path}.dimensions[{index}]",
                "an integer or string",
                item,
            )
    _expect_string(value["support"], f"{path}.support")
    _expect_string(value["transform"], f"{path}.transform")
    _expect_optional_string(value["unit"], f"{path}.unit")
    _expect_string(value["differentiability"], f"{path}.differentiability")
    if value["differentiability"] not in _DIFFERENTIABILITY_VALUES:
        raise IRValidationError(
            f"{path}.differentiability has an unsupported value.",
            code="IR-SCHEMA-VALUE-001",
            details={
                "path": f"{path}.differentiability",
                "allowed": sorted(_DIFFERENTIABILITY_VALUES),
            },
        )
    _validate_string_array(value["dependencies"], f"{path}.dependencies")
    _expect_optional_string(value["source_location"], f"{path}.source_location")
    _expect_json_object(value["annotations"], f"{path}.annotations")


def _validate_parameter(value: Mapping[str, Any], path: str) -> None:
    _expect_string(value["name"], f"{path}.name", nonempty=True)
    initial = value["initial"]
    if initial is not None:
        if isinstance(initial, list):
            for index, item in enumerate(initial):
                _expect_number(item, f"{path}.initial[{index}]")
        else:
            _expect_number(initial, f"{path}.initial")
    bounds = value["bounds"]
    if bounds is not None:
        items = _expect_array(bounds, f"{path}.bounds")
        if len(items) != 2:
            raise IRValidationError(
                f"{path}.bounds must contain exactly two items.",
                code="IR-SCHEMA-VALUE-001",
                details={
                    "path": f"{path}.bounds",
                    "constraint": "length",
                    "length": 2,
                },
            )
        for index, item in enumerate(items):
            if item is not None:
                _expect_number(item, f"{path}.bounds[{index}]")
    _expect_boolean(value["fixed"], f"{path}.fixed")
    _expect_string(value["role"], f"{path}.role")


def _validate_node_document(value: Any, expected_node_type: str, path: str) -> None:
    node = _expect_object(value, path)
    required = _NODE_METADATA_FIELDS | _NODE_REQUIRED_FIELDS[expected_node_type]
    allowed = required | _NODE_OPTIONAL_FIELDS.get(expected_node_type, frozenset())
    _validate_fields(node, path, required=required, allowed=allowed)
    _expect_string(node["node_type"], f"{path}.node_type")
    if node["node_type"] != expected_node_type:
        raise IRValidationError(
            f"Unexpected IR node type {node['node_type']!r}; expected {expected_node_type!r}.",
            code="IR-NODE-TYPE-001",
            details={
                "path": f"{path}.node_type",
                "expected": expected_node_type,
                "actual": node["node_type"],
            },
        )
    _validate_node_metadata(node, path)

    if expected_node_type == "fixed_effect":
        _expect_string(node["name"], f"{path}.name")
        _expect_string(node["expression"], f"{path}.expression")
        _validate_string_array(node["columns"], f"{path}.columns")
    elif expected_node_type == "random_effect":
        _validate_string_array(node["terms"], f"{path}.terms", nonempty=True)
        _expect_string(node["group"], f"{path}.group", nonempty=True)
        _expect_boolean(node["correlated"], f"{path}.correlated")
        _expect_string(node["covariance"], f"{path}.covariance")
        known_matrix = node.get("known_matrix")
        if known_matrix is not None:
            rows = _expect_array(known_matrix, f"{path}.known_matrix")
            for row_index, row in enumerate(rows):
                values = _expect_array(row, f"{path}.known_matrix[{row_index}]")
                for column_index, item in enumerate(values):
                    _expect_number(
                        item,
                        f"{path}.known_matrix[{row_index}][{column_index}]",
                    )
    elif expected_node_type == "predictor":
        _expect_string(node["name"], f"{path}.name")
        _expect_string(node["expression"], f"{path}.expression")
        _expect_string(node["kind"], f"{path}.kind")
    elif expected_node_type == "likelihood":
        _expect_string(node["response"], f"{path}.response")
        _expect_string(node["family"], f"{path}.family")
        _expect_string(node["link"], f"{path}.link")
        _expect_string(node["component"], f"{path}.component")
        _expect_json_object(node["formulas"], f"{path}.formulas")
    elif expected_node_type == "covariance":
        _expect_string(node["structure"], f"{path}.structure")
        _expect_string(node["target"], f"{path}.target")
        if node["dimension"] is not None:
            _expect_integer(node["dimension"], f"{path}.dimension", minimum=1)
        _expect_optional_string(node["index"], f"{path}.index")
        _expect_optional_string(node["group"], f"{path}.group")
        _expect_json_object(node["options"], f"{path}.options")
    elif expected_node_type == "state_equation":
        _expect_string(node["state"], f"{path}.state")
        _expect_string(node["rhs"], f"{path}.rhs")
        _expect_string(node["equation_kind"], f"{path}.equation_kind")
        if not isinstance(node["initial"], str):
            _expect_number(node["initial"], f"{path}.initial")
    elif expected_node_type == "event":
        _expect_string(node["event_type"], f"{path}.event_type")
        _expect_optional_string(node["target"], f"{path}.target")
        _expect_json_object(node["fields"], f"{path}.fields")
    elif expected_node_type == "parameter":
        _validate_parameter(node, path)
    elif expected_node_type == "transform":
        _expect_string(node["name"], f"{path}.name")
        _expect_string(node["kind"], f"{path}.kind")
        _expect_json_object(node["options"], f"{path}.options")
    elif expected_node_type == "prior":
        _expect_string(node["target"], f"{path}.target")
        _expect_string(node["distribution"], f"{path}.distribution")
        _expect_json_object(node["parameters"], f"{path}.parameters")
    elif expected_node_type == "output":
        _expect_string(node["name"], f"{path}.name")
        _expect_string(node["expression"], f"{path}.expression")
        _expect_string(node["output_kind"], f"{path}.output_kind")


def _validate_model_ir_v1_document(document: Mapping[str, Any]) -> None:
    value = _expect_object(document, "$")
    _validate_fields(
        value,
        "$",
        required=_TOP_LEVEL_REQUIRED_FIELDS,
        allowed=frozenset(_TOP_LEVEL_FIELDS),
    )
    if value["schema_version"] != MODEL_IR_SCHEMA_VERSION:
        raise IRVersionError(
            f"Expected IR schema {MODEL_IR_SCHEMA_VERSION!r}; got {value['schema_version']!r}.",
            code="IR-VERSION-001",
        )
    for key in ("name", "source", "formula", "response"):
        if key in value:
            _expect_optional_string(value[key], f"$.{key}")
    _expect_string(value["family"], "$.family", nonempty=True)

    sequence_types: dict[str, str] = {
        "fixed_effects": "fixed_effect",
        "random_effects": "random_effect",
        "predictors": "predictor",
        "likelihoods": "likelihood",
        "covariance_structures": "covariance",
        "state_equations": "state_equation",
        "events": "event",
        "parameters": "parameter",
        "transforms": "transform",
        "priors": "prior",
        "outputs": "output",
    }
    for key, expected_node_type in sequence_types.items():
        nodes = _expect_array(value[key], f"$.{key}")
        for index, node in enumerate(nodes):
            _validate_node_document(node, expected_node_type, f"$.{key}[{index}]")
    _expect_json_object(value["data_schema"], "$.data_schema")
    _expect_json_object(value["estimator"], "$.estimator")
    _expect_json_object(value["metadata"], "$.metadata")


def _node_from_dict(value: Mapping[str, Any], expected: type[IRNode]) -> IRNode:
    if not isinstance(value, Mapping):
        raise IRValidationError(
            f"Expected a JSON object for {expected.node_type}; got {type(value).__name__}.",
            code="IR-NODE-FIELDS-001",
        )
    node_type = value.get("node_type")
    if not isinstance(node_type, str):
        raise IRValidationError(
            f"Missing or invalid node_type for {expected.node_type}.",
            code="IR-NODE-TYPE-001",
        )
    implementation = _NODE_TYPES.get(node_type)
    if implementation is None or not issubclass(implementation, expected):
        raise IRValidationError(
            f"Unexpected IR node type {node_type!r}; expected {expected.node_type!r}.",
            code="IR-NODE-TYPE-001",
        )
    arguments = dict(value)
    arguments.pop("node_type", None)
    try:
        return implementation(**arguments)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IRValidationError(
            f"Invalid {node_type} node fields: {exc}.",
            code="IR-NODE-FIELDS-001",
        ) from exc


@dataclass(frozen=True, slots=True, eq=False)
class ModelIR:
    """Complete backend-neutral scientific model graph."""

    schema_version: str = MODEL_IR_SCHEMA_VERSION
    name: str | None = None
    source: str | None = None
    formula: str | None = None
    response: str | None = None
    family: str = "gaussian"
    fixed_effects: tuple[FixedEffectIR, ...] = ()
    random_effects: tuple[RandomEffectIR, ...] = ()
    predictors: tuple[PredictorIR, ...] = ()
    likelihoods: tuple[LikelihoodIR, ...] = ()
    covariance_structures: tuple[CovarianceIR, ...] = ()
    state_equations: tuple[StateEquationIR, ...] = ()
    events: tuple[EventIR, ...] = ()
    parameters: tuple[ParameterIR, ...] = ()
    transforms: tuple[TransformIR, ...] = ()
    priors: tuple[PriorIR, ...] = ()
    outputs: tuple[OutputIR, ...] = ()
    data_schema: Mapping[str, FrozenJSON] = field(default_factory=dict)
    estimator: Mapping[str, FrozenJSON] = field(default_factory=dict)
    metadata: Mapping[str, FrozenJSON] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_IR_SCHEMA_VERSION:
            raise IRVersionError(
                f"ModelIR construction requires schema version {MODEL_IR_SCHEMA_VERSION}; "
                f"got {self.schema_version!r}. Use ModelIR.from_dict() to migrate old data.",
                code="IR-VERSION-CONSTRUCTION-001",
            )
        node_fields: tuple[tuple[str, type[IRNode]], ...] = (
            ("fixed_effects", FixedEffectIR),
            ("random_effects", RandomEffectIR),
            ("predictors", PredictorIR),
            ("likelihoods", LikelihoodIR),
            ("covariance_structures", CovarianceIR),
            ("state_equations", StateEquationIR),
            ("events", EventIR),
            ("parameters", ParameterIR),
            ("transforms", TransformIR),
            ("priors", PriorIR),
            ("outputs", OutputIR),
        )
        for field_name, expected in node_fields:
            value = tuple(getattr(self, field_name))
            if not all(isinstance(item, expected) for item in value):
                raise IRValidationError(
                    f"{field_name} contains an invalid node.",
                    code="IR-NODE-TYPE-001",
                )
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "data_schema", _freeze(self.data_schema))
        object.__setattr__(self, "estimator", _freeze(self.estimator))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        parameter_names = [item.name for item in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise IRValidationError(
                "Parameter names must be unique.",
                code="IR-DUPLICATE-PARAMETER-001",
            )
        if self.response and self.likelihoods:
            responses = {item.response for item in self.likelihoods}
            if self.response not in responses:
                raise IRValidationError(
                    "Top-level response does not match any likelihood component.",
                    code="IR-RESPONSE-MISMATCH-001",
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible schema-v1 document."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "source": self.source,
            "formula": self.formula,
            "response": self.response,
            "family": self.family,
            "fixed_effects": [item.to_dict() for item in self.fixed_effects],
            "random_effects": [item.to_dict() for item in self.random_effects],
            "predictors": [item.to_dict() for item in self.predictors],
            "likelihoods": [item.to_dict() for item in self.likelihoods],
            "covariance_structures": [item.to_dict() for item in self.covariance_structures],
            "state_equations": [item.to_dict() for item in self.state_equations],
            "events": [item.to_dict() for item in self.events],
            "parameters": [item.to_dict() for item in self.parameters],
            "transforms": [item.to_dict() for item in self.transforms],
            "priors": [item.to_dict() for item in self.priors],
            "outputs": [item.to_dict() for item in self.outputs],
            "data_schema": _thaw(self.data_schema),
            "estimator": _thaw(self.estimator),
            "metadata": _thaw(self.metadata),
        }

    def canonical_json(self) -> str:
        """Return deterministic compact JSON used for identity and hashes."""

        return _canonical_json(self.to_dict())

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the model to JSON."""

        if indent is None:
            return self.canonical_json()
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @property
    def semantic_hash(self) -> str:
        """SHA-256 digest of the canonical mathematical representation."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def hash(self) -> str:
        """Alias for :attr:`semantic_hash`."""

        return self.semantic_hash

    def semantically_equal(self, other: object) -> bool:
        """Whether two models have identical canonical scientific meaning."""

        return isinstance(other, ModelIR) and self.canonical_json() == other.canonical_json()

    def __eq__(self, other: object) -> bool:
        return self.semantically_equal(other)

    def __hash__(self) -> int:
        return int.from_bytes(bytes.fromhex(self.semantic_hash[:16]), "big", signed=False)

    @classmethod
    def from_dict(
        cls,
        document: Mapping[str, Any],
        *,
        migrate: bool = True,
    ) -> ModelIR:
        """Validate, optionally migrate, and construct a model IR document."""

        value = dict(_expect_object(document, "$"))
        version = value.get("schema_version")
        if not isinstance(version, str):
            raise IRVersionError(
                "Serialized model IR is missing a string schema_version.",
                code="IR-VERSION-MISSING-001",
            )
        if version != MODEL_IR_SCHEMA_VERSION:
            if not migrate:
                raise IRVersionError(
                    f"IR schema {version!r} is not the current supported version.",
                    code="IR-MIGRATION-REQUIRED-001",
                )
            value = migrate_ir(value)
        _validate_model_ir_v1_document(value)
        sequence_types: dict[str, type[IRNode]] = {
            "fixed_effects": FixedEffectIR,
            "random_effects": RandomEffectIR,
            "predictors": PredictorIR,
            "likelihoods": LikelihoodIR,
            "covariance_structures": CovarianceIR,
            "state_equations": StateEquationIR,
            "events": EventIR,
            "parameters": ParameterIR,
            "transforms": TransformIR,
            "priors": PriorIR,
            "outputs": OutputIR,
        }
        arguments = dict(value)
        for key, expected in sequence_types.items():
            raw = arguments[key]
            arguments[key] = tuple(
                item if isinstance(item, expected) else _node_from_dict(item, expected)
                for item in raw
            )
        known = {item.name for item in fields(cls)}
        extras = sorted(set(arguments) - known)
        if extras:
            raise IRValidationError(
                f"Unknown model IR fields: {', '.join(extras)}.",
                code="IR-UNKNOWN-FIELD-001",
                details={"fields": extras},
            )
        try:
            return cls(**arguments)
        except (TypeError, ValueError, OverflowError) as exc:
            raise IRValidationError(
                f"Invalid model IR fields: {exc}.",
                code="IR-FIELDS-001",
            ) from exc

    @classmethod
    def from_json(cls, document: str | bytes, *, migrate: bool = True) -> ModelIR:
        """Load a model from a JSON string or bytes."""

        try:
            value = json.loads(document)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise IRValidationError(
                "Model IR is not valid JSON.",
                code="IR-JSON-001",
            ) from exc
        if not isinstance(value, Mapping):
            raise IRValidationError("Model IR JSON root must be an object.")
        return cls.from_dict(value, migrate=migrate)

    def diff(self, other: ModelIR) -> ModelDiff:
        """Return a deterministic semantic change report."""

        return diff_models(self, other)


Migration = Callable[[dict[str, Any]], dict[str, Any]]
_MIGRATIONS: dict[tuple[str, str], Migration] = {}


def register_ir_migration(
    from_version: str,
    to_version: str,
    migration: Migration,
    *,
    replace: bool = False,
) -> None:
    """Register one explicit forward migration edge."""

    if from_version == to_version:
        raise IRVersionError("A migration must change the schema version.")
    edge = (from_version, to_version)
    if edge in _MIGRATIONS and not replace:
        raise IRVersionError(
            f"Migration {from_version!r} -> {to_version!r} already exists.",
            code="IR-MIGRATION-DUPLICATE-001",
        )
    _MIGRATIONS[edge] = migration


def _migrate_0_1_to_1_0(document: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(document)
    migrated["schema_version"] = MODEL_IR_SCHEMA_VERSION
    aliases = {
        "fixed": "fixed_effects",
        "random": "random_effects",
        "covariance": "covariance_structures",
        "states": "state_equations",
    }
    for old, new in aliases.items():
        if old in migrated:
            if new in migrated:
                raise IRVersionError(
                    f"Legacy IR contains both {old!r} and {new!r}.",
                    code="IR-MIGRATION-AMBIGUOUS-001",
                )
            migrated[new] = migrated.pop(old)
    migrated.setdefault("family", "gaussian")
    for key in (
        "fixed_effects",
        "random_effects",
        "predictors",
        "likelihoods",
        "covariance_structures",
        "state_equations",
        "events",
        "parameters",
        "transforms",
        "priors",
        "outputs",
    ):
        migrated.setdefault(key, [])
    migrated.setdefault("data_schema", {})
    migrated.setdefault("estimator", {})
    migrated.setdefault("metadata", {})
    return migrated


register_ir_migration(LEGACY_IR_SCHEMA_VERSION, MODEL_IR_SCHEMA_VERSION, _migrate_0_1_to_1_0)


def migrate_ir(
    document: Mapping[str, Any],
    *,
    target_version: str = MODEL_IR_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Migrate a serialized IR through registered explicit forward edges.

    Unknown versions and downgrade attempts are rejected rather than guessed.
    The input mapping is never mutated.
    """

    value = _thaw(_freeze(document))
    version = value.get("schema_version")
    if not isinstance(version, str):
        raise IRVersionError(
            "Cannot migrate an IR without schema_version.",
            code="IR-VERSION-MISSING-001",
        )
    if target_version != MODEL_IR_SCHEMA_VERSION:
        raise IRVersionError(
            f"Unsupported migration target {target_version!r}.",
            code="IR-MIGRATION-TARGET-001",
        )
    visited: set[str] = set()
    while version != target_version:
        if version in visited:
            raise IRVersionError("IR migration graph contains a cycle.")
        visited.add(version)
        candidates = sorted(edge for edge in _MIGRATIONS if edge[0] == version)
        if len(candidates) != 1:
            raise IRVersionError(
                f"No unambiguous migration path from schema {version!r}.",
                code="IR-MIGRATION-PATH-001",
                details={"supported_versions": list(SUPPORTED_IR_SCHEMA_VERSIONS)},
            )
        edge = candidates[0]
        migrated = _MIGRATIONS[edge](dict(value))
        if not isinstance(migrated, dict) or migrated.get("schema_version") != edge[1]:
            raise IRVersionError(
                f"Migration {edge[0]!r} -> {edge[1]!r} violated its version contract.",
                code="IR-MIGRATION-CONTRACT-001",
            )
        value = migrated
        version = edge[1]
    return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One atomic deterministic change between two model IR documents."""

    category: str
    path: str
    change: str
    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "path": self.path,
            "change": self.change,
            "before": _thaw(self.before),
            "after": _thaw(self.after),
        }


@dataclass(frozen=True, slots=True)
class ModelDiff:
    """Serializable model change-impact report."""

    before_hash: str
    after_hash: str
    entries: tuple[DiffEntry, ...] = ()

    @property
    def equal(self) -> bool:
        return not self.entries

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({entry.category for entry in self.entries}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "equal": self.equal,
            "categories": list(self.categories),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )


def _diff_category(path: str) -> str:
    first = path.lstrip("/").split("/", 1)[0]
    mapping = {
        "formula": "formulas",
        "fixed_effects": "formulas",
        "random_effects": "formulas",
        "likelihoods": "families",
        "family": "families",
        "covariance_structures": "covariance",
        "events": "events",
        "priors": "priors",
        "estimator": "estimators",
        "data_schema": "data_schema",
        "parameters": "parameters",
        "transforms": "parameters",
        "state_equations": "states",
        "outputs": "outputs",
    }
    return mapping.get(first, "metadata" if first == "metadata" else "model")


def _walk_diff(before: Any, after: Any, path: str, output: list[DiffEntry]) -> None:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after)):
            child = f"{path}/{key}"
            if key not in before:
                output.append(DiffEntry(_diff_category(child), child, "added", None, after[key]))
            elif key not in after:
                output.append(DiffEntry(_diff_category(child), child, "removed", before[key], None))
            else:
                _walk_diff(before[key], after[key], child, output)
        return
    if isinstance(before, list) and isinstance(after, list):
        maximum = max(len(before), len(after))
        for index in range(maximum):
            child = f"{path}/{index}"
            if index >= len(before):
                output.append(DiffEntry(_diff_category(child), child, "added", None, after[index]))
            elif index >= len(after):
                output.append(
                    DiffEntry(_diff_category(child), child, "removed", before[index], None)
                )
            else:
                _walk_diff(before[index], after[index], child, output)
        return
    if before != after:
        output.append(DiffEntry(_diff_category(path), path or "/", "modified", before, after))


def diff_models(before: ModelIR, after: ModelIR) -> ModelDiff:
    """Compare two model IR objects and classify every semantic change."""

    if not isinstance(before, ModelIR) or not isinstance(after, ModelIR):
        raise TypeError("diff_models requires two ModelIR objects")
    entries: list[DiffEntry] = []
    _walk_diff(before.to_dict(), after.to_dict(), "", entries)
    return ModelDiff(before.semantic_hash, after.semantic_hash, tuple(entries))


model_diff = diff_models


__all__ = [
    "IR_SCHEMA_VERSION",
    "LEGACY_IR_SCHEMA_VERSION",
    "MODEL_IR_SCHEMA_VERSION",
    "SUPPORTED_IR_SCHEMA_VERSIONS",
    "CovarianceIR",
    "DiffEntry",
    "EventIR",
    "FixedEffectIR",
    "IRNode",
    "LikelihoodIR",
    "ModelDiff",
    "ModelIR",
    "OutputIR",
    "ParameterIR",
    "PredictorIR",
    "PriorIR",
    "RandomEffectIR",
    "StateEquationIR",
    "TransformIR",
    "diff_models",
    "migrate_ir",
    "model_diff",
    "register_ir_migration",
]
