"""Typed, introspectable declarations for pharmacometric models.

This module provides a small Python-native declaration language.  Arithmetic
builds immutable expression trees; it is not evaluated with ``eval`` and it
does not embed optimizer-specific control statements.

The :func:`model` decorator intentionally does *not* claim that arbitrary
Python functions are safe to inspect.  A decorated function is executed lazily
when it is compiled, just like any other Python function.  Projects that must
inspect untrusted model text should deserialize a validated model IR instead of
importing and executing that text.  Once compiled, the returned
:class:`CompiledModel` is data-only and fully introspectable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import update_wrapper
from inspect import signature
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, ParamSpec, TypeVar, overload

import numpy as np

from ..ir import (
    CovarianceIR,
    EventIR,
    LikelihoodIR,
    ModelIR,
    OutputIR,
    ParameterIR,
    PredictorIR,
    RandomEffectIR,
    StateEquationIR,
    TransformIR,
)


class DSLValidationError(ValueError):
    """Raised when declarations do not form a valid pharmacometric model."""

    code = "DSL-INVALID-001"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, frozenset):
        return [_serialize(item) for item in sorted(value, key=repr)]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class Expr:
    """An immutable symbolic expression node."""

    operation: str
    arguments: tuple[Expr, ...] = ()
    value: float | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    __array_priority__ = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def _binary(self, operation: str, other: ExpressionLike) -> Expr:
        return Expr(operation, (self, as_expr(other)))

    def __add__(self, other: ExpressionLike) -> Expr:
        return self._binary("add", other)

    def __radd__(self, other: ExpressionLike) -> Expr:
        return as_expr(other)._binary("add", self)

    def __sub__(self, other: ExpressionLike) -> Expr:
        return self._binary("subtract", other)

    def __rsub__(self, other: ExpressionLike) -> Expr:
        return as_expr(other)._binary("subtract", self)

    def __mul__(self, other: ExpressionLike) -> Expr:
        return self._binary("multiply", other)

    def __rmul__(self, other: ExpressionLike) -> Expr:
        return as_expr(other)._binary("multiply", self)

    def __truediv__(self, other: ExpressionLike) -> Expr:
        return self._binary("divide", other)

    def __rtruediv__(self, other: ExpressionLike) -> Expr:
        return as_expr(other)._binary("divide", self)

    def __pow__(self, other: ExpressionLike) -> Expr:
        return self._binary("power", other)

    def __rpow__(self, other: ExpressionLike) -> Expr:
        return as_expr(other)._binary("power", self)

    def __neg__(self) -> Expr:
        return Expr("negative", (self,))

    def __pos__(self) -> Expr:
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the expression tree."""

        result: dict[str, Any] = {"operation": self.operation}
        if self.value is not None:
            result["value"] = self.value
        if self.arguments:
            result["arguments"] = [argument.to_dict() for argument in self.arguments]
        if self.metadata:
            result["metadata"] = _serialize(self.metadata)
        return result

    def format(self) -> str:
        """Return a deterministic, human-readable mathematical expression."""

        if self.operation == "constant":
            return repr(self.value)
        if self.operation in {"parameter", "eta", "state", "symbol"}:
            return str(self.value)
        if self.operation == "negative":
            return f"-({self.arguments[0].format()})"
        infix = {
            "add": "+",
            "subtract": "-",
            "multiply": "*",
            "divide": "/",
            "power": "**",
        }
        if self.operation in infix:
            left, right = self.arguments
            return f"({left.format()} {infix[self.operation]} {right.format()})"
        if len(self.arguments) == 1:
            return f"{self.operation}({self.arguments[0].format()})"
        return f"{self.operation}({', '.join(argument.format() for argument in self.arguments)})"

    def evaluate(self, values: Mapping[str, float]) -> float:
        """Evaluate using an explicit symbol table.

        This method traverses the already-built expression tree.  It never
        evaluates source code.  Missing names raise ``KeyError``.
        """

        if self.operation == "constant":
            assert self.value is not None
            return float(self.value)
        if self.operation in {"parameter", "eta", "state", "symbol"}:
            return float(values[str(self.value)])
        args = tuple(argument.evaluate(values) for argument in self.arguments)
        if self.operation == "add":
            return args[0] + args[1]
        if self.operation == "subtract":
            return args[0] - args[1]
        if self.operation == "multiply":
            return args[0] * args[1]
        if self.operation == "divide":
            return args[0] / args[1]
        if self.operation == "power":
            return args[0] ** args[1]
        if self.operation == "negative":
            return -args[0]
        functions: dict[str, Callable[[float], float]] = {
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "log1p": np.log1p,
        }
        if self.operation in functions:
            return float(functions[self.operation](args[0]))
        raise DSLValidationError(f"cannot evaluate expression operation {self.operation!r}")


class _Symbolic:
    """Mixin providing arithmetic by converting a declaration to an Expr."""

    __array_priority__ = 1000

    def as_expr(self) -> Expr:
        raise NotImplementedError

    def __add__(self, other: ExpressionLike) -> Expr:
        return self.as_expr() + other

    def __radd__(self, other: ExpressionLike) -> Expr:
        return as_expr(other) + self.as_expr()

    def __sub__(self, other: ExpressionLike) -> Expr:
        return self.as_expr() - other

    def __rsub__(self, other: ExpressionLike) -> Expr:
        return as_expr(other) - self.as_expr()

    def __mul__(self, other: ExpressionLike) -> Expr:
        return self.as_expr() * other

    def __rmul__(self, other: ExpressionLike) -> Expr:
        return as_expr(other) * self.as_expr()

    def __truediv__(self, other: ExpressionLike) -> Expr:
        return self.as_expr() / other

    def __rtruediv__(self, other: ExpressionLike) -> Expr:
        return as_expr(other) / self.as_expr()

    def __pow__(self, other: ExpressionLike) -> Expr:
        return self.as_expr() ** other

    def __rpow__(self, other: ExpressionLike) -> Expr:
        return as_expr(other) ** self.as_expr()

    def __neg__(self) -> Expr:
        return -self.as_expr()


@dataclass(frozen=True, slots=True)
class Param(_Symbolic):
    """A population parameter with an explicit natural-scale constraint."""

    name: str
    init: float
    constraint: Literal["real", "positive", "bounded"] = "real"
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "parameter")
        if not isfinite(float(self.init)):
            raise DSLValidationError(f"initial value for {self.name!r} must be finite")
        if self.constraint == "positive" and self.init <= 0:
            raise DSLValidationError(f"positive parameter {self.name!r} requires init > 0")
        if self.constraint == "bounded":
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise DSLValidationError(f"bounded parameter {self.name!r} requires lower < upper")
            if not self.lower < self.init < self.upper:
                raise DSLValidationError(
                    f"initial value for {self.name!r} must lie strictly inside its bounds"
                )
        _register("parameters", self)

    @classmethod
    def real(
        cls,
        name: str,
        *,
        init: float = 0.0,
        unit: str | None = None,
        description: str | None = None,
    ) -> Param:
        return cls(name, float(init), "real", unit=unit, description=description)

    @classmethod
    def positive(
        cls,
        name: str,
        *,
        init: float,
        unit: str | None = None,
        description: str | None = None,
    ) -> Param:
        return cls(name, float(init), "positive", unit=unit, description=description)

    @classmethod
    def bounded(
        cls,
        name: str,
        *,
        init: float,
        lower: float,
        upper: float,
        unit: str | None = None,
        description: str | None = None,
    ) -> Param:
        return cls(
            name,
            float(init),
            "bounded",
            float(lower),
            float(upper),
            unit,
            description,
        )

    def as_expr(self) -> Expr:
        return Expr(
            "parameter",
            value=self.name,
            metadata={"constraint": self.constraint, "unit": self.unit},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "init": self.init,
            "constraint": self.constraint,
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class Eta(_Symbolic):
    """A named random effect and its covariance-block declaration."""

    name: str
    block: str
    covariance: Literal["correlated", "diagonal"] = "diagonal"
    level: str = "subject"

    def __post_init__(self) -> None:
        _validate_name(self.name, "random effect")
        _validate_name(self.block, "random-effect block")
        _register("etas", self)

    @classmethod
    def correlated(
        cls,
        *names: str,
        block: str | None = None,
        level: str = "subject",
    ) -> tuple[Eta, ...]:
        if not names:
            raise DSLValidationError("Eta.correlated requires at least one name")
        block_name = block or "omega_" + "_".join(names)
        return tuple(cls(name, block_name, "correlated", level) for name in names)

    @classmethod
    def independent(
        cls,
        *names: str,
        block: str | None = None,
        level: str = "subject",
    ) -> tuple[Eta, ...]:
        if not names:
            raise DSLValidationError("Eta.independent requires at least one name")
        block_name = block or "omega_" + "_".join(names)
        return tuple(cls(name, block_name, "diagonal", level) for name in names)

    def as_expr(self) -> Expr:
        return Expr(
            "eta",
            value=self.name,
            metadata={
                "block": self.block,
                "covariance": self.covariance,
                "level": self.level,
            },
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "block": self.block,
            "covariance": self.covariance,
            "level": self.level,
        }


@dataclass(frozen=True, slots=True)
class State(_Symbolic):
    """An ODE state/compartment declaration."""

    name: str
    unit: str | None = None
    initial: float = 0.0

    def __post_init__(self) -> None:
        _validate_name(self.name, "state")
        if not isfinite(float(self.initial)):
            raise DSLValidationError(f"initial state {self.name!r} must be finite")
        _register("states", self)

    def as_expr(self) -> Expr:
        return Expr("state", value=self.name, metadata={"unit": self.unit})

    def derivative(self, expression: ExpressionLike) -> DifferentialEquation:
        """Declare ``d(state)/dt`` inside the active model."""

        return derivative(self, expression)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "initial": self.initial}


@dataclass(frozen=True, slots=True)
class Symbol(_Symbolic):
    """A named covariate or external model input."""

    name: str
    role: str = "covariate"
    unit: str | None = None
    reference: float | str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, self.role)
        _register("symbols", self)

    def as_expr(self) -> Expr:
        return Expr(
            "symbol",
            value=self.name,
            metadata={"role": self.role, "unit": self.unit, "reference": self.reference},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "unit": self.unit,
            "reference": self.reference,
        }


ExpressionLike = Expr | _Symbolic | int | float | np.number


def as_expr(value: ExpressionLike) -> Expr:
    """Convert declarations and numeric constants to an expression node."""

    if isinstance(value, Expr):
        return value
    if isinstance(value, _Symbolic):
        return value.as_expr()
    if isinstance(value, (int, float, np.number)) and isfinite(float(value)):
        return Expr("constant", value=float(value))
    raise TypeError(f"{value!r} cannot be used in a pharmacometric expression")


def symbol(
    name: str,
    *,
    role: str = "covariate",
    unit: str | None = None,
    reference: float | str | None = None,
) -> Symbol:
    """Declare a typed covariate or other external model input."""

    return Symbol(name, role, unit, reference)


def covariate(
    name: str,
    *,
    unit: str | None = None,
    reference: float | str | None = None,
) -> Symbol:
    """Declare a covariate symbol."""

    return symbol(name, role="covariate", unit=unit, reference=reference)


def exp(value: ExpressionLike) -> Expr:
    """Build a symbolic exponential expression without evaluating ``value``."""

    return Expr("exp", (as_expr(value),))


def log(value: ExpressionLike) -> Expr:
    """Build a symbolic natural-log expression without checking its domain."""

    return Expr("log", (as_expr(value),))


def sqrt(value: ExpressionLike) -> Expr:
    """Build a symbolic square-root expression without checking its domain."""

    return Expr("sqrt", (as_expr(value),))


def log1p(value: ExpressionLike) -> Expr:
    """Build a symbolic ``log(1 + value)`` expression without evaluating it."""

    return Expr("log1p", (as_expr(value),))


@dataclass(frozen=True, slots=True)
class Dose:
    """Mapping from canonical event fields to a model state."""

    state: State
    amount: str = "AMT"
    rate: str | None = "RATE"
    duration: str | None = "DUR"
    compartment: str = "CMT"
    lag: str | float | None = None
    bioavailability: str | float | None = None
    route: str = "iv"

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise TypeError("Dose.state must be a State")
        _register("doses", self)

    @classmethod
    def into(
        cls,
        state: State,
        *,
        amount: str = "AMT",
        rate: str | None = "RATE",
        duration: str | None = "DUR",
        compartment: str = "CMT",
        lag: str | float | None = None,
        bioavailability: str | float | None = None,
        route: str = "iv",
    ) -> Dose:
        """Declare how dose records enter a state."""

        return cls(
            state,
            amount,
            rate,
            duration,
            compartment,
            lag,
            bioavailability,
            route,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "amount": self.amount,
            "rate": self.rate,
            "duration": self.duration,
            "compartment": self.compartment,
            "lag": self.lag,
            "bioavailability": self.bioavailability,
            "route": self.route,
        }


@dataclass(frozen=True, slots=True)
class DifferentialEquation:
    """One first-order ODE declaration."""

    state: State
    expression: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.name, "expression": self.expression.to_dict()}

    def format(self) -> str:
        return f"d({self.state.name})/dt = {self.expression.format()}"


def derivative(state: State, expression: ExpressionLike) -> DifferentialEquation:
    """Declare a state derivative in the active model."""

    if not isinstance(state, State):
        raise TypeError("derivative target must be a State")
    equation = DifferentialEquation(state, as_expr(expression))
    _register("equations", equation)
    return equation


def d(state: State, expression: ExpressionLike) -> DifferentialEquation:
    """Short alias for :func:`derivative`.

    Python does not permit the illustrative syntax ``d(state) = expression``;
    PyMixEF therefore uses ``d(state, expression)`` or
    ``state.derivative(expression)``.
    """

    return derivative(state, expression)


@dataclass(frozen=True, slots=True)
class Observation:
    """An endpoint, prediction expression, and residual-error declaration."""

    endpoint: str
    mean: Expr
    error: Any
    censored_below: str | float | None = None
    censored_above: str | float | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _validate_name(self.endpoint, "observation endpoint")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "mean": self.mean.to_dict(),
            "error": _serialize(self.error),
            "censored_below": self.censored_below,
            "censored_above": self.censored_above,
            "metadata": _serialize(self.metadata),
        }


def observe(
    endpoint: str,
    *,
    mean: ExpressionLike,
    error: Any,
    censored_below: str | float | None = None,
    censored_above: str | float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Observation:
    """Declare an observation endpoint in the active model."""

    observation = Observation(
        endpoint,
        as_expr(mean),
        error,
        censored_below,
        censored_above,
        _freeze_mapping(metadata),
    )
    _register("observations", observation)
    return observation


@dataclass(slots=True)
class _BuildContext:
    parameters: list[Param] = field(default_factory=list)
    etas: list[Eta] = field(default_factory=list)
    states: list[State] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    doses: list[Dose] = field(default_factory=list)
    equations: list[DifferentialEquation] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)


_CURRENT_CONTEXT: ContextVar[_BuildContext | None] = ContextVar(
    "pymixef_pharmacometrics_model_context", default=None
)


def _register(collection: str, value: Any) -> None:
    context = _CURRENT_CONTEXT.get()
    if context is not None:
        getattr(context, collection).append(value)


def _validate_name(name: str, kind: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise DSLValidationError(f"{kind} name must be a non-empty string")
    if not name.isidentifier():
        raise DSLValidationError(f"{kind} name {name!r} must be a valid identifier")


@dataclass(frozen=True, slots=True)
class ValidationMessage:
    """One coded severity and message emitted by dry-run model validation."""

    code: str
    severity: Literal["error", "warning", "info"]
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass(frozen=True, slots=True)
class ModelValidation:
    """Dry-run validation report for a compiled model."""

    valid: bool
    messages: tuple[ValidationMessage, ...]
    dimensions: Mapping[str, int]
    estimator_compatibility: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "dimensions", _freeze_mapping(self.dimensions))
        object.__setattr__(
            self, "estimator_compatibility", _freeze_mapping(self.estimator_compatibility)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "messages": [message.to_dict() for message in self.messages],
            "dimensions": dict(self.dimensions),
            "estimator_compatibility": dict(self.estimator_compatibility),
        }

    def raise_for_errors(self) -> None:
        errors = [message.message for message in self.messages if message.severity == "error"]
        if errors:
            raise DSLValidationError("; ".join(errors))


_IR_EXPRESSION_ARITY: Mapping[str, int] = MappingProxyType(
    {
        "constant": 0,
        "parameter": 0,
        "eta": 0,
        "state": 0,
        "symbol": 0,
        "negative": 1,
        "add": 2,
        "subtract": 2,
        "multiply": 2,
        "divide": 2,
        "power": 2,
        "exp": 1,
        "log": 1,
        "sqrt": 1,
        "log1p": 1,
    }
)


def _ir_expression_dependencies(
    expression: Expr,
    declarations: Mapping[str, frozenset[str]],
    *,
    context: str,
) -> tuple[str, ...]:
    """Validate one expression tree and return stable semantic dependencies."""

    expected_arity = _IR_EXPRESSION_ARITY.get(expression.operation)
    if expected_arity is None:
        raise DSLValidationError(
            f"[DSL-IR-EXPRESSION-UNSUPPORTED-001] {context} uses operation "
            f"{expression.operation!r}, which has no ModelIR representation"
        )
    if len(expression.arguments) != expected_arity:
        raise DSLValidationError(
            f"[DSL-IR-EXPRESSION-ARITY-001] {context} operation "
            f"{expression.operation!r} requires {expected_arity} arguments; "
            f"got {len(expression.arguments)}"
        )
    if expression.operation == "constant":
        try:
            finite_constant = expression.value is not None and isfinite(float(expression.value))
        except (TypeError, ValueError):
            finite_constant = False
        if not finite_constant:
            raise DSLValidationError(
                f"[DSL-IR-CONSTANT-001] {context} contains a non-finite constant"
            )
        return ()
    if expression.operation in {"parameter", "eta", "state", "symbol"}:
        name = str(expression.value)
        if expression.value is None or name not in declarations[expression.operation]:
            raise DSLValidationError(
                f"[DSL-IR-REFERENCE-001] {context} references undeclared "
                f"{expression.operation} {name!r}"
            )
        return (name,)
    dependencies = {
        name
        for argument in expression.arguments
        for name in _ir_expression_dependencies(argument, declarations, context=context)
    }
    return tuple(sorted(dependencies))


def _ir_error_declaration(
    observation: Observation,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return a supported residual-error declaration and symbolic parameters."""

    serializer = getattr(observation.error, "to_dict", None)
    if not callable(serializer):
        raise DSLValidationError(
            f"[DSL-IR-ERROR-UNSUPPORTED-001] observation "
            f"{observation.endpoint!r} has an error model without to_dict()"
        )
    declaration = serializer()
    if not isinstance(declaration, Mapping):
        raise DSLValidationError(
            f"[DSL-IR-ERROR-UNSUPPORTED-001] observation "
            f"{observation.endpoint!r} error to_dict() must return a mapping"
        )
    serialized = {str(key): _serialize(value) for key, value in declaration.items()}
    error_type = serialized.get("type")
    parameter_fields = {
        "additive": ("sigma",),
        "proportional": ("sigma",),
        "power": ("sigma", "power"),
        "combined": ("additive_sigma", "proportional_sigma", "power"),
        "lognormal": ("sigma",),
    }
    if not isinstance(error_type, str) or error_type not in parameter_fields:
        raise DSLValidationError(
            f"[DSL-IR-ERROR-UNSUPPORTED-001] observation "
            f"{observation.endpoint!r} uses residual-error type {error_type!r}; "
            f"supported types are {', '.join(sorted(parameter_fields))}"
        )
    symbolic = tuple(
        str(serialized[field_name])
        for field_name in parameter_fields[error_type]
        if isinstance(serialized.get(field_name), str)
    )
    return serialized, tuple(dict.fromkeys(symbolic))


def _parameter_ir(parameter: Param) -> tuple[ParameterIR, TransformIR]:
    if parameter.constraint == "real":
        support = "real"
        transform = "identity"
        bounds = None
        options: dict[str, Any] = {}
    elif parameter.constraint == "positive":
        support = "positive"
        transform = "log"
        bounds = (0.0, None)
        options = {}
    else:
        support = "interval"
        transform = "bounded"
        bounds = (parameter.lower, parameter.upper)
        options = {"lower": parameter.lower, "upper": parameter.upper}
    annotations = {
        "constraint": parameter.constraint,
        "description": parameter.description,
    }
    return (
        ParameterIR(
            name=parameter.name,
            initial=float(parameter.init),
            bounds=bounds,
            role="population-parameter",
            support=support,
            transform=transform,
            unit=parameter.unit,
            annotations=annotations,
        ),
        TransformIR(
            name=f"{parameter.name}:optimizer-to-natural",
            kind=transform,
            options=options,
            dependencies=(parameter.name,),
        ),
    )


@dataclass(frozen=True, slots=True)
class CompiledModel:
    """Data-only pharmacometric model declaration."""

    name: str
    parameters: tuple[Param, ...] = ()
    etas: tuple[Eta, ...] = ()
    states: tuple[State, ...] = ()
    symbols: tuple[Symbol, ...] = ()
    doses: tuple[Dose, ...] = ()
    equations: tuple[DifferentialEquation, ...] = ()
    observations: tuple[Observation, ...] = ()
    schema_version: str = "1.0"
    authoring_mode: str = "executed-python-declarations"

    def to_ir(self) -> ModelIR:
        """Compile this declaration into the common, versioned :class:`ModelIR`.

        Only expression and residual-error operations with defined semantic
        mappings are accepted.  Custom operations fail here rather than being
        reduced to an opaque string that a backend could misinterpret.
        """

        self.validate(raise_on_error=True)
        endpoint_names = [observation.endpoint for observation in self.observations]
        duplicate_endpoints = sorted(
            name for name in set(endpoint_names) if endpoint_names.count(name) > 1
        )
        if duplicate_endpoints:
            raise DSLValidationError(
                "[DSL-IR-ENDPOINT-DUPLICATE-001] observation endpoints must be "
                f"unique; duplicates: {', '.join(duplicate_endpoints)}"
            )

        block_semantics: dict[str, tuple[str, str]] = {}
        for eta in self.etas:
            semantics = (eta.level, eta.covariance)
            previous = block_semantics.setdefault(eta.block, semantics)
            if previous != semantics:
                raise DSLValidationError(
                    f"[DSL-IR-RANDOM-BLOCK-001] random-effect block {eta.block!r} "
                    "mixes covariance or hierarchy declarations"
                )

        declarations = {
            "parameter": frozenset(parameter.name for parameter in self.parameters),
            "eta": frozenset(eta.name for eta in self.etas),
            "state": frozenset(state.name for state in self.states),
            "symbol": frozenset(symbol.name for symbol in self.symbols),
        }
        equation_dependencies: dict[str, tuple[str, ...]] = {}
        for equation in self.equations:
            equation_dependencies[equation.state.name] = _ir_expression_dependencies(
                equation.expression,
                declarations,
                context=f"equation for state {equation.state.name!r}",
            )
        observation_dependencies: dict[str, tuple[str, ...]] = {}
        error_declarations: dict[str, dict[str, Any]] = {}
        error_parameter_names: dict[str, tuple[str, ...]] = {}
        inferred_error_parameters: set[str] = set()
        for observation in self.observations:
            observation_dependencies[observation.endpoint] = _ir_expression_dependencies(
                observation.mean,
                declarations,
                context=f"observation {observation.endpoint!r}",
            )
            error_declaration, symbolic_parameters = _ir_error_declaration(observation)
            error_declarations[observation.endpoint] = error_declaration
            error_parameter_names[observation.endpoint] = symbolic_parameters
            inferred_error_parameters.update(
                name for name in symbolic_parameters if name not in declarations["parameter"]
            )
        namespace_without_parameters = (
            declarations["eta"] | declarations["state"] | declarations["symbol"]
        )
        collisions = sorted(inferred_error_parameters & namespace_without_parameters)
        if collisions:
            raise DSLValidationError(
                "[DSL-IR-ERROR-PARAMETER-COLLISION-001] symbolic residual-error "
                f"parameters collide with other declarations: {', '.join(collisions)}"
            )

        parameter_pairs = tuple(_parameter_ir(parameter) for parameter in self.parameters)
        parameters = [pair[0] for pair in parameter_pairs]
        transforms = [pair[1] for pair in parameter_pairs]
        for name in sorted(inferred_error_parameters):
            parameters.append(
                ParameterIR(
                    name=name,
                    initial=None,
                    bounds=(0.0, None),
                    role="observation-error-parameter",
                    support="positive",
                    transform="log",
                    annotations={"inferred_from_observation_error": True},
                )
            )
            transforms.append(
                TransformIR(
                    name=f"{name}:optimizer-to-natural",
                    kind="log",
                    dependencies=(name,),
                    annotations={"inferred_from_observation_error": True},
                )
            )

        eta_blocks: dict[str, list[Eta]] = {}
        for eta in self.etas:
            eta_blocks.setdefault(eta.block, []).append(eta)
        random_effects = tuple(
            RandomEffectIR(
                terms=tuple(eta.name for eta in etas),
                group=etas[0].level,
                correlated=etas[0].covariance == "correlated",
                covariance=("unstructured" if etas[0].covariance == "correlated" else "diagonal"),
                dependencies=tuple(eta.name for eta in etas),
                annotations={"block": block},
            )
            for block, etas in eta_blocks.items()
        )
        covariance_structures = tuple(
            CovarianceIR(
                structure=("unstructured" if etas[0].covariance == "correlated" else "diagonal"),
                target=f"random-effect:{block}",
                dimension=len(etas),
                group=etas[0].level,
                options={"terms": [eta.name for eta in etas]},
            )
            for block, etas in eta_blocks.items()
        )
        predictors = tuple(
            PredictorIR(
                name=symbol.name,
                expression=symbol.name,
                kind=symbol.role,
                unit=symbol.unit,
                annotations={"reference": symbol.reference},
            )
            for symbol in self.symbols
        )
        state_equations = tuple(
            StateEquationIR(
                state=equation.state.name,
                rhs=equation.expression.format(),
                initial=float(equation.state.initial),
                unit=equation.state.unit,
                dependencies=equation_dependencies[equation.state.name],
                annotations={"expression_tree": equation.expression.to_dict()},
            )
            for equation in self.equations
        )
        events = tuple(
            EventIR(
                event_type="dose-mapping",
                target=dose.state.name,
                fields={key: value for key, value in dose.to_dict().items() if key != "state"},
                dependencies=(dose.state.name,),
            )
            for dose in self.doses
        )
        likelihoods: list[LikelihoodIR] = []
        outputs: list[OutputIR] = []
        for observation in self.observations:
            error_declaration = error_declarations[observation.endpoint]
            error_type = str(error_declaration["type"])
            family = "lognormal" if error_type == "lognormal" else "gaussian"
            formulas: dict[str, Any] = {
                "mean": observation.mean.format(),
                "error": error_declaration,
            }
            if observation.censored_below is not None:
                formulas["censored_below"] = observation.censored_below
            if observation.censored_above is not None:
                formulas["censored_above"] = observation.censored_above
            dependencies = tuple(
                sorted(
                    set(observation_dependencies[observation.endpoint])
                    | set(error_parameter_names[observation.endpoint])
                )
            )
            likelihoods.append(
                LikelihoodIR(
                    response=observation.endpoint,
                    family=family,
                    link="log" if family == "lognormal" else "identity",
                    formulas=formulas,
                    dependencies=dependencies,
                    annotations=dict(observation.metadata),
                )
            )
            outputs.append(
                OutputIR(
                    name=f"{observation.endpoint}_prediction",
                    expression=observation.mean.format(),
                    output_kind="observation-prediction",
                    dependencies=observation_dependencies[observation.endpoint],
                )
            )

        families = {likelihood.family for likelihood in likelihoods}
        top_level_family = (
            next(iter(families))
            if len(families) == 1
            else ("none" if not families else "multi-endpoint")
        )
        response = endpoint_names[0] if len(endpoint_names) == 1 else None
        event_columns = sorted(
            {
                value
                for dose in self.doses
                for value in (
                    dose.amount,
                    dose.rate,
                    dose.duration,
                    dose.compartment,
                )
                if isinstance(value, str)
            }
        )
        return ModelIR(
            name=self.name,
            source="pharmacometrics-dsl",
            response=response,
            family=top_level_family,
            random_effects=random_effects,
            predictors=predictors,
            likelihoods=tuple(likelihoods),
            covariance_structures=covariance_structures,
            state_equations=state_equations,
            events=events,
            parameters=tuple(parameters),
            transforms=tuple(transforms),
            outputs=tuple(outputs),
            data_schema={
                "covariates": {
                    symbol.name: {
                        "role": symbol.role,
                        "unit": symbol.unit,
                        "reference": symbol.reference,
                    }
                    for symbol in self.symbols
                },
                "event_columns": event_columns,
            },
            metadata={
                "authoring_surface": "pharmacometrics-dsl",
                "authoring_mode": self.authoring_mode,
                "source_schema_version": self.schema_version,
                "estimator_compatibility": dict(self.validate().estimator_compatibility),
            },
        )

    def validate(self, *, raise_on_error: bool = False) -> ModelValidation:
        """Validate names, equation coverage, mappings, and engine readiness."""

        messages: list[ValidationMessage] = []

        def duplicates(values: Sequence[Any]) -> set[str]:
            names = [value.name for value in values]
            return {name for name in names if names.count(name) > 1}

        for kind, values in (
            ("parameter", self.parameters),
            ("random effect", self.etas),
            ("state", self.states),
            ("symbol", self.symbols),
        ):
            for duplicate in sorted(duplicates(values)):
                messages.append(
                    ValidationMessage(
                        "DSL-DUPLICATE-NAME-001",
                        "error",
                        f"duplicate {kind} name {duplicate!r}",
                    )
                )
        namespace_names = [
            declaration.name
            for declarations in (self.parameters, self.etas, self.states, self.symbols)
            for declaration in declarations
        ]
        for duplicate in sorted(
            name for name in set(namespace_names) if namespace_names.count(name) > 1
        ):
            messages.append(
                ValidationMessage(
                    "DSL-NAMESPACE-COLLISION-001",
                    "error",
                    f"name {duplicate!r} is used by multiple declaration kinds",
                )
            )

        state_names = {state.name for state in self.states}
        equation_states = [equation.state.name for equation in self.equations]
        for state_name in sorted(state_names):
            count = equation_states.count(state_name)
            if count == 0:
                messages.append(
                    ValidationMessage(
                        "DSL-STATE-DERIVATIVE-MISSING-001",
                        "error",
                        f"state {state_name!r} has no derivative",
                    )
                )
            elif count > 1:
                messages.append(
                    ValidationMessage(
                        "DSL-STATE-DERIVATIVE-DUPLICATE-001",
                        "error",
                        f"state {state_name!r} has {count} derivatives",
                    )
                )
        for dose in self.doses:
            if dose.state.name not in state_names:
                messages.append(
                    ValidationMessage(
                        "DSL-DOSE-STATE-UNKNOWN-001",
                        "error",
                        f"dose targets undeclared state {dose.state.name!r}",
                    )
                )
        if not self.observations:
            messages.append(
                ValidationMessage(
                    "DSL-NO-OBSERVATION-001",
                    "warning",
                    "model has no observation endpoint",
                )
            )
        error = any(message.severity == "error" for message in messages)
        dimensions = {
            "parameters": len(self.parameters),
            "random_effects": len(self.etas),
            "states": len(self.states),
            "covariates": len(self.symbols),
            "observations": len(self.observations),
        }
        report = ModelValidation(
            valid=not error,
            messages=tuple(messages),
            dimensions=dimensions,
            estimator_compatibility={
                "simulation": not error,
                "conditional_mode": not error and bool(self.observations),
                # The initial package exposes FOCEI-ready components, not a
                # production population FOCEI engine.
                "focei_fit": False,
                "saem": False,
            },
        )
        if raise_on_error:
            report.raise_for_errors()
        return report

    def explain(self) -> str:
        """Print the implied equations, transforms, event mappings, and units."""

        lines = [
            f"PyMixEF pharmacometric model: {self.name}",
            f"schema: {self.schema_version}",
            f"authoring: {self.authoring_mode}",
            "",
            "Parameters:",
        ]
        if self.parameters:
            for parameter in self.parameters:
                bounds = ""
                if parameter.constraint == "bounded":
                    bounds = f" ({parameter.lower}, {parameter.upper})"
                unit = "" if parameter.unit is None else f" [{parameter.unit}]"
                lines.append(
                    f"  {parameter.name}: {parameter.constraint}{bounds}, "
                    f"init={parameter.init:g}{unit}"
                )
        else:
            lines.append("  (none)")
        lines.append("Random effects:")
        if self.etas:
            for eta in self.etas:
                lines.append(
                    f"  {eta.name}: block={eta.block}, covariance={eta.covariance}, "
                    f"level={eta.level}"
                )
        else:
            lines.append("  (none)")
        lines.append("States and equations:")
        if self.states:
            equation_by_state = {
                equation.state.name: equation.format() for equation in self.equations
            }
            for state in self.states:
                unit = "" if state.unit is None else f" [{state.unit}]"
                equation = equation_by_state.get(state.name, "(derivative missing)")
                lines.append(f"  {state.name}{unit}, initial={state.initial:g}: {equation}")
        else:
            lines.append("  (none)")
        lines.append("Event mappings:")
        if self.doses:
            for dose in self.doses:
                lines.append(
                    f"  {dose.amount}/{dose.rate}/{dose.duration} -> {dose.state.name} "
                    f"(CMT={dose.compartment}, route={dose.route})"
                )
        else:
            lines.append("  (none)")
        lines.append("Observations:")
        if self.observations:
            for observation in self.observations:
                lines.append(
                    f"  {observation.endpoint}: mean={observation.mean.format()}, "
                    f"error={type(observation.error).__name__}"
                )
        else:
            lines.append("  (none)")
        report = self.validate()
        lines.append(f"Validation: {'valid' if report.valid else 'invalid'}")
        for message in report.messages:
            lines.append(f"  [{message.severity}] {message.code}: {message.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete, optimizer-independent declaration."""

        return {
            "schema_version": self.schema_version,
            "kind": "pharmacometrics",
            "name": self.name,
            "authoring_mode": self.authoring_mode,
            "parameters": [_serialize(value) for value in self.parameters],
            "random_effects": [_serialize(value) for value in self.etas],
            "states": [_serialize(value) for value in self.states],
            "symbols": [_serialize(value) for value in self.symbols],
            "doses": [_serialize(value) for value in self.doses],
            "equations": [_serialize(value) for value in self.equations],
            "observations": [_serialize(value) for value in self.observations],
            "validation": self.validate().to_dict(),
        }


P = ParamSpec("P")
R = TypeVar("R")


class ModelDefinition:
    """Lazy wrapper produced by :func:`model`.

    Calling or compiling it executes the original Python declaration function
    once per call in an isolated context.  This behavior is explicit in
    ``CompiledModel.authoring_mode``.
    """

    def __init__(self, function: Callable[..., Any], *, name: str | None = None) -> None:
        if not callable(function):
            raise TypeError("model expects a callable")
        self.function = function
        self.name = name or function.__name__
        update_wrapper(self, function)

    def compile(self, *args: Any, **kwargs: Any) -> CompiledModel:
        context = _BuildContext()
        token = _CURRENT_CONTEXT.set(context)
        try:
            returned = self.function(*args, **kwargs)
        finally:
            _CURRENT_CONTEXT.reset(token)
        if isinstance(returned, CompiledModel):
            return returned
        if returned is not None:
            raise DSLValidationError(
                "a model declaration function must return None or a CompiledModel"
            )
        compiled = CompiledModel(
            name=self.name,
            parameters=tuple(context.parameters),
            etas=tuple(context.etas),
            states=tuple(context.states),
            symbols=tuple(context.symbols),
            doses=tuple(context.doses),
            equations=tuple(context.equations),
            observations=tuple(context.observations),
        )
        compiled.validate(raise_on_error=True)
        return compiled

    def __call__(self, *args: Any, **kwargs: Any) -> CompiledModel:
        return self.compile(*args, **kwargs)

    def validate(self, *args: Any, **kwargs: Any) -> ModelValidation:
        return self.compile(*args, **kwargs).validate()

    def explain(self, *args: Any, **kwargs: Any) -> str:
        return self.compile(*args, **kwargs).explain()

    def to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.compile(*args, **kwargs).to_dict()

    def to_ir(self, *args: Any, **kwargs: Any) -> ModelIR:
        """Compile the declaration into the common versioned model IR."""

        return self.compile(*args, **kwargs).to_ir()

    @property
    def declaration_signature(self) -> str:
        return str(signature(self.function))


@overload
def model(function: Callable[P, R], /) -> ModelDefinition: ...


@overload
def model(*, name: str | None = None) -> Callable[[Callable[P, R]], ModelDefinition]: ...


def model(
    function: Callable[..., Any] | None = None, /, *, name: str | None = None
) -> ModelDefinition | Callable[[Callable[..., Any]], ModelDefinition]:
    """Decorate a Python function as a lazy model declaration."""

    if function is None:
        return lambda supplied: ModelDefinition(supplied, name=name)
    return ModelDefinition(function, name=name)


def compiled_model(
    name: str,
    *,
    parameters: Sequence[Param] = (),
    etas: Sequence[Eta] = (),
    states: Sequence[State] = (),
    symbols: Sequence[Symbol] = (),
    doses: Sequence[Dose] = (),
    equations: Sequence[DifferentialEquation] = (),
    observations: Sequence[Observation] = (),
    validate: bool = True,
) -> CompiledModel:
    """Build a data-only model directly, without executing a declaration function."""

    result = CompiledModel(
        name=name,
        parameters=tuple(parameters),
        etas=tuple(etas),
        states=tuple(states),
        symbols=tuple(symbols),
        doses=tuple(doses),
        equations=tuple(equations),
        observations=tuple(observations),
        authoring_mode="direct-data-declarations",
    )
    if validate:
        result.validate(raise_on_error=True)
    return result


__all__ = [
    "CompiledModel",
    "DSLValidationError",
    "DifferentialEquation",
    "Dose",
    "Eta",
    "Expr",
    "ModelDefinition",
    "ModelValidation",
    "Observation",
    "Param",
    "State",
    "Symbol",
    "ValidationMessage",
    "as_expr",
    "compiled_model",
    "covariate",
    "d",
    "derivative",
    "exp",
    "log",
    "log1p",
    "model",
    "observe",
    "sqrt",
    "symbol",
]
