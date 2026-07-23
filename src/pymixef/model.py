"""Public model builders, dry-run compilation, and backend dispatch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._contracts import ReproducibilityClass, WarningRecord
from .convergence import ConvergenceReport
from .diagnostics import DiagnosticTable
from .errors import EngineCompatibilityError, ValidationError
from .families import Family, Gaussian
from .formula import (
    DesignMatrices,
    FormulaSpec,
    compile_formula,
    explain_formula,
    parse_formula,
)
from .ir import CovarianceIR, FixedEffectIR, LikelihoodIR, ModelIR, PriorIR
from .provenance import RunManifest, RunTimer
from .results import FitResult


@dataclass(frozen=True, slots=True)
class Response:
    """Explicit response declaration for the structured builder."""

    name: str
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("Response name cannot be empty.")


@dataclass(frozen=True, slots=True)
class Fixed:
    """Fixed-effects expression in the safe formula grammar."""

    expression: str

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise ValidationError("Fixed-effects expression cannot be empty.")


@dataclass(frozen=True, slots=True)
class Random:
    """One structured random-effects declaration."""

    expression: str
    group: str
    covariance: str = "unstructured"
    correlated: bool | None = None

    def __post_init__(self) -> None:
        if not self.expression.strip() or not self.group.strip():
            raise ValidationError("Random effects require an expression and group.")
        if self.correlated is None:
            object.__setattr__(self, "correlated", self.covariance.lower() != "diagonal")


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One deterministic model or engine compatibility finding."""

    code: str
    severity: str
    message: str
    component: str | None = None
    suggested_engines: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "component": self.component,
            "suggested_engines": list(self.suggested_engines),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Dry-run semantic and estimator compatibility result."""

    findings: tuple[ValidationFinding, ...]
    engine: str
    method: str
    compatible_engines: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.findings)

    def raise_for_errors(self) -> None:
        errors = tuple(item for item in self.findings if item.severity == "error")
        if not errors:
            return
        suggested = tuple(
            dict.fromkeys(engine for item in errors for engine in item.suggested_engines)
        )
        raise EngineCompatibilityError(
            "; ".join(f"[{item.code}] {item.message}" for item in errors),
            suggested_engines=suggested or self.compatible_engines,
            details={"findings": [item.to_dict() for item in errors]},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "engine": self.engine,
            "method": self.method,
            "compatible_engines": list(self.compatible_engines),
            "findings": [item.to_dict() for item in self.findings],
        }


def _family_name(family: Family) -> str:
    return str(getattr(family, "name", type(family).__name__)).lower()


def _family_link(family: Family) -> str:
    link = getattr(family, "link", None)
    return str(getattr(link, "name", link or "identity"))


def _column(data: Any, name: str) -> np.ndarray:
    if isinstance(data, Mapping):
        if name not in data:
            raise ValidationError(
                f"Required column {name!r} is absent.",
                code="DATA-COLUMN-MISSING-001",
                details={"column": name},
            )
        return np.asarray(data[name])
    try:
        return np.asarray(data[name])
    except (KeyError, TypeError, IndexError) as error:
        raise ValidationError(
            f"Required column {name!r} is absent.",
            code="DATA-COLUMN-MISSING-001",
            details={"column": name},
        ) from error


def _missing_mask(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        return np.isnan(array)
    if np.issubdtype(array.dtype, np.datetime64):
        return np.isnat(array)
    if array.dtype.kind in {"i", "u", "b"}:
        return np.zeros(array.shape, dtype=bool)
    output = np.zeros(array.shape, dtype=bool)
    for index, value in np.ndenumerate(array):
        output[index] = value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        )
    return output


def _canonical_engine_method(
    *,
    family_name: str,
    residual: Any,
    engine: str | None,
    method: str | None,
) -> tuple[str, str]:
    raw_engine = None if engine is None else engine.lower().replace("_", "-")
    raw_method = None if method is None else method.lower().replace("_", "-")
    if raw_engine in {"laplace", "aghq", "adaptive-quadrature"}:
        raw_method = raw_method or raw_engine
        raw_engine = "glmm"
    aliases = {
        "gaussian-lmm": "lmm",
        "dense-lmm": "lmm",
        "reference-lmm": "lmm",
        "clinical": "mmrm",
        "generalized": "glmm",
    }
    raw_engine = aliases.get(raw_engine, raw_engine)
    if raw_engine is None:
        if family_name == "gaussian":
            raw_engine = "mmrm" if residual is not None else "lmm"
        else:
            raw_engine = "glmm"
    if raw_method is None:
        raw_method = "laplace" if raw_engine == "glmm" else "reml"
    return raw_engine, raw_method


_ENGINE_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "lmm": {
        "maxiter": 1_000,
        "tolerance": 1e-8,
        "compute_hessian": True,
    },
    "glmm": {
        "maxiter": 500,
        "tolerance": 1e-7,
        "inner_maxiter": 100,
        "inner_tolerance": 1e-9,
        "compute_hessian": True,
    },
    "mmrm": {
        "df_method": "satterthwaite",
        "confidence_level": 0.95,
        "maxiter": 1_000,
        "tolerance": 1e-8,
        "compute_hessian": True,
    },
}

_ENGINE_OPTIMIZERS: Mapping[str, tuple[str, ...]] = {
    "lmm": ("L-BFGS-B", "Powell rescue", "L-BFGS-B refinement"),
    "glmm": (
        "L-BFGS-B outer",
        "Powell outer rescue",
        "L-BFGS-B outer refinement",
        "BFGS conditional mode",
    ),
    "mmrm": ("L-BFGS-B", "Powell rescue", "L-BFGS-B refinement"),
}


def _backend_settings(engine: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    defaults = dict(_ENGINE_DEFAULTS[engine])
    unknown = sorted(set(settings) - set(defaults))
    if unknown:
        raise ValidationError(
            f"Unknown {engine} fit settings: {', '.join(unknown)}.",
            code="ENGINE-SETTING-UNKNOWN-001",
            remediation=f"Choose from {sorted(defaults)}.",
            details={
                "engine": engine,
                "unknown": unknown,
                "supported": sorted(defaults),
            },
        )
    defaults.update(settings)
    return defaults


def _prior_ir_nodes(priors: Mapping[str, Any]) -> tuple[PriorIR, ...]:
    """Normalize the public prior declarations into typed IR nodes.

    A prior may be supplied as a ``PriorIR``, a distribution name, or a
    mapping with a required ``distribution`` key.  Mapping parameters can be
    inline or nested under ``parameters``; mixing the two forms is rejected so
    the serialized meaning cannot depend on precedence rules.
    """

    result: list[PriorIR] = []
    for raw_target, declaration in priors.items():
        target = str(raw_target).strip()
        if not target:
            raise ValidationError(
                "Prior targets must be non-empty strings.",
                code="MODEL-PRIOR-TARGET-001",
            )
        if isinstance(declaration, PriorIR):
            if declaration.target != target:
                raise ValidationError(
                    f"Prior mapping target {target!r} does not match the "
                    f"PriorIR target {declaration.target!r}.",
                    code="MODEL-PRIOR-TARGET-002",
                    details={
                        "mapping_target": target,
                        "node_target": declaration.target,
                    },
                )
            result.append(declaration)
            continue
        if isinstance(declaration, str):
            distribution = declaration.strip()
            if not distribution:
                raise ValidationError(
                    f"Prior for {target!r} requires a non-empty distribution name.",
                    code="MODEL-PRIOR-DISTRIBUTION-001",
                )
            parameters: Mapping[str, Any] = {}
        elif isinstance(declaration, Mapping):
            raw_distribution = declaration.get("distribution")
            if not isinstance(raw_distribution, str) or not raw_distribution.strip():
                raise ValidationError(
                    f"Prior for {target!r} requires a non-empty 'distribution' string.",
                    code="MODEL-PRIOR-DISTRIBUTION-001",
                )
            distribution = raw_distribution.strip()
            inline = {
                str(key): value
                for key, value in declaration.items()
                if key not in {"distribution", "parameters"}
            }
            nested = declaration.get("parameters")
            if nested is not None and not isinstance(nested, Mapping):
                raise ValidationError(
                    f"Prior parameters for {target!r} must be a mapping.",
                    code="MODEL-PRIOR-PARAMETERS-001",
                )
            if nested is not None and inline:
                raise ValidationError(
                    f"Prior for {target!r} mixes nested and inline parameters.",
                    code="MODEL-PRIOR-PARAMETERS-002",
                    remediation="Use either {'distribution': ..., 'parameters': {...}} "
                    "or inline parameter keys.",
                )
            parameters = dict(nested) if isinstance(nested, Mapping) else inline
        else:
            raise ValidationError(
                f"Prior for {target!r} must be a PriorIR, distribution name, or mapping.",
                code="MODEL-PRIOR-DECLARATION-001",
            )
        result.append(
            PriorIR(
                target=target,
                distribution=distribution,
                parameters=parameters,
            )
        )
    return tuple(result)


@dataclass(slots=True)
class Model:
    """Backend-neutral scientific model.

    Construct with a formula through :meth:`from_formula`, or provide
    :class:`Response`, :class:`Fixed`, and :class:`Random` declarations directly.
    """

    response: Response | str | None = None
    fixed: Fixed | str | None = None
    random: Sequence[Random] = ()
    family: Family = field(default_factory=Gaussian)
    residual: Any = None
    formula: str | None = None
    zero_inflation: str | None = None
    dispersion: str | None = None
    shape: str | None = None
    priors: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.response, str):
            self.response = Response(self.response)
        if isinstance(self.fixed, str):
            self.fixed = Fixed(self.fixed)
        self.random = tuple(self.random)
        if isinstance(self.family, type) and issubclass(self.family, Family):
            self.family = self.family()
        if not isinstance(self.family, Family):
            raise ValidationError("family must be a PyMixEF Family instance.")
        if self.formula is None and self.response is None:
            raise ValidationError("Model requires a formula or response declaration.")
        # Parsing at construction prevents unsafe or ambiguous syntax from being
        # stored and discovered only during optimization.
        parse_formula(self.formula_text())

    @classmethod
    def from_formula(
        cls,
        formula: str,
        *,
        family: Family | None = None,
        residual: Any = None,
        zero_inflation: str | None = None,
        dispersion: str | None = None,
        shape: str | None = None,
        priors: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Model:
        spec = parse_formula(formula)
        random = tuple(
            Random(
                expression=(
                    ("1 + " if term.intercept and term.terms else "1") + " + ".join(term.terms)
                ).strip(" +"),
                group=term.group,
                correlated=term.correlated,
                covariance="unstructured" if term.correlated else "diagonal",
            )
            for term in spec.random_terms
        )
        fixed_parts = (("1",) if spec.intercept else ("0",)) + spec.fixed_terms
        return cls(
            response=Response(spec.response),
            fixed=Fixed(" + ".join(fixed_parts)),
            random=random,
            family=family or Gaussian(),
            residual=residual,
            formula=formula,
            zero_inflation=zero_inflation,
            dispersion=dispersion,
            shape=shape,
            priors=dict(priors or {}),
            metadata=dict(metadata or {}),
        )

    def formula_text(self) -> str:
        if self.formula is not None:
            return self.formula
        assert isinstance(self.response, Response)
        fixed = self.fixed.expression if isinstance(self.fixed, Fixed) else "1"
        random_parts = [
            f"({item.expression} {'|' if item.correlated else '||'} {item.group})"
            for item in self.random
        ]
        rhs = " + ".join((fixed, *random_parts))
        return f"{self.response.name} ~ {rhs}"

    @property
    def specification(self) -> FormulaSpec:
        return parse_formula(self.formula_text())

    def _ir(
        self,
        *,
        engine: str | None = None,
        method: str | None = None,
        matrices: DesignMatrices | None = None,
    ) -> ModelIR:
        family_name = _family_name(self.family)
        spec = self.specification
        base = spec.to_ir(family=family_name)
        formulas = {
            name: value
            for name, value in {
                "zero_inflation": self.zero_inflation,
                "dispersion": self.dispersion,
                "shape": self.shape,
            }.items()
            if value is not None
        }
        likelihoods = (
            LikelihoodIR(
                response=spec.response,
                family=family_name,
                link=_family_link(self.family),
                formulas=formulas,
            ),
        )
        covariance_structures = ()
        if self.residual is not None:
            covariance_structures = (
                CovarianceIR(
                    structure=str(
                        getattr(
                            self.residual,
                            "name",
                            type(self.residual).__name__.lower(),
                        )
                    ),
                    target="residual",
                    dimension=getattr(self.residual, "dimension", None),
                    index=getattr(self.residual, "index", None),
                    group=getattr(self.residual, "group", None),
                ),
            )
        estimator = {
            key: value
            for key, value in {"engine": engine, "method": method}.items()
            if value is not None
        }
        fixed_effects = base.fixed_effects
        data_schema: dict[str, Any] = {}
        if matrices is not None:
            fixed_effects = tuple(
                FixedEffectIR(name=name, expression=name, columns=(name,))
                for name in matrices.fixed_names
            )
            data_schema = {
                "row_count": len(matrices.response),
                "row_ids": list(matrices.row_ids),
                "factor_levels": {
                    name: list(values) for name, values in matrices.factor_levels.items()
                },
                "contrast_coding": dict(matrices.contrast_coding),
                "audit": matrices.audit.to_dict(),
            }
        metadata = {
            **dict(self.metadata),
            "authoring_surface": "formula" if self.formula else "structured-builder",
        }
        return replace(
            base,
            likelihoods=likelihoods,
            covariance_structures=covariance_structures,
            fixed_effects=fixed_effects,
            priors=_prior_ir_nodes(self.priors),
            data_schema=data_schema,
            estimator=estimator,
            metadata=metadata,
        )

    def to_ir(
        self,
        *,
        engine: str | None = None,
        method: str | None = None,
    ) -> ModelIR:
        """Compile data-independent semantics into the shared versioned IR."""

        return self._ir(engine=engine, method=method)

    def validate(
        self,
        *,
        engine: str | None = None,
        method: str | None = None,
    ) -> ValidationReport:
        family_name = _family_name(self.family)
        selected_engine, selected_method = _canonical_engine_method(
            family_name=family_name,
            residual=self.residual,
            engine=engine,
            method=method,
        )
        findings: list[ValidationFinding] = []
        compatible: list[str] = []
        if family_name == "gaussian":
            compatible.append("lmm")
            if self.residual is not None:
                compatible.append("mmrm")
        if family_name in {
            "bernoulli",
            "binomial",
            "poisson",
            "negative-binomial-2",
        }:
            compatible.append("glmm")
        if selected_engine not in {"lmm", "glmm", "mmrm"}:
            findings.append(
                ValidationFinding(
                    "ENGINE-UNSUPPORTED-001",
                    "error",
                    f"Engine {selected_engine!r} is not available for formula models.",
                    "engine",
                    tuple(compatible),
                )
            )
        if selected_engine == "lmm":
            if family_name != "gaussian":
                findings.append(
                    ValidationFinding(
                        "ENGINE-FAMILY-001",
                        "error",
                        "The LMM engine requires a Gaussian family.",
                        "family",
                        tuple(compatible),
                    )
                )
            if selected_method not in {"ml", "reml"}:
                findings.append(
                    ValidationFinding(
                        "ENGINE-METHOD-001",
                        "error",
                        "The LMM engine supports ML and REML.",
                        "method",
                        ("lmm",),
                    )
                )
        if selected_engine == "glmm":
            if family_name not in {
                "bernoulli",
                "binomial",
                "poisson",
                "negative-binomial-2",
            }:
                findings.append(
                    ValidationFinding(
                        "ENGINE-FAMILY-002",
                        "error",
                        f"The reference Laplace GLMM does not fit family {family_name!r}.",
                        "family",
                        tuple(compatible),
                    )
                )
            if selected_method != "laplace":
                findings.append(
                    ValidationFinding(
                        "ENGINE-METHOD-002",
                        "error",
                        "The reference GLMM engine implements Laplace only; AGHQ is "
                        "not silently substituted.",
                        "method",
                        ("glmm",),
                    )
                )
            if self.residual is not None:
                findings.append(
                    ValidationFinding(
                        "ENGINE-RESIDUAL-001",
                        "error",
                        "Structured residual covariance is not supported by the "
                        "reference GLMM engine.",
                        "residual",
                        (),
                    )
                )
        if selected_engine == "mmrm":
            if family_name != "gaussian":
                findings.append(
                    ValidationFinding(
                        "ENGINE-FAMILY-003",
                        "error",
                        "MMRM requires a Gaussian response.",
                        "family",
                        ("mmrm",),
                    )
                )
            if self.residual is None:
                findings.append(
                    ValidationFinding(
                        "ENGINE-RESIDUAL-002",
                        "error",
                        "MMRM requires an explicit within-subject covariance structure.",
                        "residual",
                        ("lmm",),
                    )
                )
            if self.specification.random_terms:
                findings.append(
                    ValidationFinding(
                        "ENGINE-RANDOM-001",
                        "error",
                        "The dedicated MMRM engine does not combine residual blocks "
                        "with formula random effects in the initial reference path.",
                        "random_effects",
                        ("lmm",),
                    )
                )
            if selected_method != "reml":
                findings.append(
                    ValidationFinding(
                        "ENGINE-METHOD-003",
                        "error",
                        "The dedicated MMRM path uses REML.",
                        "method",
                        ("mmrm",),
                    )
                )
        for component, formula in (
            ("zero_inflation", self.zero_inflation),
            ("dispersion", self.dispersion),
            ("shape", self.shape),
        ):
            if formula is not None:
                findings.append(
                    ValidationFinding(
                        "ENGINE-DISTRIBUTIONAL-PREDICTOR-001",
                        "error",
                        f"The {component.replace('_', ' ')} predictor is represented "
                        "in ModelIR but is not executable by the 0.1 reference "
                        "formula backends; it will not be silently ignored.",
                        component,
                        (),
                    )
                )
        return ValidationReport(
            findings=tuple(findings),
            engine=selected_engine,
            method=selected_method,
            compatible_engines=tuple(dict.fromkeys(compatible)),
        )

    def explain(
        self,
        data: Any | None = None,
        *,
        engine: str | None = None,
        method: str | None = None,
        missing: str = "drop",
    ) -> str:
        validation = self.validate(engine=engine, method=method)
        formula_text = explain_formula(self.formula_text(), data, missing=missing)
        findings = "\n".join(
            f"  [{item.code}] {item.severity}: {item.message}" for item in validation.findings
        )
        return (
            f"{formula_text}\n"
            f"Family: {_family_name(self.family)} ({_family_link(self.family)} link)\n"
            f"Engine: {validation.engine}; method: {validation.method}\n"
            f"Compatibility: {'valid' if validation.valid else 'invalid'}"
            + (f"\nFindings:\n{findings}" if findings else "")
        )

    def compile(
        self,
        data: Any,
        *,
        engine: str | None = None,
        method: str | None = None,
        missing: str = "drop",
        **settings: Any,
    ) -> ExecutionPlan:
        validation = self.validate(engine=engine, method=method)
        validation.raise_for_errors()
        matrices = compile_formula(self.specification, data, missing=missing)
        if self.residual is not None:
            retained_positions = np.asarray(
                [
                    record.input_position
                    for record in matrices.audit.records
                    if record.action == "retained"
                ],
                dtype=int,
            )
            for structural_name in (
                getattr(self.residual, "group", None),
                getattr(self.residual, "index", None),
            ):
                if not structural_name:
                    continue
                structural_values = _column(data, structural_name)
                if len(structural_values) != len(matrices.response):
                    structural_values = structural_values[retained_positions]
                missing_structural = _missing_mask(structural_values)
                if np.any(missing_structural):
                    affected = np.flatnonzero(missing_structural).tolist()
                    raise ValidationError(
                        f"Residual covariance key {structural_name!r} contains "
                        "missing values on analysis rows.",
                        code="DATA-STRUCTURAL-MISSING-001",
                        details={
                            "column": structural_name,
                            "analysis_positions": affected,
                            "policy": "filter or impute explicitly before compilation",
                        },
                    )
        ir = self._ir(
            engine=validation.engine,
            method=validation.method,
            matrices=matrices,
        )
        return ExecutionPlan(
            model=self,
            matrices=matrices,
            model_ir=ir,
            source_data=data,
            engine=validation.engine,
            method=validation.method,
            settings={
                "missing": missing,
                **_backend_settings(validation.engine, settings),
            },
            validation=validation,
        )

    def fit(
        self,
        data: Any,
        *,
        engine: str | None = None,
        method: str | None = None,
        missing: str = "drop",
        **settings: Any,
    ) -> FitResult:
        return self.compile(
            data,
            engine=engine,
            method=method,
            missing=missing,
            **settings,
        ).fit()


def _diagnostic_tables(value: Mapping[str, Any]) -> dict[str, DiagnosticTable]:
    tables: dict[str, DiagnosticTable] = {}
    for name, raw in value.items():
        if isinstance(raw, DiagnosticTable):
            tables[name] = raw
            continue
        if isinstance(raw, Mapping) and {"name", "columns"} <= raw.keys():
            tables[name] = DiagnosticTable.from_dict(raw)
            continue
        if isinstance(raw, Mapping):
            columns: dict[str, np.ndarray] = {}
            metadata: dict[str, Any] = {}
            lengths: set[int] = set()
            for column_name, column_value in raw.items():
                array = np.asarray(column_value)
                if array.ndim == 0:
                    metadata[column_name] = column_value
                else:
                    columns[column_name] = array
                    lengths.add(len(array))
            if len(lengths) <= 1:
                tables[name] = DiagnosticTable(name, columns, metadata)
                continue
        # Preserve an irregular backend diagnostic as one JSON-like metadata row.
        tables[name] = DiagnosticTable(
            name,
            {"record": np.asarray([0])},
            {"backend_value": raw},
        )
    return tables


def _warning_records(values: Sequence[Any]) -> tuple[WarningRecord, ...]:
    output: list[WarningRecord] = []
    for value in values:
        if isinstance(value, WarningRecord):
            output.append(value)
        elif isinstance(value, Mapping):
            output.append(
                WarningRecord(
                    code=str(value.get("code", "UNSPECIFIED-WARNING")),
                    severity=str(value.get("severity", "review")),
                    message=str(value.get("message", "")),
                    component=value.get("component"),
                    remediation=value.get("remediation"),
                    details=dict(value.get("details", {})),
                )
            )
    return tuple(output)


@dataclass(slots=True)
class ExecutionPlan:
    """Deterministic compiled model, data audit, and engine settings."""

    model: Model
    matrices: DesignMatrices
    model_ir: ModelIR
    source_data: Any
    engine: str
    method: str
    settings: Mapping[str, Any]
    validation: ValidationReport

    def validate(self) -> ValidationReport:
        return self.validation

    def explain(self) -> str:
        audit = self.matrices.audit
        return (
            f"{self.matrices.explain()}\n"
            f"Family: {_family_name(self.model.family)} "
            f"({_family_link(self.model.family)} link)\n"
            f"Engine: {self.engine}; method: {self.method}\n"
            f"Model IR: {self.model_ir.semantic_hash}\n"
            f"Data audit: {audit.input_rows} input, {audit.analysis_rows} analysis, "
            f"{audit.excluded_rows} excluded; reasons={dict(audit.reason_counts)}"
        )

    def to_backend_data(self) -> dict[str, Any]:
        value = self.matrices.to_backend_data()
        value.update(
            {
                "family": self.model.family,
                "family_name": _family_name(self.model.family),
                "method": self.method,
                "zero_inflation": self.model.zero_inflation,
                "dispersion": self.model.dispersion,
                "shape": self.model.shape,
            }
        )
        if self.model.residual is not None:
            value["residual_covariance"] = self.model.residual
            value["covariance"] = self.model.residual
            group_name = getattr(self.model.residual, "group", None)
            index_name = getattr(self.model.residual, "index", None)
            retained_positions = np.asarray(
                [
                    record.input_position
                    for record in self.matrices.audit.records
                    if record.action == "retained"
                ],
                dtype=int,
            )
            if group_name:
                subjects = _column(self.source_data, group_name)
                # Align to analysis rows after exclusions.
                if len(subjects) != len(self.matrices.response):
                    subjects = subjects[retained_positions]
                value.update({"subjects": subjects, "subject": subjects, "groups": subjects})
            if index_name:
                visits = _column(self.source_data, index_name)
                if len(visits) != len(self.matrices.response):
                    visits = visits[retained_positions]
                value.update(
                    {
                        "visits": visits,
                        "visit": visits,
                        "times": visits,
                        "visit_times": visits,
                    }
                )
        return value

    def fit(self) -> FitResult:
        backend_data = self.to_backend_data()
        backend_settings = dict(self.settings)
        backend_settings.pop("missing", None)
        with RunTimer() as timer:
            if self.engine == "lmm":
                from .backends.lmm import fit_lmm

                payload = fit_lmm(backend_data, method=self.method, **backend_settings)
            elif self.engine == "glmm":
                from .backends.glmm import fit_glmm

                payload = fit_glmm(
                    backend_data,
                    family=self.model.family,
                    method=self.method,
                    **backend_settings,
                )
            elif self.engine == "mmrm":
                from .backends.mmrm import fit_mmrm

                payload = fit_mmrm(
                    backend_data,
                    covariance=self.model.residual,
                    method=self.method,
                    **backend_settings,
                )
            else:  # Statically unreachable after validation, retained for plugins.
                raise EngineCompatibilityError(
                    f"No backend dispatch is registered for {self.engine!r}.",
                    suggested_engines=self.validation.compatible_engines,
                )
        convergence = ConvergenceReport.from_dict(payload["convergence"])
        warnings = _warning_records(payload["convergence"].get("warnings", ()))
        random_values = payload.get("random_effects", {})
        if isinstance(random_values, Mapping):
            random_effects = dict(random_values)
        else:
            array = np.asarray(random_values)
            raw_names = payload.get("diagnostic_data", {}).get("random_effects", {})
            names = raw_names.get("name", ()) if isinstance(raw_names, Mapping) else ()
            if len(names) == array.size:
                random_effects = {
                    str(name): float(number)
                    for name, number in zip(names, array.reshape(-1), strict=True)
                }
            else:
                random_effects = {
                    f"random_effect[{index}]": float(number)
                    for index, number in enumerate(array.reshape(-1))
                }
        extra = dict(payload.get("extra", {}))
        extra.update(
            {
                "family": _family_name(self.model.family),
                "row_ids": list(self.matrices.row_ids),
                "data_audit": self.matrices.audit.to_dict(),
                "factor_levels": {
                    name: list(values) for name, values in self.matrices.factor_levels.items()
                },
                "contrast_coding": dict(self.matrices.contrast_coding),
            }
        )
        if "residual_scale" not in extra:
            for candidate in ("residual_sd", "sigma", "sd(residual)"):
                if candidate in payload["parameters"]:
                    extra["residual_scale"] = payload["parameters"][candidate]
                    break
        reproducibility = (
            ReproducibilityClass.STOCHASTIC_MONTE_CARLO
            if self.method in {"saem", "mcem", "hmc", "nuts", "variational"}
            else ReproducibilityClass.DETERMINISTIC_TOLERANCE
        )
        manifest = RunManifest.capture(
            model_ir=self.model_ir.to_dict(),
            data=self.source_data,
            engine=self.engine,
            method=self.method,
            settings={
                **backend_settings,
                "optimizer_sequence": list(_ENGINE_OPTIMIZERS[self.engine]),
            },
            seeds={
                key: int(value)
                for key, value in backend_settings.items()
                if "seed" in key and isinstance(value, (int, np.integer))
            },
            reproducibility_class=reproducibility,
            elapsed_seconds=timer.elapsed_seconds,
            convergence=convergence.to_dict(),
            warnings=[item.to_dict() for item in warnings],
            source={
                "model_ir_schema_version": self.model_ir.schema_version,
                "authoring_surface": self.model_ir.metadata.get("authoring_surface"),
            },
        )
        result = FitResult(
            model_ir=self.model_ir,
            parameters=payload["parameters"],
            unconstrained_parameters=payload["unconstrained_parameters"],
            parameter_covariance=payload.get("parameter_covariance"),
            fitted_values=payload["fitted_values"],
            residuals=payload["residuals"],
            random_effects=random_effects,
            objective=float(payload["objective"]),
            log_likelihood=payload.get("log_likelihood"),
            method=str(payload.get("method", self.method)).lower(),
            engine=str(payload.get("engine", self.engine)),
            convergence=convergence,
            manifest=manifest,
            warnings=warnings,
            diagnostic_data=_diagnostic_tables(payload.get("diagnostic_data", {})),
            extra=extra,
        )
        result.manifest = manifest.with_outputs(
            {
                "parameters": result.parameters,
                "objective": result.objective,
                "fitted_values": result.fitted_values,
                "residuals": result.residuals,
            }
        )
        return result


def fit(
    formula: str | Model,
    *,
    data: Any,
    family: Family | None = None,
    residual: Any = None,
    method: str | None = None,
    engine: str | None = None,
    inference: str | None = None,
    zero_inflation: str | None = None,
    dispersion: str | None = None,
    shape: str | None = None,
    missing: str = "drop",
    **settings: Any,
) -> FitResult:
    """Fit a formula or prebuilt model through static compatibility dispatch."""

    if isinstance(formula, Model):
        model = formula
        if any(
            value is not None for value in (family, residual, zero_inflation, dispersion, shape)
        ):
            raise ValidationError(
                "Family/residual predictor overrides belong on the supplied Model.",
                code="MODEL-OVERRIDE-001",
            )
    else:
        model = Model.from_formula(
            formula,
            family=family,
            residual=residual,
            zero_inflation=zero_inflation,
            dispersion=dispersion,
            shape=shape,
        )
    if inference is not None:
        raise ValidationError(
            "The generic inference selector is not executable by the 0.1 "
            "formula backends and will not be silently ignored.",
            code="ENGINE-INFERENCE-UNSUPPORTED-001",
        )
    return model.fit(
        data,
        method=method,
        engine=engine,
        missing=missing,
        **settings,
    )


__all__ = [
    "ExecutionPlan",
    "Fixed",
    "Model",
    "Random",
    "Response",
    "ValidationFinding",
    "ValidationReport",
    "fit",
]
