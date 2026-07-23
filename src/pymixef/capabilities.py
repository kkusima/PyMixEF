"""Evidence-gated capability registry.

The registry is deliberately machine-readable.  Presence in the public API is
not treated as evidence that a method has reached reference-validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ._contracts import Maturity, ReproducibilityClass


@dataclass(frozen=True, slots=True)
class Capability:
    """One scientific or platform capability and its evidence state."""

    identifier: str
    name: str
    stage: str
    maturity: Maturity
    implemented: bool
    reproducibility: ReproducibilityClass | None
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "stage": self.stage,
            "maturity": self.maturity.value,
            "implemented": self.implemented,
            "reproducibility": (
                None if self.reproducibility is None else self.reproducibility.value
            ),
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


_D = ReproducibilityClass.DETERMINISTIC_TOLERANCE
_S = ReproducibilityClass.STOCHASTIC_MONTE_CARLO


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "ARCH-001",
        "Versioned immutable model IR",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_ir.py", "src/pymixef/schemas/model-ir-v1.json"),
    ),
    Capability(
        "ARCH-002",
        "Estimator compatibility validation",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_model.py",),
    ),
    Capability(
        "API-001",
        "Dry-run compile, validate, and explain",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_formula.py", "tests/test_model.py"),
    ),
    Capability(
        "API-002",
        "Stable result archival",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_results.py",),
    ),
    Capability(
        "API-003",
        "Data audit without silent mutation",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_data_covariance.py",),
    ),
    Capability(
        "COV-001",
        "Positive-definite covariance parameterizations",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_data_covariance.py",),
    ),
    Capability(
        "COV-002",
        "Covariance singularity reporting",
        "classical-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_lmm.py",),
    ),
    Capability(
        "DIST-001",
        "Normalized likelihood policy",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_families.py",),
    ),
    Capability(
        "DIST-002",
        "Family derivatives/CDF/simulation contract",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_families.py",),
        ("Analytic derivatives vary by family; finite-difference backend is used otherwise.",),
    ),
    Capability(
        "LMM-001",
        "Gaussian LMM ML/REML reference engine",
        "classical-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_lmm.py", "benchmarks/b01_sleepstudy.json"),
        (
            "The 0.1 reference engine uses dense marginal covariance and is not "
            "the blueprint's million-row sparse production core.",
        ),
    ),
    Capability(
        "LMM-002",
        "Sparse million-row LMM engine",
        "classical-core",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        ("Requires the planned compiled sparse backend.",),
    ),
    Capability(
        "GLMM-001",
        "Laplace GLMM",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_glmm_mmrm.py",),
        ("Initial implementation targets documented low-dimensional random blocks.",),
    ),
    Capability(
        "GLMM-002",
        "Adaptive Gauss-Hermite quadrature",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        ("Compatibility validation rejects AGHQ until an order-sensitivity suite lands.",),
    ),
    Capability(
        "MMRM-001",
        "MMRM REML with Satterthwaite and labeled approximate KR",
        "clinical-longitudinal",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_glmm_mmrm.py",),
        (
            "The dense reference path is intended for small problems. Exact "
            "Kenward-Roger is rejected; only a clearly labeled approximation is available.",
        ),
    ),
    Capability(
        "DATA-001",
        "Canonical pharmacometric event records",
        "nlme-foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_events.py",),
    ),
    Capability(
        "DATA-002",
        "Deterministic same-time event ordering",
        "nlme-foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_events.py",),
    ),
    Capability(
        "ODE-001",
        "Reference ODE integration",
        "nlme-foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_ode_pk.py",),
    ),
    Capability(
        "ODE-002",
        "Dose/event-aware ODE integration",
        "nlme-foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_events.py", "tests/test_ode_pk.py"),
        (
            "Covers the documented bolus, infusion, reset, and ADDL reference subset; "
            "steady-state event semantics are rejected.",
        ),
    ),
    Capability(
        "NLME-001",
        "FOCEI-oriented conditional objective building blocks",
        "nlme-foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_pharmacometrics_dsl.py",),
        (
            "Provides subject objectives, conditional modes, and Laplace aggregation "
            "only. fit_focei() deliberately rejects because no integrated population "
            "optimizer or FitResult path exists.",
        ),
    ),
    Capability(
        "SAEM",
        "Integrated SAEM population estimator",
        "pharmacometrics-breadth",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (
            "src/pymixef/pharmacometrics/estimation.py",
            "tests/test_pharmacometrics_dsl.py",
        ),
        (
            "A callback-driven research kernel exists, but it is not connected to "
            "ModelIR, event/error-model compilation, population diagnostics, or the "
            "stable FitResult contract.",
        ),
    ),
    Capability(
        "EST-001",
        "Unified convergence object",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_results.py",),
    ),
    Capability(
        "DIAG-001",
        "Diagnostic data-first contract",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_results.py",),
    ),
    Capability(
        "DIAG-002",
        "VPC calculations",
        "pharmacometrics-breadth",
        Maturity.EXPERIMENTAL,
        True,
        _S,
        ("tests/test_results.py",),
        (
            "The 0.1 helper computes binned VPC tables from supplied simulations; it "
            "does not provide an integrated NLME simulation/refit workflow.",
        ),
    ),
    Capability(
        "INT-001",
        "Machine-readable compatibility reports",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_interoperability.py",),
    ),
    Capability(
        "INT-002",
        "Release-gated thin R wrapper",
        "foundation",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (
            "r/pymixef/R/pymixef.R",
            "r/pymixef/man/pymixef-package.Rd",
            "r/pymixef/tests/testthat/test-wrapper-api.R",
            "r/pymixef/tests/testthat/test-python-parity.R",
        ),
        (
            "The alpha reticulate package has Rd files and mocked/live testthat suites, "
            "R CMD build succeeds, and a dependency-complete local R CMD check "
            "--no-manual passes. No cross-platform R CI gate exists, and the "
            "maintainer address is an explicit non-routable placeholder.",
        ),
    ),
    Capability(
        "PERF-001",
        "Reduced JSON benchmark harness",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("benchmarks/run.py",),
        (
            "The current harness contains one reduced synthetic LMM workload and is "
            "not the blueprint's cross-platform performance corpus.",
        ),
    ),
    Capability(
        "PERF-002",
        "Declared reproducibility classes",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_provenance.py",),
    ),
    Capability(
        "REG-001",
        "Validation bundle generator",
        "regulated-workflow-support",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_validation.py",),
        ("Supports evidence generation; it is not a universal validation claim.",),
    ),
    Capability(
        "UX-001",
        "Stable warning catalog",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("src/pymixef/warning_catalog.json",),
    ),
    Capability(
        "UX-002",
        "Deterministic model diff",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_ir.py",),
    ),
)

CAPABILITIES += (
    Capability(
        "ARCH-003",
        "Full backend conformance suite",
        "foundation",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (
            "src/pymixef/backends/base.py",
            "tests/test_backend_conformance.py",
        ),
        (
            "A reusable fit-payload suite covers every built-in backend and checks "
            "validation, deterministic repeat fitting, input immutability, and row "
            "alignment. The blueprint's objective, gradient, optional Hessian-vector "
            "product, and simulation contracts are not yet part of the Backend Protocol.",
        ),
    ),
    Capability(
        "DATA-003",
        "Missingness contract and reason-coded audit",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_data_covariance.py",),
    ),
    Capability(
        "DIST-003",
        "Distributional predictor representation",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_ir.py", "tests/test_families.py"),
        ("Not every represented predictor is executable by the initial Laplace backend.",),
    ),
    Capability(
        "LMM-003",
        "Profile, bootstrap, and robust LMM inference",
        "classical-core",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        ("A generic restartable bootstrap exists; profile and sandwich paths remain gated.",),
    ),
    Capability(
        "GLMM-003",
        "Binary separation indicator",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_glmm_mmrm.py",),
        (
            "The indicator is a heuristic for Bernoulli/binomial fits; calibrated "
            "rare-event diagnostics and recovery benchmarks remain gated.",
        ),
    ),
    Capability(
        "GLMM-004",
        "glmmTMB Salamanders parity",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        ("Zero-inflated NB2 parity remains a later evidence gate.",),
    ),
    Capability(
        "MMRM-002",
        "MMRM covariance construction checks",
        "clinical-longitudinal",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_data_covariance.py", "tests/test_glmm_mmrm.py"),
        (
            "Evidence covers positive-definite construction and local pathology tests, "
            "not external-software covariance-estimate conformance.",
        ),
    ),
    Capability(
        "MMRM-003",
        "Missing-data sensitivity transformations",
        "clinical-longitudinal",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_sensitivity_workflows.py",),
        (
            "Applies audited additive response-scale deltas to explicitly identified, "
            "already-imputed cells. It does not perform imputation or choose clinically "
            "justified sensitivity deltas.",
        ),
    ),
    Capability(
        "NLME-002",
        "Population parameter transforms",
        "nlme-foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_ir.py", "tests/test_pharmacometrics_dsl.py"),
    ),
    Capability(
        "NLME-003",
        "Stable BQL/censoring likelihoods",
        "pharmacometrics-breadth",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_ode_pk.py",),
    ),
    Capability(
        "NLME-004",
        "Interoccasion event representation",
        "pharmacometrics-breadth",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_events.py",),
        ("Population IOV estimation remains part of the production FOCEI gate.",),
    ),
    Capability(
        "NLME-005",
        "Finite-mixture population estimation",
        "pharmacometrics-breadth",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        ("Mixture weights can be represented; label-stable population fitting is unavailable.",),
    ),
    Capability(
        "ODE-003",
        "Independently validated ODE sensitivities",
        "nlme-foundation",
        Maturity.EXPERIMENTAL,
        False,
        None,
        ("src/pymixef/pharmacometrics/ode.py", "tests/test_ode_pk.py"),
        (
            "Forward and central finite-difference diagnostics plus one analytic decay "
            "check exist. Analytic/automatic sensitivities, event-discontinuity cases, "
            "and an independent multi-model validation suite do not.",
        ),
    ),
    Capability(
        "EST-002",
        "Independent derivative verification suite",
        "foundation",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (
            "src/pymixef/backends/base.py",
            "src/pymixef/pharmacometrics/estimation.py",
            "tests/test_pharmacometrics_dsl.py",
            "tests/test_lmm.py",
        ),
        (
            "Finite-difference helpers are used for optimization diagnostics, but no "
            "separate derivative implementation and systematic cross-engine verification "
            "suite is available.",
        ),
    ),
    Capability(
        "EST-003",
        "Approximation sensitivity workflow",
        "generalized-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_sensitivity_workflows.py",),
        (
            "Deep-copied callback refits compare aligned parameters, covariance-derived "
            "standard errors, objectives, and caller-declared materiality flags. Scenario "
            "selection and thresholds are analyst-supplied and archived.",
        ),
    ),
    Capability(
        "INF-001",
        "Uncertainty provenance",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_results.py", "src/pymixef/reporting.py"),
    ),
    Capability(
        "INF-002",
        "Restartable bootstrap with failure accounting",
        "classical-core",
        Maturity.EXPERIMENTAL,
        True,
        _S,
        ("tests/test_random_inference.py",),
        (
            "This is a generic callback/cluster bootstrap helper, not yet an integrated "
            "profile, BCa, or model-specific inference workflow.",
        ),
    ),
    Capability(
        "DIAG-003",
        "Grouping-safe influence analysis",
        "classical-core",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_sensitivity_workflows.py",),
        (
            "Full delete-group refits require archived fixed_effect_rank; Cook-style "
            "distance requires a finite symmetric positive-semidefinite baseline "
            "covariance. Optional approximations are reported beside, never substituted "
            "for, the full refit.",
        ),
    ),
    Capability(
        "ADV-001",
        "Backend-neutral priors exported to two samplers",
        "advanced-engines",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        ("Priors are represented in the IR; two-backend equivalence is not yet validated.",),
    ),
    Capability(
        "ADV-002",
        "Robust sensitivity comparison",
        "advanced-engines",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        (
            "No robust-likelihood fit path or automated cross-model sensitivity report "
            "is implemented.",
        ),
    ),
    Capability(
        "ADV-003",
        "Joint longitudinal-event simulation",
        "advanced-engines",
        Maturity.EXPERIMENTAL,
        False,
        None,
        (),
        (
            "No joint longitudinal/event model, shared random-effect simulator, or "
            "validated event-time likelihood is implemented.",
        ),
    ),
    Capability(
        "PERF-003",
        "Explicit numerical thread controls",
        "foundation",
        Maturity.EXPERIMENTAL,
        False,
        None,
        ("src/pymixef/provenance.py", "tests/test_provenance.py"),
        (
            "Run manifests record ambient thread-related environment variables, but "
            "PyMixEF does not configure or enforce numerical library thread counts.",
        ),
    ),
    Capability(
        "REG-002",
        "Change-impact classification",
        "regulated-workflow-support",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_validation.py",),
    ),
    Capability(
        "VAL-001",
        "Public traceability matrix",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        ReproducibilityClass.BITWISE,
        ("tests/test_validation.py",),
    ),
    Capability(
        "VAL-002",
        "Selected independent reference calculations",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_families.py",),
        (
            "External reference checks currently cover selected normalized family "
            "likelihoods; they are not independent full-engine or cross-software parity "
            "reports.",
        ),
    ),
    Capability(
        "VAL-003",
        "Initial failure and pathology corpus",
        "foundation",
        Maturity.EXPERIMENTAL,
        True,
        _D,
        ("tests/test_formula.py", "tests/test_data_covariance.py", "tests/test_glmm_mmrm.py"),
        (
            "The initial corpus covers representative parser, covariance, and engine "
            "failures; broad adversarial and platform-specific cases remain future work.",
        ),
    ),
)


def get_capability(identifier: str) -> Capability:
    """Look up a capability by stable requirement identifier."""

    for capability in CAPABILITIES:
        if capability.identifier == identifier:
            return capability
    raise KeyError(f"Unknown capability identifier {identifier!r}.")


def iter_capabilities(
    *,
    implemented: bool | None = None,
    stage: str | None = None,
    maturity: Maturity | None = None,
) -> Iterable[Capability]:
    """Filter the immutable capability registry."""

    for capability in CAPABILITIES:
        if implemented is not None and capability.implemented is not implemented:
            continue
        if stage is not None and capability.stage != stage:
            continue
        if maturity is not None and capability.maturity is not maturity:
            continue
        yield capability
