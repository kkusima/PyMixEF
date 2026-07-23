"""PyMixEF: mixed-effects statistics and pharmacometrics in Python.

The package is fully offline and emits no telemetry. Public capabilities carry
explicit evidence/maturity labels; use :func:`iter_capabilities` or the CLI
``pymixef capabilities`` before relying on an experimental calculation path.
"""

from __future__ import annotations

from . import covariance, diagnostics, families, interoperability, pharmacometrics
from ._contracts import Maturity, ReproducibilityClass, WarningRecord
from ._version import __version__
from .capabilities import CAPABILITIES, get_capability, iter_capabilities
from .compare import (
    ApproximationSensitivityResult,
    ComparisonResult,
    approximation_sensitivity,
    compare,
)
from .convergence import (
    BoundaryRecord,
    ConvergenceReport,
    HessianDiagnostics,
)
from .data import PatternMixtureResult, pattern_mixture_adjust
from .diagnostics import GroupInfluenceResult, group_influence
from .errors import (
    CompatibilityError,
    CovarianceError,
    DataError,
    EngineCompatibilityError,
    FormulaError,
    IRValidationError,
    IRVersionError,
    PluginError,
    PyMixEFError,
    UnsupportedCapabilityError,
    ValidationError,
)
from .inference import BootstrapResult, bootstrap
from .ir import ModelIR, diff_models
from .model import ExecutionPlan, Fixed, Model, Random, Response, fit
from .provenance import RunManifest
from .random import RandomStreamManager, random_streams
from .reporting import render_report
from .results import FitResult
from .validation import (
    change_impact,
    create_validation_bundle,
    traceability_matrix,
    verify_validation_bundle,
)

cov = covariance
load = FitResult.load

__all__ = [
    "CAPABILITIES",
    "ApproximationSensitivityResult",
    "BootstrapResult",
    "BoundaryRecord",
    "ComparisonResult",
    "CompatibilityError",
    "ConvergenceReport",
    "CovarianceError",
    "DataError",
    "EngineCompatibilityError",
    "ExecutionPlan",
    "FitResult",
    "Fixed",
    "FormulaError",
    "GroupInfluenceResult",
    "HessianDiagnostics",
    "IRValidationError",
    "IRVersionError",
    "Maturity",
    "Model",
    "ModelIR",
    "PatternMixtureResult",
    "PluginError",
    "PyMixEFError",
    "Random",
    "RandomStreamManager",
    "ReproducibilityClass",
    "Response",
    "RunManifest",
    "UnsupportedCapabilityError",
    "ValidationError",
    "WarningRecord",
    "__version__",
    "approximation_sensitivity",
    "bootstrap",
    "change_impact",
    "compare",
    "cov",
    "covariance",
    "create_validation_bundle",
    "diagnostics",
    "diff_models",
    "families",
    "fit",
    "get_capability",
    "group_influence",
    "interoperability",
    "iter_capabilities",
    "load",
    "pattern_mixture_adjust",
    "pharmacometrics",
    "random_streams",
    "render_report",
    "traceability_matrix",
    "verify_validation_bundle",
]
