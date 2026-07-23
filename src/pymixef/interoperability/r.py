"""R/lme4/glmmTMB formula translation."""

from __future__ import annotations

from .._contracts import CompatibilityIssue
from .base import CompatibilityReport, InterchangeResult


def translate_r_formula(formula: str) -> InterchangeResult[str]:
    """Translate the supported lme4 formula subset without changing semantics.

    PyMixEF intentionally uses the same operators for its safe formula grammar.
    R function calls and environment lookups are refused because arbitrary code
    evaluation is outside that grammar.
    """

    value = formula.strip()
    detected: list[CompatibilityIssue] = []
    if "~" not in value:
        detected.append(
            CompatibilityIssue(
                "formula",
                "unsupported",
                "An R model formula must contain '~'.",
            )
        )
    else:
        detected.append(
            CompatibilityIssue(
                "formula operators",
                "exact",
                "Fixed effects, interactions, nesting, and random |/|| operators "
                "share PyMixEF semantics.",
            )
        )
    unsafe_tokens = ("::", "$", "[[", "I(", "poly(", "ns(", "bs(")
    for token in unsafe_tokens:
        if token in value:
            detected.append(
                CompatibilityIssue(
                    token,
                    "unsupported",
                    "R environment/function evaluation is not executed by PyMixEF; "
                    "create the transformed column explicitly.",
                )
            )
    return InterchangeResult(
        value=value,
        report=CompatibilityReport(
            source_format="R formula",
            target_format="PyMixEF formula",
            issues=tuple(detected),
        ),
    )
