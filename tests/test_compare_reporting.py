from __future__ import annotations

from pathlib import Path

from pymixef.compare import compare
from pymixef.reporting import render_report

from .test_results import _result


def test_comparison_requires_explicit_conventions(tmp_path: Path) -> None:
    fit = _result()
    result = compare(
        fit,
        reference={
            "parameters": {"REF": 1.51, "residual_sd": 0.49},
            "objective": 2.2,
        },
        mapping={"beta[Intercept]": "REF"},
        conventions={
            "method": "reml",
            "objective_convention": "negative normalized REML log likelihood",
            "likelihood_includes_data_constants": True,
        },
    )
    assert result.compatibility.supported
    assert result.objective_difference is not None
    target = result.write_report(tmp_path / "comparison.html")
    assert target.exists()


def test_comparison_refuses_mismatched_objective_conventions() -> None:
    result = compare(
        _result(),
        reference={"parameters": _result().parameters, "objective": 4.0},
        conventions={
            "method": "reml",
            "objective_convention": "-2 log likelihood",
            "likelihood_includes_data_constants": True,
        },
    )
    assert not result.compatibility.supported
    assert result.objective_difference is None


def test_manifest_driven_markdown_report(tmp_path: Path) -> None:
    target = render_report(_result(), tmp_path / "report.md")
    text = target.read_text(encoding="utf-8")
    assert "# PyMixEF model result" in text
    assert "not a universal regulatory validation certificate" in text
