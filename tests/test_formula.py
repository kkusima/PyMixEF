from __future__ import annotations

import numpy as np
import pytest

from pymixef.backends.base import prepare_data as prepare_backend_data
from pymixef.errors import FormulaError
from pymixef.formula import compile_formula, explain_formula, parse_formula


@pytest.fixture
def longitudinal_data() -> dict[str, list[object]]:
    return {
        "y": [1.0, 1.8, 2.5, 2.0, 3.1, 4.2],
        "treatment": ["A", "A", "A", "B", "B", "B"],
        "visit": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        "subject": ["s1", "s1", "s1", "s2", "s2", "s2"],
        "site": ["north", "north", "north", "south", "south", "south"],
    }


def test_parse_interactions_and_random_operators() -> None:
    spec = parse_formula("y ~ treatment * visit + (1 + visit | subject) + (0 + visit || site)")
    assert spec.response == "y"
    assert spec.fixed_terms == ("treatment", "visit", "treatment:visit")
    assert spec.random_terms[0].term_names == ("Intercept", "visit")
    assert spec.random_terms[0].correlated
    assert spec.random_terms[1].term_names == ("visit",)
    assert not spec.random_terms[1].correlated


def test_nesting_expands_fixed_and_grouping_terms() -> None:
    spec = parse_formula("y ~ treatment / visit + (1 | site / subject)")
    assert spec.fixed_terms == ("treatment", "treatment:visit")
    assert [term.group for term in spec.random_terms] == ["site", "site:subject"]


def test_compile_deterministic_design_and_backend_contract(longitudinal_data: dict) -> None:
    compiled = compile_formula(
        "y ~ treatment * visit + (visit | subject)",
        longitudinal_data,
    )
    np.testing.assert_allclose(
        compiled.fixed,
        [
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 2, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 2, 2],
        ],
    )
    assert compiled.fixed_names == (
        "Intercept",
        "treatment[B]",
        "visit",
        "treatment[B]:visit",
    )
    block = compiled.random_blocks[0]
    assert block.term_names == ("Intercept", "visit")
    assert block.group_levels == ("s1", "s2")
    np.testing.assert_array_equal(block.group_codes, [0, 0, 0, 1, 1, 1])
    backend = prepare_backend_data(compiled)
    assert backend.X.shape == (6, 4)
    assert backend.random_design.shape == (6, 4)
    assert (
        compiled.row_ids
        == compile_formula("y ~ treatment * visit + (visit | subject)", longitudinal_data).row_ids
    )


def test_safe_transform_and_no_eval(longitudinal_data: dict) -> None:
    compiled = compile_formula(
        "y ~ center(visit) + I(visit ** 2) + (1 | subject)",
        longitudinal_data,
    )
    np.testing.assert_allclose(compiled.fixed[:, 1], [-1, 0, 1, -1, 0, 1])
    np.testing.assert_allclose(compiled.fixed[:, 2], [0, 1, 4, 0, 1, 4])
    with pytest.raises(FormulaError, match=r"Unsafe|safe"):
        compile_formula(
            'y ~ __import__("os").system("echo should-not-run")',
            longitudinal_data,
        )
    with pytest.raises(FormulaError, match="safe"):
        compile_formula("y ~ visit.__class__", longitudinal_data)


def test_missing_rows_are_audited(longitudinal_data: dict) -> None:
    data = dict(longitudinal_data)
    data["y"] = list(data["y"])
    data["visit"] = list(data["visit"])
    data["y"][1] = np.nan
    data["visit"][4] = np.nan
    compiled = compile_formula("y ~ visit + (1 | subject)", data)
    assert compiled.response.size == 4
    assert compiled.audit.input_rows == 6
    assert compiled.audit.excluded_rows == 2
    assert {record.reason_code for record in compiled.audit.records} >= {
        "DATA-MISSING-RESPONSE-001",
        "DATA-MISSING-COVARIATE-001",
    }
    with pytest.raises(Exception, match="missingness"):
        compile_formula("y ~ visit", data, missing="raise")


def test_declared_categorical_reference_survives_response_filtering() -> None:
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(
        {
            "y": [np.nan, 1.0, 2.0],
            "arm": pd.Categorical(
                ["placebo", "active", "active"],
                categories=["placebo", "active", "rescue"],
                ordered=True,
            ),
        }
    )

    compiled = compile_formula("y ~ arm", data)

    assert compiled.factor_levels["arm"] == ("placebo", "active", "rescue")
    assert compiled.audit.factor_levels["arm"] == ("placebo", "active", "rescue")
    assert compiled.fixed_names == ("Intercept", "arm[active]", "arm[rescue]")
    np.testing.assert_allclose(compiled.fixed, [[1, 1, 0], [1, 1, 0]])


def test_mapping_reference_is_inferred_from_source_before_filtering() -> None:
    compiled = compile_formula(
        "y ~ arm",
        {
            "y": [np.nan, 1.0, 2.0],
            "arm": ["control", "treated", "treated"],
        },
    )

    assert compiled.factor_levels["arm"] == ("control", "treated")
    assert compiled.fixed_names == ("Intercept", "arm[treated]")
    np.testing.assert_allclose(compiled.fixed, [[1, 1], [1, 1]])


def test_single_level_mapping_factor_has_an_empty_treatment_block() -> None:
    compiled = compile_formula(
        "y ~ cohort",
        {
            "y": [1.0, 2.0, 3.0],
            "cohort": ["only", "only", "only"],
        },
    )

    assert compiled.factor_levels["cohort"] == ("only",)
    assert compiled.fixed_names == ("Intercept",)
    np.testing.assert_allclose(compiled.fixed, np.ones((3, 1)))


@pytest.fixture
def crossed_factors() -> dict[str, list[object]]:
    return {
        "y": [1.0, 2.0, 3.0, 4.0],
        "a": ["A", "A", "B", "B"],
        "b": ["C", "D", "C", "D"],
        "x": [1.0, 2.0, 3.0, 4.0],
    }


def test_interaction_only_factors_retain_the_full_cell_mean_span(
    crossed_factors: dict[str, list[object]],
) -> None:
    compiled = compile_formula("y ~ a:b", crossed_factors)

    assert compiled.fixed_names == (
        "Intercept",
        "a[A]:b[C]",
        "a[A]:b[D]",
        "a[B]:b[C]",
        "a[B]:b[D]",
    )
    np.testing.assert_allclose(
        compiled.fixed,
        [
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ],
    )
    assert np.linalg.matrix_rank(compiled.fixed) == 4
    assert compiled.contrast_coding == {"a": "full", "b": "full"}

    no_intercept = compile_formula("y ~ 0 + a:b", crossed_factors)
    assert no_intercept.fixed_names == compiled.fixed_names[1:]
    np.testing.assert_allclose(no_intercept.fixed, np.eye(4))


def test_hierarchical_interactions_use_r_compatible_contrast_spans(
    crossed_factors: dict[str, list[object]],
) -> None:
    factorial = compile_formula("y ~ a * b", crossed_factors)
    assert factorial.fixed_names == (
        "Intercept",
        "a[B]",
        "b[D]",
        "a[B]:b[D]",
    )
    assert np.linalg.matrix_rank(factorial.fixed) == 4

    one_main_effect = compile_formula("y ~ a + a:b", crossed_factors)
    assert one_main_effect.fixed_names == (
        "Intercept",
        "a[B]",
        "a[A]:b[D]",
        "a[B]:b[D]",
    )
    assert np.linalg.matrix_rank(one_main_effect.fixed) == 4
    assert one_main_effect.contrast_coding == {
        "a": "term-dependent",
        "b": "treatment",
    }


def test_numeric_by_factor_interaction_does_not_drop_a_group_slope(
    crossed_factors: dict[str, list[object]],
) -> None:
    interaction_only = compile_formula("y ~ a:x", crossed_factors)
    assert interaction_only.fixed_names == ("Intercept", "a[A]:x", "a[B]:x")
    assert np.linalg.matrix_rank(interaction_only.fixed) == 3

    with_numeric_main = compile_formula("y ~ x + a:x", crossed_factors)
    assert with_numeric_main.fixed_names == ("Intercept", "x", "a[B]:x")
    assert np.linalg.matrix_rank(with_numeric_main.fixed) == 3


def test_composite_grouping_keys_cannot_collide_on_separator_text() -> None:
    compiled = compile_formula(
        "y ~ 1 + (1 | first:second)",
        {
            "y": [1.0, 2.0],
            "first": ["a:b", "a"],
            "second": ["c", "b:c"],
        },
    )

    block = compiled.random_blocks[0]
    assert block.group_labels == (("a:b", "c"), ("a", "b:c"))
    assert len(block.group_levels) == 2
    assert block.group_codes[0] != block.group_codes[1]


def test_dry_run_explains_dimensions_rank_and_covariance(longitudinal_data: dict) -> None:
    explanation = explain_formula(
        "y ~ treatment * visit + (visit || subject)",
        longitudinal_data,
    )
    assert "Fixed design: X(6, 4), rank=4" in explanation
    assert "covariance=diagonal" in explanation
    assert "Excluded source rows: 0" in explanation
