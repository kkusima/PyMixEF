from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pymixef.capabilities import CAPABILITIES
from pymixef.errors import ValidationError
from pymixef.validation import (
    TRACEABILITY_LINKS,
    change_impact,
    create_validation_bundle,
    traceability_matrix,
    verify_validation_bundle,
)

from .test_results import _result


def test_validation_bundle_is_self_verifying(tmp_path: Path) -> None:
    target = create_validation_bundle(_result(), tmp_path / "bundle.zip")
    verified = verify_validation_bundle(target)
    assert verified["valid"]
    assert "result.json" in verified["files"]
    assert "analysis-data.json" not in verified["files"]


def test_validation_bundle_rejects_unlisted_members(tmp_path: Path) -> None:
    target = create_validation_bundle(_result(), tmp_path / "bundle.zip")
    with zipfile.ZipFile(target, "a") as archive:
        archive.writestr("unlisted.txt", b"not in SHA256SUMS")
    with pytest.raises(ValidationError) as captured:
        verify_validation_bundle(target)
    assert captured.value.code == "VALIDATION-BUNDLE-MEMBERS-001"


def test_traceability_and_change_impact() -> None:
    matrix = traceability_matrix()
    requirements = {item.requirement for item in matrix}
    assert {"ARCH-001", "LMM-001", "REG-001"} <= requirements
    sparse_lmm = next(item for item in matrix if item.requirement == "LMM-002")
    assert not sparse_lmm.implemented
    assert sparse_lmm.reproducibility is None
    impact = change_impact(["src/pymixef/backends/lmm.py"])
    assert impact["classifications"]["statistical-method"]
    assert "tests/test_lmm.py" in impact["recommended_reruns"]


def test_traceability_mapping_is_complete_and_paths_are_concrete() -> None:
    repository = Path(__file__).parents[1]
    capability_ids = {item.identifier for item in CAPABILITIES}
    assert set(TRACEABILITY_LINKS) == capability_ids

    for record in traceability_matrix():
        assert record.specification_files
        if record.implemented:
            assert record.source_files
            assert any(path.endswith(".py") for path in record.source_files)
        for path in (*record.source_files, *record.specification_files):
            assert not Path(path).is_absolute()
            assert ".." not in Path(path).parts
            assert (repository / path).is_file(), (record.requirement, path)


def test_gated_requirements_cannot_be_mistaken_for_partial_primitives() -> None:
    capabilities = {item.identifier: item for item in CAPABILITIES}
    expected_gated = {
        "ARCH-003",
        "EST-002",
        "INT-002",
        "ODE-003",
        "PERF-003",
        "SAEM",
    }
    for identifier in expected_gated:
        capability = capabilities[identifier]
        assert not capability.implemented
        assert capability.reproducibility is None
        assert capability.limitations

    assert all(item.limitations for item in CAPABILITIES if not item.implemented)


def test_promoted_workflows_have_integrated_test_evidence() -> None:
    capabilities = {item.identifier: item for item in CAPABILITIES}
    expected_evidence = {
        "MMRM-003": "tests/test_sensitivity_workflows.py",
        "EST-003": "tests/test_sensitivity_workflows.py",
        "DIAG-003": "tests/test_sensitivity_workflows.py",
    }
    for identifier, evidence in expected_evidence.items():
        capability = capabilities[identifier]
        assert capability.implemented
        assert capability.reproducibility is not None
        assert evidence in capability.evidence
        assert capability.limitations


def test_traceability_serialization_includes_specifications() -> None:
    record = next(item for item in traceability_matrix() if item.requirement == "LMM-001")
    payload = record.to_dict()
    assert payload["source_files"] == ["src/pymixef/backends/lmm.py"]
    assert payload["specification_files"] == ["docs/methods/lmm.md"]
