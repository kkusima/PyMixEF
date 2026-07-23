"""Traceability, validation evidence, and change-impact bundles."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._serialization import canonical_json
from .capabilities import CAPABILITIES
from .errors import ValidationError
from .results import FitResult


@dataclass(frozen=True, slots=True)
class RequirementLinks:
    """Version-controlled implementation and specification links."""

    source_files: tuple[str, ...]
    specification_files: tuple[str, ...]


# This mapping is deliberately separate from test/evidence declarations in the
# capability registry.  A green requirement must point to both the code that
# implements its deliberately narrow 0.1 scope and the document that specifies
# that scope.  Entries for gated requirements may point to partial primitives;
# their ``implemented`` flag and limitations remain authoritative.
TRACEABILITY_LINKS: dict[str, RequirementLinks] = {
    "ARCH-001": RequirementLinks(("src/pymixef/ir.py",), ("docs/concepts/model-ir.md",)),
    "ARCH-002": RequirementLinks(("src/pymixef/model.py",), ("docs/concepts/model-ir.md",)),
    "ARCH-003": RequirementLinks(
        (
            "src/pymixef/backends/base.py",
            "src/pymixef/backends/lmm.py",
            "src/pymixef/backends/glmm.py",
            "src/pymixef/backends/mmrm.py",
        ),
        ("docs/developer.md", "docs/reference/capability-catalog.md"),
    ),
    "API-001": RequirementLinks(
        ("src/pymixef/model.py", "src/pymixef/formula.py"),
        ("docs/concepts/model-ir.md",),
    ),
    "API-002": RequirementLinks(
        ("src/pymixef/results.py",),
        ("docs/concepts/conventions.md",),
    ),
    "API-003": RequirementLinks(("src/pymixef/data.py",), ("docs/concepts/conventions.md",)),
    "COV-001": RequirementLinks(
        ("src/pymixef/covariance.py",),
        ("docs/concepts/conventions.md",),
    ),
    "COV-002": RequirementLinks(
        ("src/pymixef/covariance.py", "src/pymixef/diagnostics.py"),
        ("docs/methods/lmm.md",),
    ),
    "DATA-001": RequirementLinks(
        ("src/pymixef/pharmacometrics/events.py",),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "DATA-002": RequirementLinks(
        ("src/pymixef/pharmacometrics/events.py",),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "DATA-003": RequirementLinks(("src/pymixef/data.py",), ("docs/concepts/conventions.md",)),
    "DIST-001": RequirementLinks(
        ("src/pymixef/families.py",),
        ("docs/concepts/conventions.md",),
    ),
    "DIST-002": RequirementLinks(
        ("src/pymixef/families.py",),
        ("docs/concepts/conventions.md",),
    ),
    "DIST-003": RequirementLinks(
        ("src/pymixef/ir.py", "src/pymixef/families.py"),
        ("docs/concepts/model-ir.md", "docs/concepts/conventions.md"),
    ),
    "LMM-001": RequirementLinks(("src/pymixef/backends/lmm.py",), ("docs/methods/lmm.md",)),
    "LMM-002": RequirementLinks((), ("docs/methods/lmm.md",)),
    "LMM-003": RequirementLinks(
        ("src/pymixef/inference.py",),
        ("docs/reference/capability-catalog.md",),
    ),
    "GLMM-001": RequirementLinks(
        ("src/pymixef/backends/glmm.py",),
        ("docs/methods/glmm.md",),
    ),
    "GLMM-002": RequirementLinks((), ("docs/methods/glmm.md",)),
    "GLMM-003": RequirementLinks(
        ("src/pymixef/backends/glmm.py",),
        ("docs/methods/glmm.md",),
    ),
    "GLMM-004": RequirementLinks(
        ("src/pymixef/families.py",),
        ("docs/methods/glmm.md",),
    ),
    "MMRM-001": RequirementLinks(
        ("src/pymixef/backends/mmrm.py",),
        ("docs/methods/mmrm.md",),
    ),
    "MMRM-002": RequirementLinks(
        ("src/pymixef/backends/mmrm.py", "src/pymixef/covariance.py"),
        ("docs/methods/mmrm.md",),
    ),
    "MMRM-003": RequirementLinks(
        ("src/pymixef/data.py",),
        ("docs/methods/mmrm.md", "docs/reference/capability-catalog.md"),
    ),
    "ODE-001": RequirementLinks(
        ("src/pymixef/pharmacometrics/ode.py",),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "ODE-002": RequirementLinks(
        ("src/pymixef/pharmacometrics/ode.py", "src/pymixef/pharmacometrics/events.py"),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "ODE-003": RequirementLinks(
        ("src/pymixef/pharmacometrics/ode.py",),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "NLME-001": RequirementLinks(
        (
            "src/pymixef/pharmacometrics/dsl.py",
            "src/pymixef/pharmacometrics/estimation.py",
        ),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "NLME-002": RequirementLinks(
        ("src/pymixef/transforms.py", "src/pymixef/pharmacometrics/dsl.py"),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "NLME-003": RequirementLinks(
        ("src/pymixef/pharmacometrics/pk.py", "src/pymixef/pharmacometrics/estimation.py"),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "NLME-004": RequirementLinks(
        ("src/pymixef/pharmacometrics/events.py", "src/pymixef/pharmacometrics/dsl.py"),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "NLME-005": RequirementLinks(
        ("src/pymixef/ir.py", "src/pymixef/families.py"),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "SAEM": RequirementLinks(
        ("src/pymixef/pharmacometrics/estimation.py",),
        ("docs/pharmacometrics/events-and-ode.md",),
    ),
    "EST-001": RequirementLinks(
        ("src/pymixef/convergence.py",),
        ("docs/concepts/conventions.md",),
    ),
    "EST-002": RequirementLinks(
        ("src/pymixef/backends/base.py", "src/pymixef/pharmacometrics/estimation.py"),
        ("docs/reference/capability-catalog.md",),
    ),
    "EST-003": RequirementLinks(
        ("src/pymixef/compare.py",),
        ("docs/reference/capability-catalog.md",),
    ),
    "INF-001": RequirementLinks(
        ("src/pymixef/results.py", "src/pymixef/reporting.py"),
        ("docs/concepts/conventions.md",),
    ),
    "INF-002": RequirementLinks(
        ("src/pymixef/inference.py", "src/pymixef/random.py"),
        ("docs/reference/capability-catalog.md",),
    ),
    "DIAG-001": RequirementLinks(
        ("src/pymixef/diagnostics.py", "src/pymixef/results.py"),
        ("docs/concepts/conventions.md",),
    ),
    "DIAG-002": RequirementLinks(
        ("src/pymixef/diagnostics.py", "src/pymixef/results.py"),
        ("docs/reference/capability-catalog.md",),
    ),
    "DIAG-003": RequirementLinks(
        ("src/pymixef/diagnostics.py",),
        ("docs/reference/capability-catalog.md",),
    ),
    "INT-001": RequirementLinks(
        ("src/pymixef/interoperability/base.py",),
        ("docs/migration/interoperability.md",),
    ),
    "INT-002": RequirementLinks(
        ("r/pymixef/R/pymixef.R",),
        ("docs/migration/interoperability.md", "r/pymixef/README.md"),
    ),
    "PERF-001": RequirementLinks(
        ("benchmarks/run.py", "src/pymixef/provenance.py"),
        ("docs/reference/capability-catalog.md",),
    ),
    "PERF-002": RequirementLinks(
        ("src/pymixef/_contracts.py", "src/pymixef/provenance.py"),
        ("docs/reference/capability-catalog.md",),
    ),
    "PERF-003": RequirementLinks(
        ("src/pymixef/provenance.py",),
        ("docs/reference/capability-catalog.md",),
    ),
    "ADV-001": RequirementLinks(
        ("src/pymixef/ir.py",),
        ("docs/reference/capability-catalog.md",),
    ),
    "ADV-002": RequirementLinks((), ("docs/reference/capability-catalog.md",)),
    "ADV-003": RequirementLinks((), ("docs/reference/capability-catalog.md",)),
    "REG-001": RequirementLinks(("src/pymixef/validation.py",), ("docs/validation.md",)),
    "REG-002": RequirementLinks(("src/pymixef/validation.py",), ("docs/validation.md",)),
    "UX-001": RequirementLinks(("src/pymixef/warnings.py",), ("docs/warnings.md",)),
    "UX-002": RequirementLinks(("src/pymixef/ir.py",), ("docs/concepts/model-ir.md",)),
    "VAL-001": RequirementLinks(
        ("src/pymixef/validation.py", "src/pymixef/capabilities.py"),
        ("docs/validation.md",),
    ),
    "VAL-002": RequirementLinks(
        ("src/pymixef/families.py",),
        ("docs/validation.md",),
    ),
    "VAL-003": RequirementLinks(
        ("src/pymixef/errors.py", "src/pymixef/formula.py", "src/pymixef/data.py"),
        ("docs/validation.md",),
    ),
}


@dataclass(frozen=True, slots=True)
class TraceabilityRecord:
    """Requirement-to-specification/source/test/evidence link."""

    requirement: str
    capability: str
    stage: str
    maturity: str
    implemented: bool
    reproducibility: str | None
    source_files: tuple[str, ...]
    specification_files: tuple[str, ...]
    tests: tuple[str, ...]
    evidence: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "capability": self.capability,
            "stage": self.stage,
            "maturity": self.maturity,
            "implemented": self.implemented,
            "reproducibility": self.reproducibility,
            "source_files": list(self.source_files),
            "specification_files": list(self.specification_files),
            "tests": list(self.tests),
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
        }


def traceability_matrix() -> tuple[TraceabilityRecord, ...]:
    """Return the built-in public traceability matrix."""

    records: list[TraceabilityRecord] = []
    for capability in CAPABILITIES:
        links = TRACEABILITY_LINKS[capability.identifier]
        tests = tuple(item for item in capability.evidence if item.startswith("tests/"))
        evidence_sources = tuple(
            item
            for item in capability.evidence
            if item.startswith(("src/", "benchmarks/")) and item.endswith(".py")
        )
        sources = tuple(dict.fromkeys((*links.source_files, *evidence_sources)))
        classified = {*tests, *sources, *links.specification_files}
        other = tuple(item for item in capability.evidence if item not in classified)
        records.append(
            TraceabilityRecord(
                requirement=capability.identifier,
                capability=capability.name,
                stage=capability.stage,
                maturity=capability.maturity.value,
                implemented=capability.implemented,
                reproducibility=(
                    None if capability.reproducibility is None else capability.reproducibility.value
                ),
                source_files=sources,
                specification_files=links.specification_files,
                tests=tests,
                evidence=other,
                limitations=capability.limitations,
            )
        )
    return tuple(records)


def classify_change(path: str | os.PathLike[str]) -> str:
    """Classify a changed path for targeted validation reruns."""

    value = Path(path).as_posix()
    if value.startswith(("docs/", "README", "CONTRIBUTING", "GOVERNANCE")):
        return "documentation-only"
    if value.startswith(("src/pymixef/backends/", "src/pymixef/families.py")):
        return "statistical-method"
    if value.startswith("src/pymixef/pharmacometrics/"):
        return "numerical"
    if value.startswith(("pyproject.toml", "requirements", "uv.lock")):
        return "dependency"
    if value.startswith((".github/", "SECURITY")):
        return "security-or-build"
    if value.startswith("src/"):
        return "api"
    return "other"


def change_impact(paths: Iterable[str | os.PathLike[str]]) -> dict[str, Any]:
    """Classify changed paths and recommend a conservative validation subset.

    This static policy only reports test targets; it does not inspect diffs or
    execute the recommended tests.
    """

    classified: dict[str, list[str]] = {}
    for path in paths:
        category = classify_change(path)
        classified.setdefault(category, []).append(Path(path).as_posix())
    rerun: set[str] = {"tests/test_provenance.py"}
    if {"statistical-method", "numerical"} & classified.keys():
        rerun.update(
            {
                "tests/test_families.py",
                "tests/test_lmm.py",
                "tests/test_glmm_mmrm.py",
                "tests/test_ode_pk.py",
            }
        )
    if "api" in classified:
        rerun.update({"tests/test_ir.py", "tests/test_model.py", "tests/test_results.py"})
    if "dependency" in classified:
        rerun.add("full cross-platform suite")
    return {"classifications": classified, "recommended_reruns": sorted(rerun)}


_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def _zip_write(archive: zipfile.ZipFile, name: str, payload: bytes) -> str:
    information = zipfile.ZipInfo(name, _ZIP_TIME)
    information.compress_type = zipfile.ZIP_DEFLATED
    information.external_attr = 0o644 << 16
    archive.writestr(information, payload)
    return hashlib.sha256(payload).hexdigest()


def create_validation_bundle(
    result: FitResult,
    path: str | Path,
    *,
    analysis_data: Any | None = None,
    include_data: bool = False,
    additional_files: Sequence[str | Path] = (),
) -> Path:
    """Create a deterministic, self-describing validation evidence archive.

    Raw analysis data is excluded by default.  Set ``include_data=True`` only
    after confirming that the destination may contain the supplied data.
    """

    if analysis_data is not None and not include_data:
        data_note = (
            "Raw data were intentionally excluded. The run manifest contains the "
            f"input fingerprint {result.manifest.data_hash}."
        )
    elif analysis_data is None:
        data_note = "No raw analysis data were supplied to the bundle generator."
    else:
        data_note = "Raw analysis data are included as canonical JSON."
    files: dict[str, bytes] = {
        "result.json": (canonical_json(result.to_dict()) + "\n").encode("utf-8"),
        "manifest.json": (canonical_json(result.manifest.to_dict()) + "\n").encode("utf-8"),
        "traceability.json": (
            canonical_json([item.to_dict() for item in traceability_matrix()]) + "\n"
        ).encode("utf-8"),
        "README.txt": (
            "PyMixEF validation evidence bundle\n\n"
            + data_note
            + "\n\nThis archive supports context-specific validation. It is not a "
            "universal regulatory validation certificate.\n"
        ).encode("utf-8"),
    }
    if analysis_data is not None and include_data:
        files["analysis-data.json"] = (canonical_json(analysis_data) + "\n").encode("utf-8")
    for source_value in additional_files:
        source = Path(source_value)
        if not source.is_file():
            raise FileNotFoundError(source)
        files[f"attachments/{source.name}"] = source.read_bytes()
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(files.items())}
    files["SHA256SUMS.json"] = (json.dumps(hashes, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for name, payload in sorted(files.items()):
            _zip_write(archive, name, payload)
    return destination


def verify_validation_bundle(path: str | Path) -> dict[str, Any]:
    """Verify internal hashes and return the archived manifest."""

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        expected = json.loads(archive.read("SHA256SUMS.json"))
        expected_names = set(expected)
        actual_names = set(names)
        required_names = expected_names | {"SHA256SUMS.json"}
        unexpected = sorted(actual_names - required_names)
        missing = sorted(required_names - actual_names)
        if duplicates or unexpected or missing:
            raise ValidationError(
                "Validation bundle member set does not match its manifest.",
                code="VALIDATION-BUNDLE-MEMBERS-001",
                details={
                    "duplicates": duplicates,
                    "unexpected": unexpected,
                    "missing": missing,
                },
            )
        failures: dict[str, dict[str, str]] = {}
        for name, digest in expected.items():
            observed = hashlib.sha256(archive.read(name)).hexdigest()
            if observed != digest:
                failures[name] = {"expected": digest, "observed": observed}
        manifest = json.loads(archive.read("manifest.json"))
    if failures:
        raise ValidationError(
            "Validation bundle hash verification failed.",
            code="VALIDATION-BUNDLE-HASH-001",
            details={"failures": failures},
        )
    return {"valid": True, "manifest": manifest, "files": sorted(expected)}
