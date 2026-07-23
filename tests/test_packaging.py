from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_extra_and_source_manifest_are_complete() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    requirements = metadata["project"]["optional-dependencies"]["notebooks"]
    for package in ("ipykernel", "jupyterlab", "matplotlib", "nbclient", "nbformat"):
        assert any(requirement.startswith(package) for requirement in requirements)

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include examples *.py *.ipynb *.md" in manifest
    assert "prune examples/notebooks/.ipynb_checkpoints" in manifest
    assert "recursive-include scripts *.py" in manifest


def test_documentation_extra_and_read_the_docs_configuration_are_complete() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    requirements = metadata["project"]["optional-dependencies"]["docs"]
    for package in (
        "Sphinx",
        "myst-parser",
        "pydata-sphinx-theme",
        "sphinx-codeautolink",
        "sphinx-copybutton",
        "sphinx-design",
    ):
        assert any(requirement.startswith(package) for requirement in requirements)

    read_the_docs = (ROOT / ".readthedocs.yaml").read_text(encoding="utf-8")
    assert "version: 2" in read_the_docs
    assert "configuration: docs/conf.py" in read_the_docs
    assert "fail_on_warning: true" in read_the_docs
    assert "extra_requirements:" in read_the_docs
    assert "- docs" in read_the_docs

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "docs:" in makefile
    assert "sphinx -W --keep-going -b dirhtml" in makefile
    assert "extract_notebook_figures.py --check" in makefile
    assert "scripts/audit_documentation.py" in makefile

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python scripts/audit_documentation.py" in workflow


def test_release_check_builds_and_runs_strict_twine_validation() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "PYTHON ?= python" in makefile
    assert "ruff check src tests benchmarks examples scripts" in makefile
    assert "ruff format --check src tests benchmarks examples scripts" in makefile
    release_target = makefile.split("release-check:", maxsplit=1)[1].split(
        "\n\n",
        maxsplit=1,
    )[0]
    assert "release-check: notebooks" in makefile
    assert "notebooks-refresh:" in makefile
    assert "scripts/validate_notebooks.py --refresh" in makefile
    assert "$(PYTHON) -m build" in release_target
    assert "$(PYTHON) -m twine check --strict dist/*" in release_target


def test_publish_workflow_uses_a_minimal_trusted_publisher_job() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "types: [published]" in workflow
    assert "workflow_dispatch" not in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert 'python-version: "3.13"' in workflow
    assert "password:" not in workflow
    assert "api-token:" not in workflow
    assert workflow.count("id-token: write") == 1
    assert "RELEASE_TAG: ${{ github.event.release.tag_name }}" in workflow
    assert 'python scripts/check_release_tag.py "$RELEASE_TAG"' in workflow
    assert 'check_release_tag.py "${{' not in workflow
    notebook_gate = workflow.index("python scripts/validate_notebooks.py")
    build = workflow.index("python -m build")
    upload = workflow.index("actions/upload-artifact")
    assert notebook_gate < build < upload

    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "notebooks:\n    name: Tutorial notebooks" in ci
    assert 'python -m pip install -e ".[notebooks]"' in ci
    assert "python scripts/validate_notebooks.py" in ci


def test_release_tag_guard_accepts_only_the_static_project_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    command = [sys.executable, str(ROOT / "scripts/check_release_tag.py")]

    accepted = subprocess.run(
        [*command, f"v{version}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr

    rejected = subprocess.run(
        [*command, "v999.0.0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "does not match expected tag" in rejected.stderr


def test_committed_notebook_results_are_current_and_error_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_notebooks.py"),
            "--no-execute",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
