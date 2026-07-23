from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_URL = "https://pymixef.readthedocs.io/en/latest/"
REPOSITORY_URL = "https://github.com/kkusima/PyMixEF"


def _workflow_job(workflow: str, name: str) -> str:
    """Return one top-level job block from a GitHub Actions workflow."""

    jobs = workflow.split("\njobs:\n", maxsplit=1)[1]
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        jobs,
    )
    assert match is not None, f"publish workflow is missing the {name!r} job"
    return match.group(0)


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


def test_public_project_links_are_absolute_and_consistent() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    assert metadata["project"]["urls"] == {
        "Homepage": REPOSITORY_URL,
        "Documentation": DOCS_URL,
        "Repository": REPOSITORY_URL,
        "Issues": f"{REPOSITORY_URL}/issues",
        "Changelog": f"{REPOSITORY_URL}/blob/main/CHANGELOG.md",
    }

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert DOCS_URL in readme
    assert "Before the first PyPI release" not in readme
    for relative_target in (
        "](docs/",
        "](examples/",
        "](validation/",
    ):
        assert relative_target not in readme


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

    conf = (ROOT / "docs/conf.py").read_text(encoding="utf-8")
    assert f'html_baseurl = "{DOCS_URL}"' in conf


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


def test_publish_workflow_is_manual_main_only_and_derives_the_release_tag() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    trigger_match = re.search(
        r"(?ms)^on:\n(?P<body>.*?)(?=^[A-Za-z][A-Za-z0-9_-]*:)",
        workflow,
    )
    assert trigger_match is not None
    trigger = trigger_match.group("body")
    events = set(re.findall(r"(?m)^  ([A-Za-z][A-Za-z0-9_-]*):", trigger))
    assert events == {"workflow_dispatch"}
    assert "inputs:" not in trigger
    assert "github.event.release" not in workflow

    build_job = _workflow_job(workflow, "build")
    assert 'python-version: "3.13"' in build_job
    assert "refs/heads/main" in build_job
    assert "exit 1" in build_job
    assert "python scripts/check_release_tag.py --print-tag" in build_job
    assert "GITHUB_OUTPUT" in build_job
    assert "commit=$GITHUB_SHA" in build_job

    notebook_gate = build_job.index("python scripts/validate_notebooks.py")
    build = build_job.index("python -m build")
    upload = build_job.index("actions/upload-artifact")
    assert notebook_gate < build < upload


def test_publish_workflow_creates_or_reuses_a_verified_github_release() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    release_job = _workflow_job(workflow, "release")

    assert re.search(r"(?m)^\s+needs:\s*build\s*$", release_job)
    assert "contents: write" in release_job
    assert "actions/download-artifact" in release_job
    assert 'gh release view "$RELEASE_TAG"' in release_job
    assert re.search(r'gh release create\s+\\?\s*"\$RELEASE_TAG"', release_job)
    assert "RELEASE_COMMIT: ${{ needs.build.outputs.release_commit }}" in release_job
    assert "gh api" in release_job
    assert "commits/${RELEASE_TAG}" in release_job
    assert '"$existing_commit" != "$RELEASE_COMMIT"' in release_job
    assert re.search(r'gh release upload\s+\\?\s*"\$RELEASE_TAG"', release_job)
    assert "--clobber" in release_job


def test_publish_workflow_uses_a_minimal_trusted_publisher_job() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    publish_job = _workflow_job(workflow, "publish")

    assert re.search(
        r"(?m)^\s+needs:\s*(?:release|\[[^\]]*\brelease\b[^\]]*\])\s*$",
        publish_job,
    )
    assert "environment:\n      name: pypi" in publish_job
    assert "url: https://pypi.org/project/pymixef/" in publish_job
    permission_block = publish_job.split("permissions:", maxsplit=1)[1].split(
        "steps:",
        maxsplit=1,
    )[0]
    permissions = dict(
        re.findall(r"(?m)^\s+([A-Za-z][A-Za-z0-9-]*):\s*([A-Za-z]+)\s*$", permission_block)
    )
    assert permissions == {"id-token": "write"}

    assert "actions/download-artifact" in publish_job
    assert "pypa/gh-action-pypi-publish@release/v1" in publish_job
    assert "actions/checkout" not in publish_job
    assert "actions/setup-python" not in publish_job
    assert not re.search(r"(?m)^\s+run:", publish_job)
    assert "password:" not in workflow
    assert "api-token:" not in workflow
    assert workflow.count("id-token: write") == 1

    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "notebooks:\n    name: Tutorial notebooks" in ci
    assert 'python -m pip install -e ".[notebooks]"' in ci
    assert "python scripts/validate_notebooks.py" in ci


def test_release_tag_guard_accepts_only_the_static_project_version() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    command = [sys.executable, str(ROOT / "scripts/check_release_tag.py")]

    derived = subprocess.run(
        [*command, "--print-tag"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert derived.returncode == 0, derived.stderr
    assert derived.stdout.strip() == f"v{version}"

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


def test_release_tag_guard_rejects_desynchronized_release_metadata(tmp_path: Path) -> None:
    synchronized_files = (
        "pyproject.toml",
        "src/pymixef/_version.py",
        "CITATION.cff",
        "CHANGELOG.md",
        "native/CMakeLists.txt",
        "native/src/core.cpp",
        "native/tests/test_core.cpp",
        "r/pymixef/DESCRIPTION",
    )
    for relative in synchronized_files:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text(encoding="utf-8").replace(
            f"version: {version}",
            "version: 999.0.0",
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_release_tag.py"),
            "--print-tag",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "CITATION.cff is not synchronized" in result.stderr


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
