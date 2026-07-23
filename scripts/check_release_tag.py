"""Fail when a release tag and the package's static versions disagree."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


def release_versions(root: Path) -> tuple[str, str]:
    """Return the project metadata and import-package versions."""

    with (root / "pyproject.toml").open("rb") as stream:
        project_version = str(tomllib.load(stream)["project"]["version"])
    version_source = (root / "src/pymixef/_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', version_source, flags=re.MULTILINE)
    if match is None:
        raise SystemExit("Could not read __version__ from src/pymixef/_version.py")
    return project_version, match.group(1)


def expected_release_tag(root: Path) -> tuple[str, str]:
    """Return the validated project version and its canonical release tag."""

    project_version, package_version = release_versions(root)
    if project_version != package_version:
        raise SystemExit(
            "Version mismatch: "
            f"pyproject.toml={project_version!r}, pymixef.__version__={package_version!r}"
        )
    expected_fragments = {
        "CITATION.cff": f"version: {project_version}",
        "CHANGELOG.md": f"## {project_version} -",
        "native/CMakeLists.txt": f"VERSION {project_version}",
        "native/src/core.cpp": f'return "{project_version}";',
        "native/tests/test_core.cpp": f'!= "{project_version}"',
        "r/pymixef/DESCRIPTION": f"Version: {project_version}",
    }
    for relative, fragment in expected_fragments.items():
        if fragment not in (root / relative).read_text(encoding="utf-8"):
            raise SystemExit(
                f"Version mismatch: {relative} is not synchronized to {project_version!r}"
            )
    return project_version, f"v{project_version}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a vX.Y.Z release tag against PyMixEF's static version.",
    )
    parser.add_argument("tag", nargs="?", help="Release tag, for example v0.1.1")
    parser.add_argument(
        "--print-tag",
        action="store_true",
        help="Print the canonical tag derived from the synchronized static versions",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing pyproject.toml (defaults to this checkout)",
    )
    arguments = parser.parse_args()

    project_version, expected_tag = expected_release_tag(arguments.root)
    if arguments.print_tag:
        if arguments.tag is not None:
            parser.error("tag cannot be supplied with --print-tag")
        print(expected_tag)
        return 0
    if arguments.tag is None:
        parser.error("tag is required unless --print-tag is used")
    if arguments.tag != expected_tag:
        raise SystemExit(
            f"Release tag {arguments.tag!r} does not match expected tag {expected_tag!r}"
        )
    print(f"Release tag {arguments.tag} matches PyMixEF {project_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
