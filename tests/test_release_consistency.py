from __future__ import annotations

from pathlib import Path

from pymixef import __version__


def test_release_versions_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_fragments = {
        "pyproject.toml": f'version = "{__version__}"',
        "CITATION.cff": f"version: {__version__}",
        "CHANGELOG.md": f"## {__version__} -",
        "native/CMakeLists.txt": f"VERSION {__version__}",
        "native/src/core.cpp": f'return "{__version__}";',
        "native/tests/test_core.cpp": f'!= "{__version__}"',
        "r/pymixef/DESCRIPTION": f"Version: {__version__}",
    }
    for relative, fragment in expected_fragments.items():
        text = (root / relative).read_text(encoding="utf-8")
        assert fragment in text, f"{relative} is not synchronized to {__version__}"
