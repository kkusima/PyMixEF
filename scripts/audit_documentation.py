#!/usr/bin/env python3
"""Audit PyMixEF's public API and tutorial documentation for completeness.

The audit intentionally uses only the Python standard library. Run it after a
Sphinx ``dirhtml`` build so it can verify the generated search index:

    python scripts/audit_documentation.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import pkgutil
import struct
import sys
import textwrap
from collections import Counter
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parents[1]
SEARCH_INDEX_PREFIX = "Search.setIndex("
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RAW_TEX_DELIMITERS = (r"\[", r"\]", r"\(", r"\)")

EXPECTED_TUTORIALS = (
    "01-catalyst-screening-lmm.md",
    "02-binary-catalyst-success-glmm.md",
    "03-catalyst-deactivation-mmrm.md",
    "04-multicenter-biomarker-lmm.md",
    "05-clinical-trial-mmrm.md",
    "06-binary-response-glmm.md",
    "07-pharmacometrics-event-semantics.md",
    "08-closed-form-pk-and-ode.md",
    "09-pharmacometrics-dsl-and-model-ir.md",
    "10-diagnostics-simulation-validation-interop-archives.md",
)

EXPECTED_NOTEBOOKS = (
    "01_catalyst_screening_lmm.ipynb",
    "02_binary_catalyst_success_glmm.ipynb",
    "03_catalyst_deactivation_mmrm.ipynb",
    "04_multicenter_biomarker_lmm.ipynb",
    "05_clinical_trial_mmrm.ipynb",
    "06_binary_response_glmm.ipynb",
    "07_pharmacometrics_event_semantics.ipynb",
    "08_closed_form_pk_and_ode.ipynb",
    "09_pharmacometrics_dsl_and_model_ir.ipynb",
    "10_diagnostics_simulation_validation_interop_archives.ipynb",
)

# These entries sample the primary root workflows and deliberate convenience
# aliases users are likely to search for. Module coverage is checked separately.
CURATED_SEARCH_ENTRIES = (
    "pymixef.Model",
    "pymixef.ModelIR",
    "pymixef.FitResult",
    "pymixef.fit",
    "pymixef.load",
    "pymixef.compare",
    "pymixef.bootstrap",
    "pymixef.approximation_sensitivity",
    "pymixef.group_influence",
    "pymixef.pattern_mixture_adjust",
    "pymixef.render_report",
    "pymixef.cov",
    "pymixef.random_streams",
    "pymixef.create_validation_bundle",
    "pymixef.verify_validation_bundle",
    "pymixef.change_impact",
    "pymixef.traceability_matrix",
    "pymixef.get_capability",
    "pymixef.iter_capabilities",
    "pymixef.diff_models",
    "pymixef.data.InputAdapter",
    "pymixef.formula.dry_run",
    "pymixef.covariance.get_covariance",
    "pymixef.families.Normal",
    "pymixef.families.NB2",
    "pymixef.backends.mmrm.estimated_marginal_means",
    "pymixef.ir.model_diff",
    "pymixef.pharmacometrics.one_compartment_bolus",
    "pymixef.pharmacometrics.saem",
)


def _display_path(path: Path, root: Path) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _is_public_module(name: str) -> bool:
    return not any(part.startswith("_") for part in name.split(".")[1:])


def _import_public_modules(
    root: Path,
    errors: list[str],
) -> tuple[tuple[str, ...], tuple[ModuleType, ...]]:
    """Discover and import every non-private module beneath ``pymixef``."""

    source = (root / "src").resolve()
    sys.path.insert(0, str(source))
    try:
        package = importlib.import_module("pymixef")
    except Exception as exc:
        errors.append(
            "could not import pymixef from "
            f"{_display_path(source, root)}: {type(exc).__name__}: {exc}"
        )
        return (), ()

    package_file = Path(getattr(package, "__file__", "")).resolve()
    if not package_file.is_relative_to(source):
        errors.append(
            "pymixef resolved outside this checkout "
            f"({package_file}); run the audit from the repository root"
        )

    names = {package.__name__}
    try:
        names.update(
            info.name
            for info in pkgutil.walk_packages(
                package.__path__,
                prefix=f"{package.__name__}.",
            )
            if _is_public_module(info.name)
        )
    except Exception as exc:
        errors.append(f"could not enumerate pymixef submodules: {type(exc).__name__}: {exc}")

    imported: list[ModuleType] = []
    for name in sorted(names):
        try:
            imported.append(importlib.import_module(name))
        except Exception as exc:
            errors.append(f"public module {name} is not importable: {type(exc).__name__}: {exc}")
    return tuple(sorted(names)), tuple(imported)


def _declared_docstring(value: object) -> str | None:
    """Return a docstring declared on a function or class, not an inherited one."""

    try:
        source = textwrap.dedent(inspect.getsource(value))
        tree = ast.parse(source)
    except (OSError, TypeError, IndentationError, SyntaxError):
        direct = getattr(value, "__doc__", None)
        if isinstance(value, type):
            direct = value.__dict__.get("__doc__")
        if not isinstance(direct, str) or not direct.strip():
            return None
        return inspect.cleandoc(direct)

    object_name = getattr(value, "__name__", None)
    declarations = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        if isinstance(node, declarations) and node.name == object_name:
            return ast.get_docstring(node, clean=True)
    return None


def _audit_public_docstrings(
    modules: Sequence[ModuleType],
    errors: list[str],
) -> int:
    """Check source-level docstrings on public module-level definitions."""

    audited = 0
    seen: set[tuple[str, int]] = set()
    for module in modules:
        for binding, value in sorted(vars(module).items()):
            if binding.startswith("_"):
                continue
            if not (inspect.isfunction(value) or inspect.isclass(value)):
                continue
            if getattr(value, "__module__", None) != module.__name__:
                continue
            identity = (module.__name__, id(value))
            if identity in seen:
                continue
            seen.add(identity)
            audited += 1
            if not _declared_docstring(value):
                errors.append(
                    f"{module.__name__}.{binding} has no declared docstring; "
                    "add a concise behavior and boundary description in source"
                )
    return audited


def _load_search_index(
    path: Path,
    root: Path,
    errors: list[str],
) -> tuple[set[str], set[str]]:
    """Load Sphinx's JavaScript-wrapped JSON search index."""

    display = _display_path(path, root)
    if not path.is_file():
        errors.append(
            f"{display} is missing; build the docs with "
            "sphinx-build -b dirhtml docs docs/_build/dirhtml"
        )
        return set(), set()
    try:
        payload = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        errors.append(f"could not read {display}: {exc}")
        return set(), set()
    if not payload.startswith(SEARCH_INDEX_PREFIX) or not payload.endswith(")"):
        errors.append(f"{display} is not a recognized Sphinx Search.setIndex payload")
        return set(), set()
    try:
        index = json.loads(payload[len(SEARCH_INDEX_PREFIX) : -1])
    except json.JSONDecodeError as exc:
        errors.append(f"{display} contains invalid search-index JSON: {exc}")
        return set(), set()

    objects = index.get("objects")
    titles = index.get("alltitles")
    if not isinstance(objects, dict) or not isinstance(titles, dict):
        errors.append(f"{display} lacks the expected objects/alltitles mappings")
        return set(), set()

    object_names: set[str] = set()
    for namespace, entries in objects.items():
        if not isinstance(namespace, str) or not isinstance(entries, list):
            errors.append(f"{display} contains a malformed objects entry")
            continue
        for entry in entries:
            if not isinstance(entry, list) or len(entry) < 5 or not isinstance(entry[4], str):
                errors.append(f"{display} contains a malformed object record")
                continue
            object_names.add(f"{namespace}.{entry[4]}" if namespace else entry[4])
    return object_names, {title for title in titles if isinstance(title, str)}


def _audit_search(
    module_names: Sequence[str],
    search_index: Path,
    root: Path,
    errors: list[str],
) -> None:
    object_names, titles = _load_search_index(search_index, root, errors)
    if not object_names and not titles:
        return

    missing_modules = sorted(set(module_names) - object_names)
    if missing_modules:
        errors.append(
            "the built search index is missing public module entries: "
            + ", ".join(missing_modules)
            + "; rebuild the API pages and Sphinx search index"
        )

    searchable = object_names | titles
    missing_entries = sorted(set(CURATED_SEARCH_ENTRIES) - searchable)
    if missing_entries:
        errors.append(
            "the built search index is missing curated API names/aliases: "
            + ", ".join(missing_entries)
            + "; expose each name through autodoc or docs/api/aliases.md and rebuild"
        )


def _load_json(path: Path, root: Path, errors: list[str]) -> Any | None:
    display = _display_path(path, root)
    if not path.is_file():
        errors.append(f"{display} is missing")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"could not load {display}: {exc}")
        return None


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError("not a valid PNG with an IHDR header")
    return struct.unpack(">II", header[16:24])


def _audit_tutorials(root: Path, errors: list[str]) -> tuple[int, int]:
    """Verify tutorial pages, source notebooks, and extracted figure evidence."""

    tutorials_dir = root / "docs" / "tutorials"
    expected_pages = set(EXPECTED_TUTORIALS)
    actual_pages = {path.name for path in tutorials_dir.glob("[0-9][0-9]-*.md")}
    if actual_pages != expected_pages:
        missing = sorted(expected_pages - actual_pages)
        extra = sorted(actual_pages - expected_pages)
        if missing:
            errors.append("missing tutorial pages: " + ", ".join(missing))
        if extra:
            errors.append("unexpected numbered tutorial pages: " + ", ".join(extra))

    notebook_dir = root / "examples" / "notebooks"
    expected_notebooks = set(EXPECTED_NOTEBOOKS)
    actual_notebooks = {path.name for path in notebook_dir.glob("*.ipynb")}
    if actual_notebooks != expected_notebooks:
        missing = sorted(expected_notebooks - actual_notebooks)
        extra = sorted(actual_notebooks - expected_notebooks)
        if missing:
            errors.append("missing source tutorial notebooks: " + ", ".join(missing))
        if extra:
            errors.append("unexpected source tutorial notebooks: " + ", ".join(extra))

    figures_dir = root / "docs" / "_static" / "tutorials"
    assets = {path.name: path for path in figures_dir.glob("*.png")}
    if len(assets) != 31:
        errors.append(
            f"{_display_path(figures_dir, root)} contains {len(assets)} PNG figures; "
            "expected exactly 31"
        )

    tutorial_text = ""
    for name in EXPECTED_TUTORIALS:
        page = tutorials_dir / name
        if page.is_file():
            try:
                tutorial_text += page.read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"could not read {_display_path(page, root)}: {exc}")

    manifest_path = figures_dir / "manifest.json"
    manifest = _load_json(manifest_path, root, errors)
    if not isinstance(manifest, dict):
        if manifest is not None:
            errors.append(f"{_display_path(manifest_path, root)} must contain a JSON object")
        return len(actual_pages), len(assets)

    entries = manifest.get("figures")
    if manifest.get("figure_count") != 31:
        errors.append("tutorial figure manifest figure_count must equal 31")
    if not isinstance(entries, list):
        errors.append("tutorial figure manifest must contain a figures list")
        return len(actual_pages), len(assets)
    if len(entries) != 31:
        errors.append(f"tutorial figure manifest has {len(entries)} entries; expected 31")

    manifest_filenames: list[str] = []
    manifest_notebooks: list[str] = []
    for position, entry in enumerate(entries, start=1):
        label = f"tutorial figure manifest entry {position}"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object")
            continue

        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            errors.append(f"{label} has no filename")
            continue
        manifest_filenames.append(filename)

        notebook = entry.get("notebook")
        if not isinstance(notebook, str) or not notebook:
            errors.append(f"{label} has no notebook name")
        else:
            manifest_notebooks.append(notebook)

        alt = entry.get("alt")
        if not isinstance(alt, str) or not alt.strip():
            errors.append(f"{label} has no useful alt text")

        asset = assets.get(filename)
        if asset is None:
            errors.append(f"{label} references missing asset {filename}")
            continue
        try:
            payload = asset.read_bytes()
            width, height = _png_dimensions(asset)
        except (OSError, ValueError) as exc:
            errors.append(f"could not validate {filename}: {exc}")
            continue

        expected_digest = entry.get("sha256")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_digest != digest:
            errors.append(
                f"{filename} SHA-256 does not match the manifest; "
                "rerun scripts/extract_notebook_figures.py"
            )
        if entry.get("width") != width or entry.get("height") != height:
            errors.append(
                f"{filename} dimensions are {width}x{height}, "
                f"not manifest value {entry.get('width')}x{entry.get('height')}"
            )
        if filename not in tutorial_text:
            errors.append(f"{filename} is not referenced by any tutorial page")

    duplicates = sorted(
        filename for filename, count in Counter(manifest_filenames).items() if count > 1
    )
    if duplicates:
        errors.append("duplicate tutorial figure manifest filenames: " + ", ".join(duplicates))

    manifest_asset_names = set(manifest_filenames)
    if manifest_asset_names != set(assets):
        missing = sorted(set(assets) - manifest_asset_names)
        unknown = sorted(manifest_asset_names - set(assets))
        if missing:
            errors.append("PNG assets absent from the figure manifest: " + ", ".join(missing))
        if unknown:
            errors.append("figure manifest entries without PNG assets: " + ", ".join(unknown))

    manifest_notebook_names = set(manifest_notebooks)
    if manifest_notebook_names != expected_notebooks:
        missing = sorted(expected_notebooks - manifest_notebook_names)
        unknown = sorted(manifest_notebook_names - expected_notebooks)
        if missing:
            errors.append("notebooks without manifest figures: " + ", ".join(missing))
        if unknown:
            errors.append("unknown notebooks in the figure manifest: " + ", ".join(unknown))

    figures_per_notebook = Counter(manifest_notebooks)
    too_few = [
        f"{name} ({figures_per_notebook[name]})"
        for name in EXPECTED_NOTEBOOKS
        if figures_per_notebook[name] < 3
    ]
    if too_few:
        errors.append(
            "every tutorial must include at least three necessary figures; too few: "
            + ", ".join(too_few)
        )

    return len(actual_pages), len(assets)


class _VisibleHTMLText(HTMLParser):
    """Collect visible text while ignoring code, scripts, and MathJax sources."""

    _IGNORED_TAGS: ClassVar[set[str]] = {
        "code",
        "pre",
        "script",
        "style",
        "textarea",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        classes = dict(attrs).get("class", "") or ""
        class_names = set(classes.split())
        if (
            self._ignored_depth
            or tag in self._IGNORED_TAGS
            or "math" in class_names
            or "MathJax" in class_names
        ):
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.text.append(data)


def _audit_math_and_code_links(root: Path, errors: list[str]) -> tuple[int, int]:
    """Reject raw TeX leakage and require linked Python examples in every tutorial."""

    docs = root / "docs"
    source_pages = tuple(path for path in docs.rglob("*.md") if "_build" not in path.parts)
    raw_source_pages = 0
    for path in source_pages:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"could not read {_display_path(path, root)}: {exc}")
            continue
        if any(delimiter in source for delimiter in RAW_TEX_DELIMITERS):
            raw_source_pages += 1
            errors.append(
                f"{_display_path(path, root)} contains raw \\\\(…\\\\) or "
                "\\\\[…\\\\] TeX delimiters; use MyST dollar math"
            )

    build = docs / "_build" / "dirhtml"
    linked_tutorials = 0
    math_tutorials = 0
    for source_name in EXPECTED_TUTORIALS:
        slug = source_name.removesuffix(".md")
        page = build / "tutorials" / slug / "index.html"
        if not page.is_file():
            errors.append(f"built tutorial page is missing: {_display_path(page, root)}")
            continue
        try:
            html = page.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"could not read {_display_path(page, root)}: {exc}")
            continue

        parser = _VisibleHTMLText()
        parser.feed(html)
        visible_text = "".join(parser.text)
        if any(delimiter in visible_text for delimiter in RAW_TEX_DELIMITERS):
            errors.append(
                f"{_display_path(page, root)} exposes raw TeX delimiters outside "
                "a rendered MathJax element"
            )

        display_math = html.count('class="math notranslate nohighlight"')
        if display_math == 0:
            errors.append(f"{_display_path(page, root)} has no rendered mathematical notation")
        else:
            math_tutorials += 1

        code_links = html.count('class="sphinx-codeautolink-a"')
        if code_links == 0:
            errors.append(
                f"{_display_path(page, root)} has no API links inside Python examples; "
                "check sphinx-codeautolink configuration"
            )
        else:
            linked_tutorials += 1

    if raw_source_pages:
        errors.append(f"{raw_source_pages} documentation source page(s) use raw TeX delimiters")
    return math_tutorials, linked_tutorials


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit public docstrings, the built Sphinx search index, tutorials, "
            "and extracted figures."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--search-index",
        type=Path,
        help="Sphinx searchindex.js path (default: docs/_build/dirhtml/searchindex.js)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    search_index = args.search_index
    if search_index is None:
        search_index = root / "docs" / "_build" / "dirhtml" / "searchindex.js"
    elif not search_index.is_absolute():
        search_index = root / search_index

    errors: list[str] = []
    module_names, modules = _import_public_modules(root, errors)
    public_objects = _audit_public_docstrings(modules, errors)
    _audit_search(module_names, search_index, root, errors)
    tutorial_count, figure_count = _audit_tutorials(root, errors)
    math_tutorials, linked_tutorials = _audit_math_and_code_links(root, errors)

    if errors:
        print(f"Documentation audit failed with {len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Documentation audit passed: "
        f"{len(module_names)} public modules, "
        f"{public_objects} documented public functions/classes, "
        f"{tutorial_count} tutorials, "
        f"{math_tutorials} tutorials with rendered math, "
        f"{linked_tutorials} tutorials with linked Python examples, "
        f"{figure_count} figures, and "
        f"{len(CURATED_SEARCH_ENTRIES)} curated search entries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
