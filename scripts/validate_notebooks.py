#!/usr/bin/env python3
"""Validate committed notebook results and replay them in clean Jupyter kernels."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = ROOT / "examples" / "notebooks"
EXPECTED_DOMAINS = {
    **{order: "materials-catalysis" for order in range(1, 4)},
    **{order: "bio-pharma-medical" for order in range(4, 11)},
}
FORBIDDEN_CELL_PATTERNS = (
    (re.compile(r"(?m)^\s*[!%]"), "IPython shell/line magic"),
    (re.compile(r"\bget_ipython\s*\("), "direct IPython runtime access"),
)
SOURCE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_OUTPUT_TYPES = frozenset({"stream", "display_data", "execute_result"})
MINIMUM_FIGURES_PER_NOTEBOOK = 2
VALIDATION_KERNEL_NAME = "pymixef-validation"
FIGURE_TITLE_METHODS = frozenset({"set_title", "suptitle", "title"})


class NotebookValidationError(RuntimeError):
    """Raised when a tutorial notebook violates the public contract."""


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    raise NotebookValidationError("cell source must be a string or list of strings")


def _load_notebook(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NotebookValidationError(f"{path.name}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise NotebookValidationError(f"{path.name}: notebook root must be an object")
    return document


def _source_sha256(notebook: dict[str, Any]) -> str:
    """Hash ordered cell types and sources, excluding stored execution products."""

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise NotebookValidationError("notebook cells must be a list")
    source_document = {
        "nbformat": notebook.get("nbformat"),
        "nbformat_minor": notebook.get("nbformat_minor"),
        "cells": [
            {
                "cell_type": cell.get("cell_type"),
                "source": _cell_source(cell),
            }
            for cell in cells
            if isinstance(cell, dict)
        ],
    }
    payload = json.dumps(
        source_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_envelope(path: Path, notebook: dict[str, Any], order: int) -> None:
    if notebook.get("nbformat") != 4:
        raise NotebookValidationError(f"{path.name}: expected nbformat 4")
    if not isinstance(notebook.get("nbformat_minor"), int):
        raise NotebookValidationError(f"{path.name}: nbformat_minor must be an integer")

    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        raise NotebookValidationError(f"{path.name}: missing metadata object")
    kernel = metadata.get("kernelspec")
    language = metadata.get("language_info")
    tutorial = metadata.get("pymixef")
    if not isinstance(kernel, dict) or kernel.get("language") != "python":
        raise NotebookValidationError(f"{path.name}: kernelspec must select Python")
    if not isinstance(kernel.get("name"), str) or not kernel["name"]:
        raise NotebookValidationError(f"{path.name}: kernelspec must name a Python kernel")
    if not isinstance(language, dict) or language.get("name") != "python":
        raise NotebookValidationError(f"{path.name}: language_info must select Python")
    if not isinstance(tutorial, dict):
        raise NotebookValidationError(f"{path.name}: missing metadata.pymixef")
    if tutorial.get("order") != order:
        raise NotebookValidationError(f"{path.name}: metadata.pymixef.order must be {order}")
    expected_domain = EXPECTED_DOMAINS[order]
    if tutorial.get("domain") != expected_domain:
        raise NotebookValidationError(
            f"{path.name}: order {order} must use domain {expected_domain!r}"
        )

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise NotebookValidationError(f"{path.name}: cells must be a non-empty list")
    markdown_cells = [
        cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    ]
    code_cells = [
        cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]
    if len(markdown_cells) < 8:
        raise NotebookValidationError(
            f"{path.name}: exhaustive tutorials require at least 8 markdown cells"
        )
    if len(code_cells) < 6:
        raise NotebookValidationError(
            f"{path.name}: exhaustive tutorials require at least 6 code cells"
        )

    markdown_text = "\n".join(_cell_source(cell) for cell in markdown_cells)
    if len(markdown_text.split()) < 300:
        raise NotebookValidationError(
            f"{path.name}: tutorial narrative must contain at least 300 words"
        )
    lowered_markdown = markdown_text.casefold()
    for required_section in ("objectives", "exercises"):
        if required_section not in lowered_markdown:
            raise NotebookValidationError(
                f"{path.name}: missing required {required_section!r} guidance"
            )


def _validate_code(path: Path, notebook: dict[str, Any]) -> list[tuple[int, str]]:
    code_cells: list[tuple[int, str]] = []
    cells = notebook["cells"]
    for cell_number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            raise NotebookValidationError(f"{path.name}: cell {cell_number} is not an object")
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} has unknown type {cell_type!r}"
            )
        if cell_type != "code":
            continue
        source = _cell_source(cell)
        for pattern, description in FORBIDDEN_CELL_PATTERNS:
            if pattern.search(source):
                raise NotebookValidationError(
                    f"{path.name}: cell {cell_number} uses {description}; "
                    "tutorials must be pure Python"
                )
        try:
            ast.parse(source, filename=f"{path.name}:cell-{cell_number}", mode="exec")
        except SyntaxError as exc:
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} is not valid Python: {exc}"
            ) from exc
        code_cells.append((cell_number, source))
    return code_cells


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and any(_has_content(item) for item in value.values())
    return value is not None


def _figure_alt_text(source: str) -> str:
    """Build stable alternative text from literal Matplotlib figure titles."""

    tree = ast.parse(source, mode="exec")
    titles: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        if function.attr == "set":
            for keyword in node.keywords:
                if keyword.arg != "title":
                    continue
                title = keyword.value
                if not isinstance(title, ast.Constant) or not isinstance(title.value, str):
                    continue
                normalized = " ".join(title.value.split())
                if normalized:
                    titles.append((node.lineno, normalized))
            continue
        if function.attr not in FIGURE_TITLE_METHODS:
            continue
        if not node.args:
            continue
        title = node.args[0]
        if not isinstance(title, ast.Constant) or not isinstance(title.value, str):
            continue
        normalized = " ".join(title.value.split())
        if normalized:
            titles.append((node.lineno, normalized))

    ordered_titles = list(dict.fromkeys(title for _, title in sorted(titles)))
    if not ordered_titles:
        return "PyMixEF tutorial scientific figure; see the adjacent interpretation for details."
    return f"PyMixEF tutorial figure: {'; '.join(ordered_titles)}."


def _annotate_figure_outputs(notebook: dict[str, Any]) -> None:
    """Attach alternative text to every rendered image output."""

    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        alt_text = _figure_alt_text(_cell_source(cell))
        for output in outputs:
            if not isinstance(output, dict):
                continue
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            metadata = output.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                continue
            for mime_type in data:
                if not str(mime_type).startswith("image/"):
                    continue
                mime_metadata = metadata.setdefault(str(mime_type), {})
                if not isinstance(mime_metadata, dict):
                    raise NotebookValidationError(
                        f"image metadata for {mime_type!r} must be an object"
                    )
                mime_metadata["alt"] = alt_text


def _validate_output(
    path: Path,
    cell_number: int,
    output_number: int,
    output: Any,
    execution_count: int,
) -> tuple[str, ...]:
    if not isinstance(output, dict):
        raise NotebookValidationError(
            f"{path.name}: cell {cell_number} output {output_number} is not an object"
        )
    output_type = output.get("output_type")
    if output_type == "error":
        error_name = output.get("ename", "unknown error")
        raise NotebookValidationError(
            f"{path.name}: cell {cell_number} stores an error output ({error_name})"
        )
    if output_type not in SUPPORTED_OUTPUT_TYPES:
        raise NotebookValidationError(
            f"{path.name}: cell {cell_number} stores unsupported output type {output_type!r}"
        )
    metadata = output.get("metadata", {})
    if not isinstance(metadata, dict):
        raise NotebookValidationError(
            f"{path.name}: cell {cell_number} output metadata must be an object"
        )
    if output_type == "stream":
        name = output.get("name")
        if name not in {"stdout", "stderr"}:
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} stream must be stdout or stderr"
            )
        if name == "stderr":
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} stores stderr output; "
                "showcase notebooks must execute without warnings"
            )
        if not _has_content(output.get("text")):
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} contains an empty stored stream"
            )
        return ("stream", str(name))

    data = output.get("data")
    if not isinstance(data, dict) or not data or not _has_content(data):
        raise NotebookValidationError(
            f"{path.name}: cell {cell_number} stores an empty display result"
        )
    mime_types = tuple(sorted(str(name) for name in data))
    for mime_type in mime_types:
        if not mime_type.startswith("image/"):
            continue
        mime_metadata = metadata.get(mime_type)
        if not isinstance(mime_metadata, dict) or not _has_content(mime_metadata.get("alt")):
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} image output {mime_type!r} "
                "must include descriptive alternative text"
            )
    if output_type == "execute_result":
        result_count = output.get("execution_count")
        if result_count != execution_count:
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} result count {result_count!r} "
                f"does not match cell count {execution_count}"
            )
    return (str(output_type), *mime_types)


def _validate_execution_state(
    path: Path,
    notebook: dict[str, Any],
    code_cells: list[tuple[int, str]],
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    signatures: list[tuple[tuple[str, ...], ...]] = []
    cells_with_results = 0
    figure_outputs = 0
    expected_count = 1
    code_cell_numbers = {cell_number for cell_number, _ in code_cells}
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell_number not in code_cell_numbers:
            continue
        execution_count = cell.get("execution_count")
        if type(execution_count) is not int or execution_count != expected_count:
            raise NotebookValidationError(
                f"{path.name}: cell {cell_number} execution_count must be "
                f"{expected_count}, got {execution_count!r}"
            )
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            raise NotebookValidationError(f"{path.name}: cell {cell_number} outputs must be a list")
        if outputs:
            cells_with_results += 1
        figure_outputs += sum(
            1
            for output in outputs
            if isinstance(output, dict)
            and output.get("output_type") in {"display_data", "execute_result"}
            and isinstance(output.get("data"), dict)
            and any(str(mime_type).startswith("image/") for mime_type in output["data"])
        )
        signatures.append(
            tuple(
                _validate_output(
                    path,
                    cell_number,
                    output_number,
                    output,
                    execution_count,
                )
                for output_number, output in enumerate(outputs, start=1)
            )
        )
        expected_count += 1

    minimum_result_cells = max(1, (len(code_cells) + 1) // 2)
    if cells_with_results < minimum_result_cells:
        raise NotebookValidationError(
            f"{path.name}: stored results exist for only {cells_with_results} of "
            f"{len(code_cells)} code cells; expected at least {minimum_result_cells}"
        )
    if figure_outputs < MINIMUM_FIGURES_PER_NOTEBOOK:
        raise NotebookValidationError(
            f"{path.name}: expected at least {MINIMUM_FIGURES_PER_NOTEBOOK} "
            f"rendered scientific figures, found {figure_outputs}"
        )
    return tuple(signatures)


def _validate_execution_fingerprint(path: Path, notebook: dict[str, Any]) -> None:
    metadata = notebook["metadata"]["pymixef"]
    execution = metadata.get("execution")
    if not isinstance(execution, dict):
        raise NotebookValidationError(
            f"{path.name}: missing metadata.pymixef.execution source fingerprint"
        )
    if set(execution) != {"source_sha256"}:
        raise NotebookValidationError(
            f"{path.name}: execution metadata must contain only source_sha256"
        )
    recorded = execution.get("source_sha256")
    if not isinstance(recorded, str) or SOURCE_HASH_PATTERN.fullmatch(recorded) is None:
        raise NotebookValidationError(
            f"{path.name}: execution source_sha256 must be a lowercase SHA-256 digest"
        )
    current = _source_sha256(notebook)
    if recorded != current:
        raise NotebookValidationError(
            f"{path.name}: stored outputs are stale; source hash is {current}, "
            f"but execution metadata records {recorded}"
        )


def _execute_with_jupyter(
    path: Path,
    notebook: dict[str, Any],
    *,
    timeout: int,
) -> tuple[dict[str, Any], float]:
    try:
        import nbformat
        from ipykernel.kernelspec import write_kernel_spec
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        from nbclient import NotebookClient
    except ImportError as exc:
        raise NotebookValidationError(
            "Jupyter replay requires the notebook runtime; install "
            'python -m pip install -e ".[notebooks]"'
        ) from exc

    replay = nbformat.from_dict(copy.deepcopy(notebook))
    for cell in replay.cells:
        cell.source = _cell_source(cell)
        if cell.cell_type != "code":
            continue
        cell.execution_count = None
        cell.outputs = []
        cell.metadata.pop("execution", None)
    started = time.monotonic()
    try:
        with (
            TemporaryDirectory(prefix="pymixef-kernel-") as kernel_root,
            TemporaryDirectory(prefix="pymixef-matplotlib-") as matplotlib_config,
        ):
            kernels_directory = Path(kernel_root) / "kernels"
            write_kernel_spec(kernels_directory / VALIDATION_KERNEL_NAME)
            kernel_spec_manager = KernelSpecManager(
                kernel_dirs=[str(kernels_directory)],
                ensure_native_kernel=False,
            )
            kernel_manager = KernelManager(
                kernel_name=VALIDATION_KERNEL_NAME,
                kernel_spec_manager=kernel_spec_manager,
            )
            client = NotebookClient(
                replay,
                km=kernel_manager,
                timeout=timeout,
                startup_timeout=min(timeout, 120),
                kernel_name=VALIDATION_KERNEL_NAME,
                allow_errors=False,
                record_timing=False,
                coalesce_streams=True,
                extra_arguments=["--log-level=ERROR"],
            )
            kernel_environment = os.environ.copy()
            kernel_environment["MPLCONFIGDIR"] = matplotlib_config
            source_directory = str(ROOT / "src")
            inherited_python_path = kernel_environment.get("PYTHONPATH")
            kernel_environment["PYTHONPATH"] = (
                os.pathsep.join((source_directory, inherited_python_path))
                if inherited_python_path
                else source_directory
            )
            client.execute(cwd=str(ROOT), env=kernel_environment)
        nbformat.validate(replay)
    except Exception as exc:
        message = str(exc)
        if len(message) > 4_000:
            message = f"{message[:4_000]}\n... (truncated)"
        raise NotebookValidationError(
            f"{path.name}: clean Jupyter kernel replay failed with {type(exc).__name__}: {message}"
        ) from exc
    elapsed = time.monotonic() - started
    return json.loads(nbformat.writes(replay)), elapsed


def _set_execution_fingerprint(notebook: dict[str, Any]) -> None:
    notebook["metadata"]["pymixef"]["execution"] = {
        "source_sha256": _source_sha256(notebook),
    }


def _write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    try:
        import nbformat
    except ImportError as exc:  # pragma: no cover - guarded by Jupyter execution
        raise NotebookValidationError("nbformat is required to write notebooks") from exc
    document = nbformat.from_dict(notebook)
    nbformat.validate(document)
    nbformat.write(document, path)


def validate_notebooks(
    *,
    execute: bool = True,
    refresh: bool = False,
    timeout: int = 300,
) -> list[tuple[Path, int, float | None]]:
    checkpoint_directories = sorted(
        path for path in NOTEBOOK_DIRECTORY.rglob(".ipynb_checkpoints") if path.is_dir()
    )
    if checkpoint_directories:
        relative_paths = ", ".join(str(path.relative_to(ROOT)) for path in checkpoint_directories)
        raise NotebookValidationError(
            f"remove Jupyter checkpoint directories before release: {relative_paths}"
        )

    paths = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))
    nested_notebooks = sorted(
        path for path in NOTEBOOK_DIRECTORY.rglob("*.ipynb") if path.parent != NOTEBOOK_DIRECTORY
    )
    if nested_notebooks:
        relative_paths = ", ".join(str(path.relative_to(ROOT)) for path in nested_notebooks)
        raise NotebookValidationError(f"unexpected nested notebook files found: {relative_paths}")
    if len(paths) != 10:
        raise NotebookValidationError(
            f"expected exactly 10 notebooks in {NOTEBOOK_DIRECTORY}, found {len(paths)}"
        )

    results: list[tuple[Path, int, float | None]] = []
    for order, path in enumerate(paths, start=1):
        expected_prefix = f"{order:02d}_"
        if not path.name.startswith(expected_prefix):
            raise NotebookValidationError(
                f"notebook {order} must start with {expected_prefix!r}, got {path.name!r}"
            )
        notebook = _load_notebook(path)
        _validate_envelope(path, notebook, order)
        code_cells = _validate_code(path, notebook)

        if refresh:
            replay, elapsed = _execute_with_jupyter(path, notebook, timeout=timeout)
            _annotate_figure_outputs(replay)
            _set_execution_fingerprint(replay)
            _validate_execution_state(path, replay, code_cells)
            _validate_execution_fingerprint(path, replay)
            _write_notebook(path, replay)
        else:
            stored_signature = _validate_execution_state(path, notebook, code_cells)
            _validate_execution_fingerprint(path, notebook)
            elapsed = None
            if execute:
                replay, elapsed = _execute_with_jupyter(path, notebook, timeout=timeout)
                _annotate_figure_outputs(replay)
                replay_signature = _validate_execution_state(path, replay, code_cells)
                if replay_signature != stored_signature:
                    raise NotebookValidationError(
                        f"{path.name}: clean replay output structure differs from "
                        "the committed results; refresh and review the notebook"
                    )
        results.append((path, len(code_cells), elapsed))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-execute",
        action="store_true",
        help="validate committed results and source fingerprints without kernel replay",
    )
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="replay clean kernels and replace committed results/fingerprints",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-cell Jupyter execution timeout in seconds (default: 300)",
    )
    arguments = parser.parse_args(argv)
    if arguments.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        results = validate_notebooks(
            execute=not arguments.no_execute,
            refresh=arguments.refresh,
            timeout=arguments.timeout,
        )
    except NotebookValidationError as exc:
        print(f"Notebook validation failed: {exc}", file=sys.stderr)
        return 1

    for path, code_cells, elapsed in results:
        timing = "" if elapsed is None else f", kernel replay {elapsed:.2f}s"
        print(f"OK {path.relative_to(ROOT)} ({code_cells} code cells{timing})")
    if arguments.refresh:
        action = "Refreshed stored results for"
    elif arguments.no_execute:
        action = "Validated committed results for"
    else:
        action = "Validated and clean-kernel replayed"
    print(f"{action} {len(results)} PyMixEF tutorial notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
