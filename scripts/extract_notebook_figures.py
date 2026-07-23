#!/usr/bin/env python3
"""Extract the committed notebook figures used by the documentation.

The notebooks are the source of truth.  This script deliberately reads only the
ten top-level tutorial notebooks, ignores Jupyter checkpoint directories, and
creates deterministic PNG assets plus a machine-readable manifest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "examples" / "notebooks"
OUTPUT_DIR = ROOT / "docs" / "_static" / "tutorials"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
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
EXPECTED_FIGURE_COUNT = 31


@dataclass(frozen=True)
class Figure:
    filename: str
    notebook: str
    cell: int
    output: int
    alt: str
    data: bytes


def _decode_png(value: Any) -> bytes:
    encoded = "".join(value) if isinstance(value, list) else str(value)
    return base64.b64decode(encoded, validate=True)


def _alt_text(output: dict[str, Any]) -> str:
    metadata = output.get("metadata", {})
    image_metadata = metadata.get("image/png", {})
    alt = image_metadata.get("alt", "") if isinstance(image_metadata, dict) else ""
    return str(alt).strip()


def _png_dimensions(data: bytes) -> tuple[int, int]:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature) or data[12:16] != b"IHDR":
        raise ValueError("embedded image is not a valid PNG with an IHDR chunk")
    return struct.unpack(">II", data[16:24])


def collect() -> list[Figure]:
    figures: list[Figure] = []
    actual = tuple(path.name for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")))
    if actual != EXPECTED_NOTEBOOKS:
        raise RuntimeError(
            f"expected exactly the ten documented top-level notebooks; found {actual!r}"
        )

    for notebook_name in EXPECTED_NOTEBOOKS:
        notebook_path = NOTEBOOK_DIR / notebook_name
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        figure_number = 0
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            for output_index, output in enumerate(cell.get("outputs", [])):
                data_bundle = output.get("data", {})
                if "image/png" not in data_bundle:
                    continue
                figure_number += 1
                data = _decode_png(data_bundle["image/png"])
                _png_dimensions(data)
                alt = _alt_text(output)
                if not alt:
                    raise RuntimeError(
                        f"{notebook_name}: cell {cell_index}, output "
                        f"{output_index} has no image/png alt metadata"
                    )
                filename = f"{notebook_path.stem}-figure-{figure_number}.png"
                figures.append(
                    Figure(
                        filename=filename,
                        notebook=notebook_name,
                        cell=cell_index,
                        output=output_index,
                        alt=alt,
                        data=data,
                    )
                )

    if len(figures) != EXPECTED_FIGURE_COUNT:
        raise RuntimeError(f"expected {EXPECTED_FIGURE_COUNT} figures, found {len(figures)}")
    return figures


def manifest_bytes(figures: list[Figure]) -> bytes:
    entries: list[dict[str, Any]] = []
    for figure in figures:
        width, height = _png_dimensions(figure.data)
        entries.append(
            {
                "filename": figure.filename,
                "notebook": figure.notebook,
                "cell": figure.cell,
                "output": figure.output,
                "alt": figure.alt,
                "width": width,
                "height": height,
                "sha256": hashlib.sha256(figure.data).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "source": "executed notebook image/png outputs",
        "figure_count": len(entries),
        "figures": entries,
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def unexpected_pngs(figures: list[Figure]) -> list[Path]:
    expected = {figure.filename for figure in figures}
    if not OUTPUT_DIR.exists():
        return []
    return sorted(path for path in OUTPUT_DIR.glob("*.png") if path.name not in expected)


def check(figures: list[Figure]) -> list[str]:
    problems: list[str] = []
    for figure in figures:
        target = OUTPUT_DIR / figure.filename
        if not target.exists():
            problems.append(f"missing {target.relative_to(ROOT)}")
        elif target.read_bytes() != figure.data:
            problems.append(f"stale {target.relative_to(ROOT)}")

    expected_manifest = manifest_bytes(figures)
    if not MANIFEST_PATH.exists():
        problems.append(f"missing {MANIFEST_PATH.relative_to(ROOT)}")
    elif MANIFEST_PATH.read_bytes() != expected_manifest:
        problems.append(f"stale {MANIFEST_PATH.relative_to(ROOT)}")

    for path in unexpected_pngs(figures):
        problems.append(f"unexpected generated asset {path.relative_to(ROOT)}")
    return problems


def write(figures: list[Figure]) -> None:
    extras = unexpected_pngs(figures)
    if extras:
        names = ", ".join(str(path.relative_to(ROOT)) for path in extras)
        raise RuntimeError("refusing to remove unexpected PNG assets automatically: " + names)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for figure in figures:
        (OUTPUT_DIR / figure.filename).write_bytes(figure.data)
    MANIFEST_PATH.write_bytes(manifest_bytes(figures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that extracted assets exactly match the notebooks",
    )
    args = parser.parse_args()

    try:
        figures = collect()
        if args.check:
            problems = check(figures)
            if problems:
                print("\n".join(f"ERROR: {problem}" for problem in problems))
                return 1
            print(
                f"Verified {len(figures)} documentation figures from "
                f"{len(EXPECTED_NOTEBOOKS)} executed notebooks."
            )
        else:
            write(figures)
            print(f"Extracted {len(figures)} figures to {OUTPUT_DIR.relative_to(ROOT)}.")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
