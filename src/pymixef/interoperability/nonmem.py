"""NONMEM-style data and NM-TRAN control-stream subset import."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .._contracts import CompatibilityIssue
from .base import CompatibilityReport, InterchangeResult

_RECORD = re.compile(r"^\s*\$(?P<name>[A-Z][A-Z0-9_-]*)(?P<body>.*)$", re.IGNORECASE)
_KNOWN = {
    "INPUT",
    "DATA",
    "SUBROUTINES",
    "MODEL",
    "PK",
    "DES",
    "ERROR",
    "THETA",
    "OMEGA",
    "SIGMA",
    "ESTIMATION",
    "SIMULATION",
    "TABLE",
}
_STRUCTURAL = {"PK", "DES", "ERROR"}


def _read_text(value: str | Path) -> str:
    candidate = Path(value)
    if "\n" not in str(value) and candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return str(value)


def parse_control_stream(value: str | Path) -> InterchangeResult[dict[str, list[str]]]:
    """Parse records and preserve a documented NM-TRAN subset.

    Structural code is retained verbatim but marked unsupported for automatic
    compilation.  This is deliberate refusal rather than an unsafe approximation.
    """

    text = _read_text(value)
    records: dict[str, list[str]] = {}
    current: str | None = None
    found: list[CompatibilityIssue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _RECORD.match(line)
        if match:
            current = match.group("name").upper()
            records.setdefault(current, []).append(match.group("body").strip())
            if current not in _KNOWN:
                found.append(
                    CompatibilityIssue(
                        f"${current}",
                        "unsupported",
                        "This NM-TRAN record is not in the documented import subset.",
                        f"line {line_number}",
                    )
                )
            elif current in _STRUCTURAL:
                found.append(
                    CompatibilityIssue(
                        f"${current}",
                        "unsupported",
                        "Structural NM-TRAN statements are preserved for review but "
                        "are not executed or silently translated.",
                        f"line {line_number}",
                    )
                )
            else:
                found.append(
                    CompatibilityIssue(
                        f"${current}",
                        "transformed",
                        "Record parsed into a normalized list while preserving text.",
                        f"line {line_number}",
                    )
                )
        elif current is not None:
            records[current].append(line.rstrip())
        elif line.strip() and not line.lstrip().startswith(";"):
            found.append(
                CompatibilityIssue(
                    "preamble",
                    "unsupported",
                    "Non-comment text before the first NM-TRAN record.",
                    f"line {line_number}",
                )
            )
    return InterchangeResult(
        value=records,
        report=CompatibilityReport(
            source_format="NM-TRAN",
            target_format="PyMixEF record subset",
            issues=tuple(found),
            metadata={"records": sorted(records)},
        ),
    )


def import_nonmem_data(
    data: Mapping[str, Any],
    *,
    column_mapping: Mapping[str, str] | None = None,
) -> InterchangeResult[dict[str, np.ndarray]]:
    """Normalize NONMEM-style columns; event canonicalization remains separate."""

    mapping = {
        str(source): str(target).upper() for source, target in (column_mapping or {}).items()
    }
    output: dict[str, np.ndarray] = {}
    found: list[CompatibilityIssue] = []
    for source, values in data.items():
        target = mapping.get(str(source), str(source).upper())
        if target in output:
            found.append(
                CompatibilityIssue(
                    target,
                    "unsupported",
                    f"Multiple input columns map to {target!r}.",
                )
            )
            continue
        output[target] = np.asarray(values)
        found.append(
            CompatibilityIssue(
                str(source),
                "exact" if target == str(source) else "transformed",
                f"Mapped to canonical column {target!r}.",
            )
        )
    required = {"ID", "TIME"}
    for missing in sorted(required - output.keys()):
        found.append(
            CompatibilityIssue(
                missing,
                "unsupported",
                "Required event key is absent.",
            )
        )
    return InterchangeResult(
        value=output,
        report=CompatibilityReport(
            source_format="NONMEM-style table",
            target_format="PyMixEF column mapping",
            issues=tuple(found),
        ),
    )


def import_nonmem_table(path: str | Path) -> InterchangeResult[dict[str, np.ndarray]]:
    """Import a comma- or whitespace-delimited NONMEM output table.

    Repeated ``TABLE NO.`` preambles are removed explicitly and reported. The
    importer requires one header row and refuses ragged records.
    """

    source = Path(path)
    lines = [
        line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    removed = sum(line.upper().startswith("TABLE NO.") for line in lines)
    lines = [line for line in lines if not line.upper().startswith("TABLE NO.")]
    if not lines:
        raise ValueError("NONMEM output table is empty.")
    comma = "," in lines[0]
    rows = list(csv.reader(lines)) if comma else [re.split(r"\s+", line.strip()) for line in lines]
    header = [str(name) for name in rows[0]]
    if len(set(header)) != len(header):
        raise ValueError("NONMEM output table has duplicate column names.")
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError("NONMEM output table contains ragged records.")
    output: dict[str, np.ndarray] = {}
    for index, name in enumerate(header):
        raw = [row[index] for row in rows[1:]]
        try:
            output[name] = np.asarray(raw, dtype=float)
        except ValueError:
            output[name] = np.asarray(raw, dtype=str)
    found = [
        CompatibilityIssue(
            "table columns",
            "exact",
            f"Imported {len(header)} columns and {len(rows) - 1} rows.",
        )
    ]
    if removed:
        found.append(
            CompatibilityIssue(
                "TABLE NO. preamble",
                "transformed",
                f"Removed {removed} NONMEM table preamble line(s).",
            )
        )
    return InterchangeResult(
        output,
        CompatibilityReport(
            source_format="NONMEM output table",
            target_format="PyMixEF column mapping",
            issues=tuple(found),
            metadata={"source": str(source)},
        ),
    )
