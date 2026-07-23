"""Conservative PharmML subset import and export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._contracts import CompatibilityIssue
from .base import CompatibilityReport, InterchangeResult

_CONTAINERS = {
    "PharmML",
    "ModelDefinition",
    "ParameterModel",
    "StructuralModel",
    "ObservationModel",
    "VariabilityModel",
}
_DECLARATIONS = {
    "PopulationParameter",
    "IndividualParameter",
    "RandomVariable",
    "DerivativeVariable",
    "Variable",
    "Symbol",
}
_SUPPORTED_ELEMENTS = _CONTAINERS | _DECLARATIONS
_SUPPORTED_ATTRIBUTES = {
    "PharmML": {"writtenVersion"},
    **{name: {"symbId", "name"} for name in _DECLARATIONS},
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _walk_with_locations(
    element: ET.Element,
    location: str | None = None,
) -> list[tuple[ET.Element, str]]:
    """Return every XML element with a deterministic, occurrence-specific path."""

    name = _local(element.tag)
    current = location or f"/{name}[1]"
    found = [(element, current)]
    occurrences: dict[str, int] = {}
    for child in element:
        child_name = _local(child.tag)
        occurrences[child_name] = occurrences.get(child_name, 0) + 1
        found.extend(
            _walk_with_locations(
                child,
                f"{current}/{child_name}[{occurrences[child_name]}]",
            )
        )
    return found


def import_pharmml(path: str | Path) -> InterchangeResult[dict[str, Any]]:
    """Read symbols and structural-model metadata from a PharmML document."""

    source = Path(path)
    root = ET.parse(source).getroot()
    symbols: list[dict[str, str]] = []
    unsupported: list[CompatibilityIssue] = []
    for element, location in _walk_with_locations(root):
        name = _local(element.tag)
        symbol = (
            element.attrib.get("symbId") or element.attrib.get("name")
            if name in _DECLARATIONS
            else None
        )
        if symbol:
            symbols.append({"kind": name, "name": symbol})
        elif name in _DECLARATIONS:
            unsupported.append(
                CompatibilityIssue(
                    name,
                    "unsupported",
                    "A declaration without symbId or name cannot be represented in "
                    "the initial PyMixEF PharmML subset.",
                    location,
                )
            )

        if name not in _SUPPORTED_ELEMENTS:
            unsupported.append(
                CompatibilityIssue(
                    name,
                    "unsupported",
                    "Element is preserved by the source file but not represented in "
                    "the initial PyMixEF PharmML subset.",
                    location,
                )
            )
            continue

        supported_attributes = _SUPPORTED_ATTRIBUTES.get(name, set())
        unsupported.extend(
            CompatibilityIssue(
                f"{name}.@{attribute}",
                "unsupported",
                "Attribute is outside the declaration-only PharmML import subset.",
                location,
            )
            for attribute in sorted(element.attrib)
            if attribute not in supported_attributes
        )
        if element.text and element.text.strip():
            unsupported.append(
                CompatibilityIssue(
                    f"{name}.text",
                    "unsupported",
                    "Text content is not represented by the declaration-only "
                    "PharmML import subset.",
                    location,
                )
            )
        if element.tail and element.tail.strip():
            unsupported.append(
                CompatibilityIssue(
                    f"{name}.tail",
                    "unsupported",
                    "Mixed text content is not represented by the declaration-only "
                    "PharmML import subset.",
                    location,
                )
            )
    found = [
        CompatibilityIssue(
            "symbol declarations",
            "transformed",
            f"Imported {len(symbols)} named PharmML declarations.",
        )
    ]
    found.extend(unsupported)
    value = {
        "format": "PharmML",
        "source": str(source),
        "root_tag": _local(root.tag),
        "root_attributes": dict(root.attrib),
        "symbols": symbols,
    }
    return InterchangeResult(
        value=value,
        report=CompatibilityReport(
            source_format="PharmML",
            target_format="PyMixEF IR subset",
            issues=tuple(found),
        ),
    )


def export_pharmml(model: Mapping[str, Any], path: str | Path) -> InterchangeResult[Path]:
    """Export parameter declarations to a minimal, reviewable PharmML document."""

    root = ET.Element("PharmML", {"writtenVersion": "0.9"})
    definition = ET.SubElement(root, "ModelDefinition")
    parameter_model = ET.SubElement(definition, "ParameterModel")
    parameters = tuple(model.get("parameters", ()))
    issues: list[CompatibilityIssue] = []
    exported = 0
    for index, parameter in enumerate(parameters):
        if isinstance(parameter, Mapping):
            raw_name = parameter.get("name", parameter.get("symbId"))
            if raw_name is None or not str(raw_name).strip():
                issues.append(
                    CompatibilityIssue(
                        f"parameters[{index}]",
                        "unsupported",
                        "A PharmML parameter declaration requires a non-empty name.",
                    )
                )
                continue
            name = str(raw_name)
            issues.extend(
                CompatibilityIssue(
                    f"parameters[{index}].{key}",
                    "unsupported",
                    "Parameter property is not serialized by the declaration-only "
                    "PharmML exporter.",
                )
                for key in sorted(map(str, parameter))
                if key not in {"name", "symbId"}
            )
        else:
            name = str(parameter)
        ET.SubElement(parameter_model, "PopulationParameter", {"symbId": name})
        exported += 1
    issues.extend(
        CompatibilityIssue(
            str(key),
            "unsupported",
            "Top-level model construct is not serialized by the declaration-only PharmML exporter.",
        )
        for key in sorted(map(str, model))
        if key != "parameters"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    report = CompatibilityReport(
        source_format="PyMixEF IR subset",
        target_format="PharmML",
        issues=(
            CompatibilityIssue(
                "parameter declarations",
                "exact",
                f"Exported {exported} parameter declarations.",
            ),
            CompatibilityIssue(
                "model equations",
                "unsupported",
                "The initial exporter does not serialize arbitrary equation graphs.",
            ),
            *issues,
        ),
    )
    return InterchangeResult(destination, report)
