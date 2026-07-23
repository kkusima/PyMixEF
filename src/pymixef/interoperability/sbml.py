"""SBML-compatible ODE declaration subset."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._contracts import CompatibilityIssue
from .base import CompatibilityReport, InterchangeResult

_NS = "http://www.sbml.org/sbml/level3/version2/core"
_SUPPORTED_ELEMENTS = {
    "sbml",
    "model",
    "listOfCompartments",
    "compartment",
    "listOfSpecies",
    "species",
    "listOfParameters",
    "parameter",
}
_SUPPORTED_ATTRIBUTES = {
    "sbml": {"level", "version"},
    "model": {"id", "name"},
    "listOfCompartments": set(),
    "compartment": {"id", "size", "constant"},
    "listOfSpecies": set(),
    "species": {
        "id",
        "initialAmount",
        "initialConcentration",
        "compartment",
        "hasOnlySubstanceUnits",
        "boundaryCondition",
        "constant",
    },
    "listOfParameters": set(),
    "parameter": {"id", "value", "constant"},
}
_UNSUPPORTED_MESSAGES = {
    "event": (
        "General SBML event algebra is not automatically translated; encode "
        "dosing events in the canonical event table."
    ),
    "reaction": (
        "SBML reactions and kinetic laws are outside the declaration-only ODE import subset."
    ),
    "rateRule": (
        "SBML rate rules are executable equations and are not translated by the "
        "declaration-only import subset."
    ),
    "assignmentRule": (
        "SBML assignment rules are executable equations and are not translated "
        "by the declaration-only import subset."
    ),
    "algebraicRule": (
        "SBML algebraic rules are executable equations and are not translated "
        "by the declaration-only import subset."
    ),
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str | None:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return None


def _walk_with_locations(
    element: ET.Element,
    location: str | None = None,
) -> list[tuple[ET.Element, str]]:
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


def import_sbml(path: str | Path) -> InterchangeResult[dict[str, Any]]:
    """Import the declaration-only SBML Level 3 Version 2 Core subset.

    Compartments, species, and parameters are returned as plain mappings.
    Reactions, rules, events, foreign namespaces, and unrecognized content are
    retained as ``unsupported`` compatibility issues instead of being silently
    translated. Call :meth:`InterchangeResult.require_supported` to refuse
    partial imports.
    """

    source = Path(path)
    root = ET.parse(source).getroot()
    elements = _walk_with_locations(root)
    species = [
        {
            "id": element.attrib["id"],
            "initial_amount": element.attrib.get("initialAmount"),
            "initial_concentration": element.attrib.get("initialConcentration"),
            "compartment": element.attrib.get("compartment"),
            "has_only_substance_units": element.attrib.get("hasOnlySubstanceUnits"),
            "boundary_condition": element.attrib.get("boundaryCondition"),
            "constant": element.attrib.get("constant"),
        }
        for element, _ in elements
        if _local(element.tag) == "species"
        if _namespace(element.tag) == _NS
        if "id" in element.attrib
    ]
    parameters = [
        {
            "id": element.attrib["id"],
            "value": element.attrib.get("value"),
            "constant": element.attrib.get("constant", "true"),
        }
        for element, _ in elements
        if _local(element.tag) == "parameter"
        if _namespace(element.tag) == _NS
        if "id" in element.attrib
    ]
    compartments = [
        {
            "id": element.attrib["id"],
            "size": element.attrib.get("size"),
            "constant": element.attrib.get("constant", "true"),
        }
        for element, _ in elements
        if _local(element.tag) == "compartment"
        if _namespace(element.tag) == _NS
        if "id" in element.attrib
    ]
    issues = [
        CompatibilityIssue(
            "declarations",
            "transformed",
            f"Imported {len(compartments)} compartments, {len(species)} species, "
            f"and {len(parameters)} parameters.",
        )
    ]
    if _namespace(root.tag) != _NS:
        issues.append(
            CompatibilityIssue(
                "SBML namespace",
                "unsupported",
                "Only SBML Level 3 Version 2 Core is in the documented import subset.",
                "/sbml[1]",
            )
        )
    for element, location in elements:
        name = _local(element.tag)
        if name not in _SUPPORTED_ELEMENTS or _namespace(element.tag) != _NS:
            issues.append(
                CompatibilityIssue(
                    "SBML events" if name == "event" else name,
                    "unsupported",
                    _UNSUPPORTED_MESSAGES.get(
                        name,
                        "Element is outside the declaration-only SBML import subset.",
                    ),
                    location,
                )
            )
            continue
        if name in {"species", "parameter", "compartment"} and "id" not in element.attrib:
            issues.append(
                CompatibilityIssue(
                    name,
                    "unsupported",
                    "A declaration without an id cannot be represented in the PyMixEF ODE subset.",
                    location,
                )
            )
        supported_attributes = _SUPPORTED_ATTRIBUTES[name]
        issues.extend(
            CompatibilityIssue(
                f"{name}.@{attribute}",
                "unsupported",
                "Attribute is outside the declaration-only SBML import subset.",
                location,
            )
            for attribute in sorted(element.attrib)
            if attribute not in supported_attributes
        )
        if element.text and element.text.strip():
            issues.append(
                CompatibilityIssue(
                    f"{name}.text",
                    "unsupported",
                    "Text content is not represented by the declaration-only SBML import subset.",
                    location,
                )
            )
        if element.tail and element.tail.strip():
            issues.append(
                CompatibilityIssue(
                    f"{name}.tail",
                    "unsupported",
                    "Mixed text content is not represented by the declaration-only "
                    "SBML import subset.",
                    location,
                )
            )
    return InterchangeResult(
        {
            "species": species,
            "parameters": parameters,
            "compartments": compartments,
            "source": str(source),
            "sbml_attributes": dict(root.attrib),
        },
        CompatibilityReport("SBML", "PyMixEF ODE subset", tuple(issues)),
    )


def export_sbml(model: Mapping[str, Any], path: str | Path) -> InterchangeResult[Path]:
    """Export state declarations to a reviewable SBML Level 3 Version 2 file.

    States become species in one default compartment. Rate equations and model
    fields outside the declaration-only subset are reported as ``unsupported``;
    the file is still written so callers must inspect the report or call
    :meth:`InterchangeResult.require_supported` when partial export is unsafe.
    """

    ET.register_namespace("", _NS)
    root = ET.Element(f"{{{_NS}}}sbml", {"level": "3", "version": "2"})
    model_node = ET.SubElement(
        root, f"{{{_NS}}}model", {"id": str(model.get("name", "pymixef_model"))}
    )
    compartments = ET.SubElement(model_node, f"{{{_NS}}}listOfCompartments")
    ET.SubElement(
        compartments,
        f"{{{_NS}}}compartment",
        {"id": "default", "constant": "true", "size": "1"},
    )
    species_node = ET.SubElement(model_node, f"{{{_NS}}}listOfSpecies")
    issues: list[CompatibilityIssue] = []
    exported_states = 0
    for index, state in enumerate(model.get("states", ())):
        if isinstance(state, Mapping):
            raw_name = state.get("name", state.get("id"))
            if raw_name is None or not str(raw_name).strip():
                issues.append(
                    CompatibilityIssue(
                        f"states[{index}]",
                        "unsupported",
                        "An SBML species declaration requires a non-empty name.",
                    )
                )
                continue
            name = str(raw_name)
            issues.extend(
                CompatibilityIssue(
                    f"states[{index}].{key}",
                    "unsupported",
                    "State property is not serialized by the declaration-only SBML exporter.",
                )
                for key in sorted(map(str, state))
                if key not in {"name", "id"}
            )
        else:
            name = str(state)
        ET.SubElement(
            species_node,
            f"{{{_NS}}}species",
            {
                "id": name,
                "compartment": "default",
                "initialAmount": "0",
                "hasOnlySubstanceUnits": "false",
                "boundaryCondition": "false",
                "constant": "false",
            },
        )
        exported_states += 1
    issues.extend(
        CompatibilityIssue(
            str(key),
            "unsupported",
            "Top-level model construct is not serialized by the declaration-only SBML exporter.",
        )
        for key in sorted(map(str, model))
        if key not in {"name", "states"}
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return InterchangeResult(
        destination,
        CompatibilityReport(
            "PyMixEF ODE subset",
            "SBML",
            (
                CompatibilityIssue(
                    "states",
                    "exact",
                    f"Exported {exported_states} state(s) as SBML species declarations.",
                ),
                CompatibilityIssue(
                    "rate equations",
                    "unsupported",
                    "Arbitrary Python rate callables cannot be serialized as MathML.",
                ),
                *issues,
            ),
        ),
    )
