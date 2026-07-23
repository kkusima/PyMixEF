"""SED-ML uniform-time-course subset."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._contracts import CompatibilityIssue
from .base import CompatibilityReport, InterchangeResult


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def import_sedml(path: str | Path) -> InterchangeResult[dict[str, Any]]:
    """Import uniform-time-course simulations from a SED-ML document.

    The result captures output grids and KiSAO algorithm identifiers. One-step,
    steady-state, repeated-task, and functional-range constructs are reported
    as ``unsupported``; call :meth:`InterchangeResult.require_supported` to
    reject a partial translation.
    """

    source = Path(path)
    root = ET.parse(source).getroot()
    simulations: list[dict[str, Any]] = []
    unsupported: set[str] = set()
    for element in root.iter():
        kind = _local(element.tag)
        if kind == "uniformTimeCourse":
            simulations.append(
                {
                    "id": element.attrib.get("id"),
                    "initial_time": float(element.attrib.get("initialTime", 0)),
                    "output_start_time": float(element.attrib.get("outputStartTime", 0)),
                    "output_end_time": float(element.attrib["outputEndTime"]),
                    "number_of_points": int(element.attrib["numberOfPoints"]),
                    "algorithm": next(
                        (
                            child.attrib.get("kisaoID")
                            for child in element
                            if _local(child.tag) == "algorithm"
                        ),
                        None,
                    ),
                }
            )
        elif kind in {
            "oneStep",
            "steadyState",
            "repeatedTask",
            "functionalRange",
        }:
            unsupported.add(kind)
    found = [
        CompatibilityIssue(
            "uniformTimeCourse",
            "transformed",
            f"Imported {len(simulations)} uniform time-course simulation(s).",
        )
    ]
    found.extend(
        CompatibilityIssue(
            name,
            "unsupported",
            "This SED-ML experiment construct is outside the initial subset.",
        )
        for name in sorted(unsupported)
    )
    return InterchangeResult(
        {"simulations": simulations, "source": str(source)},
        CompatibilityReport("SED-ML", "PyMixEF simulation design", tuple(found)),
    )


def export_sedml(design: Mapping[str, Any], path: str | Path) -> InterchangeResult[Path]:
    """Export one uniform-time-course design as SED-ML Level 1 Version 4.

    ``output_end_time`` and ``number_of_points`` are required. The export
    records the deterministic output grid and KiSAO identifier; richer SED-ML
    tasks and ranges are outside this function's supported subset.
    """

    namespace = "http://sed-ml.org/sed-ml/level1/version4"
    ET.register_namespace("", namespace)
    root = ET.Element(
        f"{{{namespace}}}sedML",
        {"level": "1", "version": "4"},
    )
    simulations = ET.SubElement(root, f"{{{namespace}}}listOfSimulations")
    simulation = ET.SubElement(
        simulations,
        f"{{{namespace}}}uniformTimeCourse",
        {
            "id": str(design.get("id", "simulation")),
            "initialTime": str(design.get("initial_time", 0.0)),
            "outputStartTime": str(design.get("output_start_time", 0.0)),
            "outputEndTime": str(design["output_end_time"]),
            "numberOfPoints": str(design["number_of_points"]),
        },
    )
    ET.SubElement(
        simulation,
        f"{{{namespace}}}algorithm",
        {"kisaoID": str(design.get("algorithm", "KISAO:0000019"))},
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return InterchangeResult(
        destination,
        CompatibilityReport(
            "PyMixEF simulation design",
            "SED-ML",
            (
                CompatibilityIssue(
                    "uniformTimeCourse",
                    "exact",
                    "Exported deterministic output grid and KiSAO algorithm identifier.",
                ),
            ),
        ),
    )
