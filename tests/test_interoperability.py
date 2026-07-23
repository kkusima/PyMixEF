from __future__ import annotations

from pathlib import Path

from pymixef.interoperability.nonmem import (
    import_nonmem_data,
    import_nonmem_table,
    parse_control_stream,
)
from pymixef.interoperability.pharmml import import_pharmml
from pymixef.interoperability.r import translate_r_formula
from pymixef.interoperability.sbml import export_sbml, import_sbml
from pymixef.interoperability.sedml import export_sedml, import_sedml


def test_r_formula_refuses_function_evaluation() -> None:
    exact = translate_r_formula("y ~ x + (1 | subject)")
    assert exact.report.supported
    unsafe = translate_r_formula("y ~ poly(x, 2) + (1 | subject)")
    assert not unsafe.report.supported


def test_control_stream_preserves_and_refuses_structural_code() -> None:
    result = parse_control_stream(
        "$INPUT ID TIME DV\n$THETA 1\n$PK\nCL = THETA(1)\n$ESTIMATION METHOD=1"
    )
    assert "THETA" in result.value
    assert not result.report.supported
    assert result.report.by_status("unsupported")[0].construct == "$PK"


def test_nonmem_data_requires_keys() -> None:
    result = import_nonmem_data({"id": [1], "time": [0], "dv": [2.0]})
    assert result.report.supported
    assert "ID" in result.value


def test_sbml_declaration_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "model.xml"
    exported = export_sbml({"name": "pk", "states": ["central"]}, target)
    assert target.exists()
    imported = import_sbml(target)
    assert imported.value["species"][0]["id"] == "central"
    assert not exported.report.supported  # arbitrary rates are explicitly absent


def test_pharmml_unsupported_leaf_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "model.xml"
    source.write_text(
        """\
<PharmML writtenVersion="0.9">
  <ModelDefinition>
    <ParameterModel>
      <PopulationParameter symbId="CL" />
      <UnsupportedLeaf value="not-translated" />
    </ParameterModel>
  </ModelDefinition>
</PharmML>
""",
        encoding="utf-8",
    )

    result = import_pharmml(source)

    unsupported = result.report.by_status("unsupported")
    leaf = next(issue for issue in unsupported if issue.construct == "UnsupportedLeaf")
    assert leaf.source_location == (
        "/PharmML[1]/ModelDefinition[1]/ParameterModel[1]/UnsupportedLeaf[1]"
    )
    assert not result.report.supported
    assert result.value["symbols"] == [{"kind": "PopulationParameter", "name": "CL"}]


def test_sbml_reaction_and_rate_rule_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "dynamic.xml"
    source.write_text(
        """\
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core"
      xmlns:math="http://www.w3.org/1998/Math/MathML"
      level="3" version="2">
  <model id="dynamic">
    <listOfCompartments>
      <compartment id="default" size="1" constant="true" />
    </listOfCompartments>
    <listOfSpecies>
      <species id="central" compartment="default"
               initialAmount="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false" />
    </listOfSpecies>
    <listOfRules>
      <rateRule variable="central">
        <math:math><math:cn>1</math:cn></math:math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="input" reversible="false" />
    </listOfReactions>
  </model>
</sbml>
""",
        encoding="utf-8",
    )

    result = import_sbml(source)

    unsupported = result.report.by_status("unsupported")
    constructs = {issue.construct for issue in unsupported}
    assert {"reaction", "rateRule"}.issubset(constructs)
    assert all(
        issue.source_location
        for issue in unsupported
        if issue.construct in {"reaction", "rateRule"}
    )
    assert not result.report.supported
    assert result.value["species"][0]["id"] == "central"


def test_nonmem_table_and_sedml_round_trips(tmp_path: Path) -> None:
    table = tmp_path / "sdtab.csv"
    table.write_text("ID,TIME,DV\n1,0,2.5\n1,1,2.0\n", encoding="utf-8")
    imported_table = import_nonmem_table(table)
    assert imported_table.report.supported
    assert imported_table.value["DV"].tolist() == [2.5, 2.0]

    sedml = tmp_path / "simulation.xml"
    export_sedml(
        {"output_end_time": 24.0, "number_of_points": 24},
        sedml,
    ).require_supported()
    imported_design = import_sedml(sedml).require_supported()
    assert imported_design["simulations"][0]["number_of_points"] == 24
