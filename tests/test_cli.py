from __future__ import annotations

import json
from pathlib import Path

from pymixef.cli import main


def test_capabilities_cli_json(capsys) -> None:
    assert main(["capabilities", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert any(item["identifier"] == "ARCH-001" for item in output)


def test_traceability_cli_json(capsys) -> None:
    assert main(["traceability"]) == 0
    output = json.loads(capsys.readouterr().out)
    arch = next(item for item in output if item["requirement"] == "ARCH-001")
    assert arch["implemented"] is True
    assert arch["reproducibility"] == "bitwise"
    assert arch["source_files"] == ["src/pymixef/ir.py"]
    assert arch["specification_files"] == ["docs/concepts/model-ir.md"]


def test_explain_cli_without_data(capsys) -> None:
    assert main(["explain", "y ~ x + (1 | group)"]) == 0
    output = capsys.readouterr().out
    assert "Response: y" in output
    assert "Compatibility: valid" in output


def test_fit_bundle_and_verify_cli_round_trip(tmp_path: Path, capsys) -> None:
    data = tmp_path / "analysis.csv"
    rows = ["subject,time,y"]
    offsets = (-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7)
    noise = (-0.08, 0.03, 0.05)
    for subject, offset in enumerate(offsets, start=1):
        for time, error in enumerate(noise):
            rows.append(f"s{subject:02d},{time},{2 + 0.4 * time + offset + error}")
    data.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = tmp_path / "fit.json"
    assert (
        main(
            [
                "fit",
                "y ~ time + (1 | subject)",
                "--data",
                str(data),
                "--method",
                "ml",
                "--maxiter",
                "200",
                "--allow-warning",
                "--output",
                str(result),
            ]
        )
        == 0
    )
    assert result.is_file()
    assert result.with_suffix(".json.sha256").is_file()
    assert "PyMixEF fit (lmm, ml)" in capsys.readouterr().out

    bundle = tmp_path / "validation.zip"
    assert main(["bundle", str(result), "--output", str(bundle)]) == 0
    assert bundle.is_file()
    capsys.readouterr()

    assert main(["verify-bundle", str(bundle)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True
    assert "result.json" in verified["files"]


def test_parse_nonmem_cli_reports_supported_and_unsupported(tmp_path: Path, capsys) -> None:
    supported = tmp_path / "supported.ctl"
    supported.write_text("$INPUT ID TIME DV\n$THETA 1\n", encoding="utf-8")
    supported_output = tmp_path / "supported.json"
    assert (
        main(
            [
                "parse-nonmem",
                str(supported),
                "--output",
                str(supported_output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "supported"
    payload = json.loads(supported_output.read_text(encoding="utf-8"))
    assert payload["records"]["INPUT"] == ["ID TIME DV"]

    unsupported = tmp_path / "unsupported.ctl"
    unsupported.write_text("$PK\nCL = THETA(1)\n", encoding="utf-8")
    unsupported_output = tmp_path / "unsupported.json"
    assert (
        main(
            [
                "parse-nonmem",
                str(unsupported),
                "--output",
                str(unsupported_output),
            ]
        )
        == 3
    )
    assert capsys.readouterr().out.strip() == "unsupported constructs present"
