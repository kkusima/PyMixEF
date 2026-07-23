"""Offline batch CLI for compilation, fitting, conversion, and validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._serialization import to_jsonable, write_json
from ._version import __version__
from .capabilities import CAPABILITIES
from .interoperability.nonmem import parse_control_stream
from .results import FitResult
from .validation import (
    create_validation_bundle,
    traceability_matrix,
    verify_validation_bundle,
)


def _value(raw: str) -> Any:
    if raw == "":
        return np.nan
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        columns: dict[str, list[Any]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(_value(row.get(name, "")))
    return {name: np.asarray(values) for name, values in columns.items()}


def _family(name: str) -> Any:
    from . import families

    normalized = name.lower().replace("_", "-")
    factories = {
        "gaussian": families.Gaussian,
        "normal": families.Gaussian,
        "bernoulli": families.Bernoulli,
        "binomial": families.Binomial,
        "poisson": families.Poisson,
        "negative-binomial-2": families.NegativeBinomial2,
        "nb2": families.NegativeBinomial2,
        "negative-binomial-1": families.NegativeBinomial1,
        "nb1": families.NegativeBinomial1,
        "gamma": families.Gamma,
        "beta": families.Beta,
    }
    try:
        return factories[normalized]()
    except KeyError as error:
        raise ValueError(
            f"Unknown CLI family {name!r}; choose from {sorted(factories)}."
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymixef",
        description="Mixed-effects statistics and pharmacometrics",
    )
    parser.add_argument("--version", action="version", version=f"PyMixEF {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    capabilities = commands.add_parser(
        "capabilities", help="Print evidence-gated capability states."
    )
    capabilities.add_argument("--json", action="store_true")

    commands.add_parser(
        "traceability",
        help="Print the machine-readable requirement-to-evidence matrix.",
    )

    explain = commands.add_parser("explain", help="Validate and explain a model.")
    explain.add_argument("formula")
    explain.add_argument("--data", type=Path)
    explain.add_argument("--family", default="gaussian")
    explain.add_argument("--engine")
    explain.add_argument("--method")

    fit = commands.add_parser("fit", help="Fit a formula model from a CSV table.")
    fit.add_argument("formula")
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--family", default="gaussian")
    fit.add_argument("--engine")
    fit.add_argument("--method")
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--maxiter", type=int, default=1000)
    fit.add_argument("--tolerance", type=float, default=1e-8)
    fit.add_argument(
        "--allow-warning",
        action="store_true",
        help="Return exit code 0 for a numerically suspect fit instead of 4.",
    )

    bundle = commands.add_parser("bundle", help="Generate a deterministic validation archive.")
    bundle.add_argument("result", type=Path)
    bundle.add_argument("--output", required=True, type=Path)

    verify = commands.add_parser(
        "verify-bundle", help="Verify a validation archive's internal hashes."
    )
    verify.add_argument("bundle", type=Path)

    convert = commands.add_parser("parse-nonmem", help="Parse a documented NM-TRAN record subset.")
    convert.add_argument("control_stream", type=Path)
    convert.add_argument("--output", required=True, type=Path)
    return parser


def _capabilities(as_json: bool) -> int:
    rows = [item.to_dict() for item in CAPABILITIES]
    if as_json:
        print(json.dumps(to_jsonable(rows), indent=2, sort_keys=True))
        return 0
    headers = ("ID", "stage", "implemented", "maturity", "capability")
    widths = {
        "ID": max(len("ID"), *(len(str(row["identifier"])) for row in rows)),
        "stage": max(len("stage"), *(len(str(row["stage"])) for row in rows)),
        "implemented": len("implemented"),
        "maturity": max(len("maturity"), *(len(str(row["maturity"])) for row in rows)),
    }
    print(
        f"{headers[0]:<{widths['ID']}}  {headers[1]:<{widths['stage']}}  "
        f"{headers[2]:<{widths['implemented']}}  "
        f"{headers[3]:<{widths['maturity']}}  {headers[4]}"
    )
    for row in rows:
        print(
            f"{row['identifier']:<{widths['ID']}}  "
            f"{row['stage']:<{widths['stage']}}  "
            f"{row['implemented']!s:<{widths['implemented']}}  "
            f"{row['maturity']:<{widths['maturity']}}  {row['name']}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    arguments = _parser().parse_args(argv)
    if arguments.command == "capabilities":
        return _capabilities(arguments.json)
    if arguments.command == "traceability":
        rows = [item.to_dict() for item in traceability_matrix()]
        print(json.dumps(to_jsonable(rows), indent=2, sort_keys=True))
        return 0
    if arguments.command in {"explain", "fit"}:
        from .model import Model

        model = Model.from_formula(arguments.formula, family=_family(arguments.family))
        if arguments.command == "explain":
            if arguments.data is None:
                print(model.explain(engine=arguments.engine, method=arguments.method))
            else:
                plan = model.compile(
                    _read_csv(arguments.data),
                    engine=arguments.engine,
                    method=arguments.method,
                )
                print(plan.explain())
            return 0
        data = _read_csv(arguments.data)
        result = model.fit(
            data,
            engine=arguments.engine,
            method=arguments.method,
            maxiter=arguments.maxiter,
            tolerance=arguments.tolerance,
        )
        result.save(arguments.output)
        print(result.summary())
        if result.convergence.status == "failed":
            return 2
        if result.convergence.status == "warning" and not arguments.allow_warning:
            return 4
        return 0
    if arguments.command == "bundle":
        result = FitResult.load(arguments.result)
        destination = create_validation_bundle(result, arguments.output)
        print(destination)
        return 0
    if arguments.command == "verify-bundle":
        verification = verify_validation_bundle(arguments.bundle)
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0
    if arguments.command == "parse-nonmem":
        translated = parse_control_stream(arguments.control_stream)
        write_json(
            arguments.output,
            {"records": translated.value, "compatibility": translated.report.to_dict()},
        )
        print("supported" if translated.report.supported else "unsupported constructs present")
        return 0 if translated.report.supported else 3
    raise AssertionError(f"Unhandled command {arguments.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
