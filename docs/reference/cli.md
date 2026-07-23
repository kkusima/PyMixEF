# Command-line interface

Installing PyMixEF creates the `pymixef` command. Use `pymixef --help` or
`pymixef COMMAND --help` for the installed-version synopsis.

## Global options

```bash
pymixef --version
pymixef --help
```

## Inspect capabilities

```bash
pymixef capabilities
pymixef capabilities --json
```

The text view is intended for people. The JSON view is the automation contract
for capability name, maturity, implemented state, evidence, limitations, and
open gates.

## Export traceability

```bash
pymixef traceability > traceability.json
```

This prints the requirement-to-implementation/specification/test/evidence
matrix as JSON.

## Explain a model

```bash
pymixef explain \
  "change ~ baseline + treatment * visit + (1 | subject)" \
  --data analysis.csv \
  --family gaussian \
  --engine lmm \
  --method reml
```

`--data` is optional for syntax/capability inspection but required for
data-dependent design details. Options:

| Argument | Meaning |
|---|---|
| `formula` | quoted PyMixEF formula |
| `--data PATH` | CSV input |
| `--family NAME` | response family |
| `--engine NAME` | backend selection |
| `--method NAME` | ML, REML, or Laplace as compatible |

CLI family names include Gaussian/normal, Bernoulli, binomial, Poisson, NB1,
NB2, gamma, and beta. Family recognition does not override backend
compatibility; an unavailable combination is refused.

## Fit a CSV model

```bash
pymixef fit \
  "change ~ baseline + treatment * visit + (1 | subject)" \
  --data analysis.csv \
  --family gaussian \
  --engine lmm \
  --method reml \
  --maxiter 1000 \
  --tolerance 1e-10 \
  --output result.json
```

Required arguments are `formula`, `--data`, and `--output`. Numerical controls
are forwarded only to a compatible engine. The output is the same portable
`FitResult` JSON contract used by `FitResult.save`.

By default a numerically suspect completed fit exits with code 4.
`--allow-warning` changes that case to code 0; it does not erase warnings from
the result.

## Create and verify a validation bundle

```bash
pymixef bundle result.json --output validation-bundle
pymixef verify-bundle validation-bundle
```

`bundle` loads an existing result and creates deterministic evidence files.
`verify-bundle` checks the manifest and internal hashes.

## Parse an NM-TRAN control stream

```bash
pymixef parse-nonmem model.ctl --output parsed-records.json
```

The command preserves the documented record subset and writes a compatibility
report. Structural `$PK`, `$DES`, and `$ERROR` code is not executed.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | command completed under its success contract |
| `2` | fit failed |
| `3` | NM-TRAN parse found unsupported constructs |
| `4` | fit returned but is numerically suspect; use result warnings/convergence for detail |

Argument parsing errors also follow the command parser’s standard nonzero
behavior.

## Shell automation

Prefer JSON output for capabilities, traceability, fitted results, and parsed
interchange reports. Do not parse human summaries when a structured contract is
available. Preserve the command, environment, input hash, exit code, and output
artifact together.

The implementation reference is
[`pymixef.cli`](../api/generated/pymixef.cli.rst).

