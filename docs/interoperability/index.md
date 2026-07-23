# Interoperability

PyMixEF translates only constructs whose meaning it can describe. Every import
or export returns an `InterchangeResult` containing both the translated `value`
and a `CompatibilityReport`.

## Compatibility statuses

| Status | Meaning |
|---|---|
| `exact` | represented without a known semantic change in the documented subset |
| `transformed` | normalized into an equivalent PyMixEF form, with the change reported |
| `approximated` | represented by an explicit approximation and rationale |
| `unsupported` | preserved or identified for review but not silently executed/translated |

```python
translated = pymixef.interoperability.translate_r_formula(
    "response ~ treatment * time + (1 | subject)"
)

print(translated.value)
print(translated.report.supported)
for issue in translated.report.issues:
    print(issue.to_dict())

translated.require_supported()
```

Call `require_supported()` before consuming the value when a workflow must
refuse any unsupported construct.

## Format matrix

| Format | Direction | Initial supported subset |
|---|---|---|
| NONMEM-style data | import | column normalization, required ID/TIME check |
| NONMEM output table | import | comma/whitespace table with preamble handling |
| NM-TRAN control stream | import/inspect | record preservation; structural code explicitly unsupported |
| R/lme4 formula | translate | shared fixed/random operators; arbitrary R evaluation refused |
| PharmML | import/export | named declarations / parameter declarations |
| SBML L3V2 Core | import/export | compartments, species, parameters, states |
| SED-ML L1V4 | import/export | uniform time-course grid and KiSAO algorithm |

These are conservative subsets, not full implementations of their external
standards.

## NONMEM

`import_nonmem_data` uppercases or explicitly maps columns and reports duplicate
targets or missing `ID`/`TIME`. Canonical event processing remains a separate,
auditable step.

`import_nonmem_table` removes repeated `TABLE NO.` preambles, requires one unique
header row, refuses ragged records, and parses numeric columns where possible.

`parse_control_stream` preserves NM-TRAN records. `$INPUT`, `$DATA`,
`$SUBROUTINES`, `$MODEL`, `$THETA`, `$OMEGA`, `$SIGMA`, `$ESTIMATION`,
`$SIMULATION`, and `$TABLE` are normalized for inspection. `$PK`, `$DES`, and
`$ERROR` text is retained but marked unsupported because arbitrary structural
code is not executed.

## R formulas

`translate_r_formula` accepts shared fixed effects, interactions, nesting, and
`|`/`||` random operators. R environment access and function evaluation,
including `::`, `$`, `[[`, `I(`, `poly(`, `ns(`, and `bs(`, are reported as
unsupported; create transformed columns explicitly.

The repository also contains an alpha reticulate wrapper. Its release gate and
distribution status are described in
[interoperability and refusal policy](../migration/interoperability.md).

## PharmML

Import recognizes named population/individual parameters, random variables,
derivative variables, variables, and symbols inside the initial container
subset. Unknown elements, attributes, and mixed text receive occurrence-specific
locations in the compatibility report.

Export writes minimal parameter declarations and reports model equations and
unserialized properties as unsupported.

## SBML

Import targets SBML Level 3 Version 2 Core declarations for compartments,
species, and parameters. Reactions, kinetic laws, rate/assignment/algebraic
rules, and general SBML events are not executed.

Export writes a reviewable declaration document containing a default
compartment, state/species declarations, and parameters. Dosing belongs in the
canonical event table.

## SED-ML

Import/export covers a Level 1 Version 4 `uniformTimeCourse`: initial time,
output start/end, number of points, and KiSAO algorithm identifier. One-step,
steady-state, repeated-task, and functional-range experiments are outside the
initial subset.

## Comparing numerical results

Interchange asks whether model meaning can be represented. `pymixef.compare`
asks whether fitted numerical output agrees with an explicitly mapped reference.
Comparison requires coefficient mapping and compatible objective conventions; it
does not infer whether another tool includes the same constants or uses the same
conditional/marginal definition.

## API reference

- [`pymixef.interoperability`](../api/generated/pymixef.interoperability.rst)
- [`pymixef.interoperability.base`](../api/generated/pymixef.interoperability.base.rst)
- [`pymixef.interoperability.nonmem`](../api/generated/pymixef.interoperability.nonmem.rst)
- [`pymixef.interoperability.r`](../api/generated/pymixef.interoperability.r.rst)
- [`pymixef.interoperability.pharmml`](../api/generated/pymixef.interoperability.pharmml.rst)
- [`pymixef.interoperability.sbml`](../api/generated/pymixef.interoperability.sbml.rst)
- [`pymixef.interoperability.sedml`](../api/generated/pymixef.interoperability.sedml.rst)

```{toctree}
:maxdepth: 1

../migration/interoperability
```

