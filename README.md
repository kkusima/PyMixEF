# PyMixEF

PyMixEF is a Python-native platform for mixed-effects statistics and
pharmacometrics. Its architecture follows one central rule: define the scientific
model once in a typed, versioned intermediate representation, then let compatible
estimation engines consume that representation.

The 0.1 series is a **foundation implementation with selected reference paths
toward the Classical Core**. It has not passed the blueprint's Foundation or
Classical Core stage gates. The current experimental scope includes:

- a safe R-like formula compiler and explicit builder API;
- immutable model IR, schema validation, deterministic model diffing, and JSON
  round trips;
- audited columnar data ingestion and stable source-row reconciliation;
- Gaussian LMM ML/REML, reference Laplace GLMM, and MMRM calculation paths;
- a component family/link and covariance system;
- canonical pharmacometric events, dosing-aware ODE integration, closed-form PK
  helpers, and typed pharmacometric declarations;
- structured convergence, simulations, residual tables, VPC data, manifests,
  cross-platform comparison, reporting, and validation bundles;
- conservative NONMEM, R-formula, PharmML, and SBML interchange subsets that
  report unsupported constructs instead of silently approximating them.

The source blueprint describes a multi-year target spanning lme4, glmmTMB, MMRM,
and NONMEM-class workflows. PyMixEF does **not** claim that its first release has
reference-validated parity with those targets. It also does not yet provide a
production FOCEI/SAEM estimator, compiled sparse core, AGHQ engine,
release-gated R interface, or full backend conformance suite. `pymixef
capabilities` reports the
evidence, open-gate, and limitation state of each path.

## Install

PyMixEF requires Python 3.11 or newer.

Published releases install from PyPI with:

```bash
python -m pip install pymixef
```

For development from a source checkout:

```bash
python -m pip install -e .
```

Tabular adapters, reports, and validation comparison dependencies are optional:

```bash
python -m pip install -e ".[data,report,validation]"
```

To run the tutorial notebooks from GitHub or the source archive, install their
Jupyter runtime:

```bash
python -m pip install "pymixef[notebooks]"
```

Before the first PyPI release, use `python -m pip install -e ".[notebooks]"`
from a source checkout. Maintainer publication prerequisites and the
Trusted Publishing procedure are documented in
[docs/publishing.md](docs/publishing.md).

The [ten tutorial notebooks](examples/notebooks/README.md) are committed with
reviewed results, multiple scientific figures, and assertion-backed checks, so
GitHub renders complete worked examples before a reader launches a kernel.

## Documentation

The [complete documentation](docs/index.md) includes:

- installation profiles and a five-minute quickstart;
- an analysis chooser and full LMM, GLMM, MMRM, PK, ODE, and pharmacometrics
  guides;
- data, formula, family, link, covariance, inference, diagnostic, simulation,
  provenance, validation, CLI, and interoperability reference material;
- deep dives for all ten pre-executed notebooks with all 31 validated plots and
  exact saved results;
- a task-oriented map plus generated signature reference for every public
  Python module;
- a searchable catalog of all 56 evidence-gated capabilities.

To build the Read the Docs site locally:

```bash
python -m pip install -e ".[docs]"
make docs
python -m http.server 8765 --bind 127.0.0.1 --directory docs/_build/dirhtml
```

The strict build fails on Sphinx warnings, stale notebook figures, missing API
docstrings/search entries, broken tutorial coverage, or figure-manifest drift.

## First linear mixed model

```python
import pymixef

data = {
    "change": [2.1, 3.2, 2.8, 4.4, 3.7, 5.1],
    "time": [0, 1, 2, 0, 1, 2],
    "subject": ["A", "A", "A", "B", "B", "B"],
}

result = pymixef.fit(
    "change ~ time + (1 | subject)",
    data=data,
    method="reml",
)

print(result.summary())
result.save("fit.json")
```

Compile and inspect a model without fitting:

```python
model = pymixef.Model.from_formula(
    "change ~ time + (1 | subject)",
    family=pymixef.families.Gaussian(),
)
plan = model.compile(data, engine="lmm", method="reml")
print(plan.explain())
```

## Pharmacometric events and ODEs

```python
from pymixef.pharmacometrics import canonicalize_events, simulate_ode

events = canonicalize_events(
    {
        "ID": [1, 1, 1],
        "TIME": [0.0, 1.0, 4.0],
        "EVID": [1, 0, 0],
        "AMT": [100.0, 0.0, 0.0],
        "CMT": [1, 1, 1],
    }
)
```

See [the documentation map](docs/index.md), [the validation policy](docs/validation.md),
[the public validation artifacts](validation/README.md), and
[the warning catalog](docs/warnings.md).

## Scientific and regulatory scope

Every likelihood path states its normalization and parameterization. Every fit
retains a convergence object, run manifest, approximation method, data/model
fingerprints, warnings, and serialized diagnostic inputs.

PyMixEF can support an organization's context-specific validation process. No
package version is universally “FDA validated,” and this project makes no such
claim.

## License

Apache License 2.0. Formal dependency and name review remains the responsibility
of distributors and adopters.
