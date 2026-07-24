# <img src="https://raw.githubusercontent.com/kkusima/PyMixEF/main/docs/_static/brand/pymixef-mark.svg" alt="" width="32" /> PyMixEF

PyMixEF is a Python-native package for mixed-effects statistics and
pharmacometrics. Its architecture follows one central rule: define the scientific
model once in a typed, versioned intermediate representation, then let compatible
estimation engines consume that representation.

[Documentation](https://pymixef.readthedocs.io/en/latest/) ·
[PyPI](https://pypi.org/project/pymixef/) ·
[Source](https://github.com/kkusima/PyMixEF) ·
[Issue tracker](https://github.com/kkusima/PyMixEF/issues)

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

The [ten worked tutorials](https://pymixef.readthedocs.io/en/latest/tutorials/)
are backed by
[pre-executed Jupyter notebooks](https://github.com/kkusima/PyMixEF/tree/main/examples/notebooks)
with reviewed results, multiple scientific figures, and assertion-backed checks.
GitHub therefore renders complete worked examples before a reader launches a
kernel.

## Documentation

The [complete documentation](https://pymixef.readthedocs.io/en/latest/) includes:

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

To build and preview the documentation locally:

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

See [the documentation map](https://pymixef.readthedocs.io/en/latest/),
[the validation policy](https://pymixef.readthedocs.io/en/latest/validation/),
[the public validation artifacts](https://github.com/kkusima/PyMixEF/blob/main/validation/README.md),
and [the warning catalog](https://pymixef.readthedocs.io/en/latest/warnings/).

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
