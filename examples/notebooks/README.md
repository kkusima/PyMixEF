# PyMixEF tutorial notebooks

These ten executable tutorials use deterministic synthetic data and are intended
to be read in numeric order. The first three cover materials and catalysis; the
last seven cover biomedical, clinical, and pharmacometric workflows. They
present reproducible, evidence-labeled workflows on synthetic data, not claims
about real products or treatments. Each notebook is committed with reviewed
execution counts and results so it is useful when viewed without first running
a kernel. Every tutorial also includes multiple rendered scientific figures
chosen to clarify the model results, diagnostics, or workflow structure.

| Order | Domain | Tutorial |
|---:|---|---|
| 1 | Materials/catalysis | [Catalyst screening with an LMM](01_catalyst_screening_lmm.ipynb) |
| 2 | Materials/catalysis | [Binary catalyst success with a GLMM](02_binary_catalyst_success_glmm.ipynb) |
| 3 | Materials/catalysis | [Catalyst deactivation with MMRM](03_catalyst_deactivation_mmrm.ipynb) |
| 4 | Bio/pharma/medical | [Multicenter biomarker LMM](04_multicenter_biomarker_lmm.ipynb) |
| 5 | Bio/pharma/medical | [Clinical-trial MMRM](05_clinical_trial_mmrm.ipynb) |
| 6 | Bio/pharma/medical | [Clustered binary-response GLMM](06_binary_response_glmm.ipynb) |
| 7 | Bio/pharma/medical | [Pharmacometrics event semantics](07_pharmacometrics_event_semantics.ipynb) |
| 8 | Bio/pharma/medical | [Closed-form PK and event-aware ODEs](08_closed_form_pk_and_ode.ipynb) |
| 9 | Bio/pharma/medical | [Pharmacometric declarations and ModelIR](09_pharmacometrics_dsl_and_model_ir.ipynb) |
| 10 | Bio/pharma/medical | [Diagnostics, simulation, validation, interchange, and archives](10_diagnostics_simulation_validation_interop_archives.ipynb) |

From a source checkout, install the package and notebook runtime with:

```bash
python -m pip install -e ".[notebooks]"
```

After a release is published, use:

```bash
python -m pip install "pymixef[notebooks]"
```

The wheel installs PyMixEF and the Jupyter runtime; obtain these tutorial files
from the GitHub repository or source archive. Open this directory in JupyterLab
and work through notebooks 1–10. All code cells are pure Python and require no
shell commands, notebook magics, R runtime, or external data files.

Maintainers can validate the notebook structure and execute every code cell with:

```bash
make notebooks
```

This checks sequential stored execution counts, rejects stored error outputs,
requires multiple rendered figures, verifies that the results match a hash of
the current cell sources, and replays each notebook in a clean Jupyter kernel.
The replay must succeed and retain the same output structure, but numerical text
is not compared byte-for-byte across platforms.

After intentionally changing notebook sources, refresh the committed results
and their source fingerprints with:

```bash
make notebooks-refresh
```

Review the complete notebook diff, then run `make notebooks` again before
committing. Refreshing is an explicit modifying operation; ordinary validation
never rewrites notebooks. The validation gates regular CI and the verified
distributions used by the PyPI publication workflow.
