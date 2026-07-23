# Installation

PyMixEF supports CPython 3.11, 3.12, and 3.13. The core installation depends
only on NumPy and SciPy.

## Install from PyPI

Create or activate a virtual environment, then install the published package:

```bash
python -m pip install --upgrade pip
python -m pip install pymixef
```

Confirm which interpreter and package version you are using:

```bash
python -c "import sys, pymixef; print(sys.executable); print(pymixef.__version__)"
pymixef --version
```

The distribution name and import name are both lowercase `pymixef`; the project
and documentation use the display name **PyMixEF**.

## Choose optional features

Extras keep the numerical core small while making complete workflows easy to
install.

| Install command | Adds | Choose it when |
|---|---|---|
| `pip install pymixef` | NumPy, SciPy, CLI, core models | Your inputs are mappings or NumPy-compatible arrays |
| `pip install "pymixef[data]"` | pandas, Polars, PyArrow, xarray adapters | You exchange data through dataframe or labeled-array ecosystems |
| `pip install "pymixef[plot]"` | Matplotlib | You will create plots in scripts |
| `pip install "pymixef[notebooks]"` | JupyterLab, kernel, notebook validation, Matplotlib | You will run the ten tutorials |
| `pip install "pymixef[report]"` | Markdown, PDF, and Word report dependencies | You will call `render_report` beyond plain HTML |
| `pip install "pymixef[validation]"` | pandas and statsmodels | You will run comparison/validation workflows |
| `pip install "pymixef[docs]"` | Sphinx documentation toolchain | You will build this documentation |
| `pip install "pymixef[dev]"` | tests, build, typing, lint, release tools | You will contribute to the package |

Extras can be combined:

```bash
python -m pip install "pymixef[data,notebooks,report,validation]"
```

## Install from a source checkout

An editable install reflects local source changes immediately:

```bash
git clone https://github.com/kkusima/PyMixEF.git
cd PyMixEF
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,notebooks,docs]"
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,notebooks,docs]"
```

## Verify the installation

This smoke test exercises import, formula compilation, the LMM backend, and the
structured convergence result:

```python
import pymixef

data = {
    "y": [1.0, 1.4, 2.0, 1.2, 1.8, 2.3],
    "time": [0, 1, 2, 0, 1, 2],
    "subject": ["A", "A", "A", "B", "B", "B"],
}

result = pymixef.fit("y ~ time + (1 | subject)", data, method="reml")
print(result.summary())
assert result.convergence.trustworthy
```

## Offline behavior and packaged resources

The library itself is fully offline and emits no telemetry. The wheel includes
the `py.typed` marker, the versioned ModelIR JSON schema, and the stable warning
catalog. Documentation links and package installation naturally require network
access unless you use local copies or a package mirror.

## Troubleshooting

**`No matching distribution found`**
: Confirm that the active interpreter is CPython 3.11–3.13 and update `pip`.
  If a package index or mirror has not synchronized yet, install from the
  canonical source checkout.

**A dataframe type is not recognized**
: Install the `data` extra and verify that the dataframe library is available in
  the same environment as PyMixEF.

**PDF or Word report export fails**
: Install the `report` extra. Markdown and HTML have a smaller dependency path.

**A notebook kernel cannot import PyMixEF**
: Install `pymixef[notebooks]` into that kernel’s environment, then select that
  environment from Jupyter’s kernel menu.

**A method is unavailable even though installation succeeded**
: Installation and capability are separate. Query `pymixef capabilities` or
  consult the [analysis matrix](../reference/analysis-matrix.md); unsupported
  methods are refused explicitly rather than silently replaced.

## Next

Continue to the [five-minute quickstart](quickstart.md), or go directly to
[choosing an analysis](choose-analysis.md).
