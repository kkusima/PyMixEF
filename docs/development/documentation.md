# Build and validate the documentation

The documentation is a Sphinx/MyST site using PyData Sphinx Theme,
`sphinx-design`, and `sphinx-copybutton`. Read the Docs is configured by
`.readthedocs.yaml`.

## Install the toolchain

```bash
python -m pip install -e ".[docs]"
```

Install `.[notebooks]` as well when notebook execution or authoring is part of
the change.

## Extract tutorial figures

Executed notebooks are the source of truth for tutorial plots:

```bash
python scripts/extract_notebook_figures.py
python scripts/extract_notebook_figures.py --check
```

The script reads exactly the ten top-level tutorial notebooks, extracts 31
embedded PNG outputs, requires image alt metadata, and writes
`docs/_static/tutorials/manifest.json` with source cell/output, dimensions, and
SHA-256. It refuses unexpected generated PNGs instead of deleting them.

## Build locally

```bash
make docs
```

Equivalent command:

```bash
sphinx-build -W --keep-going -b dirhtml docs docs/_build/dirhtml
```

`-W` turns warnings into failures. `--keep-going` reports all detectable
documentation problems in one run. The `dirhtml` builder produces clean
Read-the-Docs-style URLs.

Preview without a separate server dependency:

```bash
python -m http.server 8765 --directory docs/_build/dirhtml
```

Open `http://127.0.0.1:8765/`.

## Link validation

```bash
make docs-linkcheck
```

The link checker can require network access for external URLs. Internal
document, image, download, and anchor links should always pass offline.

## Read the Docs build

The v2 configuration uses Ubuntu 24.04, Python 3.13, the `dirhtml` builder, and
the `docs` extra. Warnings fail the hosted build. Search ranking prioritizes
getting-started, user-guide, tutorials, and pharmacometrics while demoting
development internals.
