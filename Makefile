PYTHON ?= python

.PHONY: test lint format-check typecheck typecheck-core typecheck-all notebooks notebooks-refresh docs docs-assets docs-linkcheck build release-check benchmark

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests benchmarks examples scripts

format-check:
	$(PYTHON) -m ruff format --check src tests benchmarks examples scripts

typecheck: typecheck-core

typecheck-core:
	$(PYTHON) -m mypy

typecheck-all:
	$(PYTHON) -m mypy src/pymixef

notebooks:
	$(PYTHON) scripts/validate_notebooks.py

notebooks-refresh:
	$(PYTHON) scripts/validate_notebooks.py --refresh

docs-assets:
	$(PYTHON) scripts/extract_notebook_figures.py

docs:
	$(PYTHON) scripts/extract_notebook_figures.py --check
	$(PYTHON) -m sphinx -W --keep-going -b dirhtml docs docs/_build/dirhtml
	$(PYTHON) scripts/audit_documentation.py

docs-linkcheck:
	$(PYTHON) -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck

build:
	$(PYTHON) -m build

release-check: notebooks
	$(PYTHON) -m build
	$(PYTHON) -m twine check --strict dist/*

benchmark:
	$(PYTHON) benchmarks/run.py --output benchmark-results.json
