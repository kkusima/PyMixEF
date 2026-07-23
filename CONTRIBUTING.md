# Contributing

Install the development environment with:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

Contributions should include:

- a linked blueprint requirement or an RFC for new semantics;
- mathematical parameterization and likelihood-constant documentation;
- unit, property, derivative, integration, and pathology tests as applicable;
- a maturity/limitation update in the capability registry;
- change-impact classification and benchmark evidence for numerical changes.

Never copy code from proprietary software. Preserve compatible upstream licenses
and attribution.

