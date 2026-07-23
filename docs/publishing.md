# Publishing PyMixEF

PyMixEF releases are designed to use PyPI Trusted Publishing from GitHub
Actions. The workflow exchanges a short-lived OpenID Connect identity for
publish permission; it does not use a stored PyPI password or API token.

Publishing a filename and version to PyPI is irreversible. Complete every
prerequisite and obtain the intended maintainer approval before creating a
GitHub release.

## Repository configuration

The canonical source repository is
[`kkusima/PyMixEF`](https://github.com/kkusima/PyMixEF), the package is
published as [`pymixef`](https://pypi.org/project/pymixef/), and the public
documentation is hosted on
[Read the Docs](https://pymixef.readthedocs.io/en/latest/).

Before publishing a release, confirm that the project retains:

1. Maintainer recovery and security-contact arrangements.
2. A GitHub environment named `pypi`, configured with required reviewers. Apply
   branch or tag protection appropriate to the repository's governance.
3. A PyPI Trusted Publisher whose values exactly match:

   - PyPI project: `pymixef`
   - GitHub owner: `kkusima`
   - GitHub repository: `PyMixEF`
   - Workflow filename: `publish.yml`
   - Environment: `pypi`

The workflow grants `id-token: write` only to the two-step publish job. The
separate build job checks out the release tag, verifies that it matches the
package version, builds the distributions, runs strict Twine metadata checks,
and transfers only those verified artifacts to the publish job.

Official setup references:

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Adding a publisher to an existing PyPI project](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
- [PyPA GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)

## Prepare a release

Start from a reviewed, clean checkout and an isolated Python environment.

1. Update the version consistently in `pyproject.toml`,
   `src/pymixef/_version.py`, `CITATION.cff`, `CHANGELOG.md`, the native ABI,
   and the R package metadata. `tests/test_release_consistency.py` enforces the
   current set.
2. Give the changelog an accurate release date and user-visible changes.
3. If any tutorial cell source changed, run `make notebooks-refresh`, review all
   stored results in the notebook diff, and commit only intentional changes.
   Committed notebooks must have sequential execution counts, no error outputs,
   and a current source fingerprint.
4. Run the full local checks:

   ```bash
   python -m pip install -e ".[dev,notebooks]"
   make test
   make lint
   make format-check
   make typecheck
   make notebooks
   make release-check
   ```

   `make release-check` executes the tutorial notebooks, `python -m build`, and
   `python -m twine check --strict dist/*`. Notebook validation first checks the
   committed results and then independently replays every tutorial in a clean
   Jupyter kernel without rewriting it. Inspect `dist/` and confirm that the
   intended version has exactly one wheel and one source archive.
5. Check the planned tag explicitly:

   ```bash
   python scripts/check_release_tag.py v0.1.0
   ```

   Replace `v0.1.0` with `v` followed by the version in `pyproject.toml`.
6. Review the wheel and source archive contents, install the wheel in a fresh
   environment, and run an import/CLI smoke test.
7. Commit the release changes, let required CI complete, and create the protected
   `vX.Y.Z` tag.

## Publish

Create a GitHub release for the reviewed `vX.Y.Z` tag. Publishing the release
triggers `.github/workflows/publish.yml`; there is deliberately no manual
workflow-dispatch path and no token-based fallback.

The `pypi` environment approval is the final human authorization boundary.
Review the tag, version, changelog, and build-job output before approving it.
Approving the environment allows the verified wheel and source archive to be
uploaded to PyPI.

## Verify a release

After the workflow succeeds:

1. Confirm the PyPI release shows both expected distribution files and
   provenance attestations.
2. In a fresh environment, run:

   ```bash
   python -m pip install --no-cache-dir "pymixef==X.Y.Z"
   python -c "import pymixef; print(pymixef.__version__)"
   pymixef --help
   ```

3. Confirm the installed version matches the release and retain the workflow,
   artifact, and validation references required by the project governance.

If any check fails before upload, fix the release commit and create a new tag or
release candidate. Never reuse an already-published version or overwrite a
published distribution filename.
