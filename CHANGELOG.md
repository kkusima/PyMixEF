# Changelog

## 0.1.1 - 2026-07-23

- Added canonical PyPI project links for the live documentation, source
  repository, issue tracker, and changelog.
- Replaced repository-relative links in the PyPI long description with public
  Read the Docs and GitHub URLs.
- Removed accidentally committed coverage databases and expanded coverage
  artifact ignore rules.
- Updated installation and release documentation to reflect the live PyPI and
  Read the Docs projects.
- Reworked PyPI publishing into a manual workflow dispatched from `main`; the
  workflow now derives the release tag from package metadata, validates every
  synchronized version, creates or reuses the matching GitHub release, and
  publishes verified artifacts with OIDC Trusted Publishing.

## 0.1.0 - 2026-07-23

- Renamed the project and import namespace to PyMixEF / `pymixef`.
- Added the versioned immutable model IR, safe formula compiler, audited data
  layer, transforms, covariance catalog, and plugin protocols.
- Added dense independent-reference LMM ML/REML, Laplace GLMM, and MMRM paths.
- Added normalized family/link components and explicit unsupported-engine checks.
- Added canonical pharmacometric events, event-aware ODE simulation, closed-form
  PK helpers, and typed NLME declarations.
- Added structured convergence, diagnostic tables, simulation/VPC helpers,
  provenance manifests, non-pickle result archival, interoperability reports,
  CLI, R wrapper, validation bundles, documentation, tests, and CI.
- Added ten assertion-backed, pre-executed tutorial notebooks with committed
  results, multiple rendered scientific figures, stale-output fingerprints,
  interpreter-pinned clean-kernel replay validation, and materials/catalysis
  plus bio/pharma/medical workflows.
- Added PyPI-ready source and wheel packaging with release-tag checks and
  credential-free Trusted Publishing automation.
