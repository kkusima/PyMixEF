# Development and release

PyMixEF treats semantics, numerical evidence, documentation, and packaging as
one release surface.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Architecture
:link: architecture
:link-type: doc

Layer boundaries, shared backend payload, ModelIR, compatibility, and capability
policy.
:::

:::{grid-item-card} Contributor guide
:link: ../developer
:link-type: doc

Typing gates, extension contracts, numerical-change evidence, and schema/RFC
expectations.
:::

:::{grid-item-card} Build the documentation
:link: documentation
:link-type: doc

Install, extract notebook plots, run strict Sphinx/link checks, test search, and
preview locally.
:::

:::{grid-item-card} Publish to PyPI
:link: ../publishing
:link-type: doc

Release consistency, wheel/sdist verification, TestPyPI, Trusted Publishing,
and post-release smoke tests.
:::

:::{grid-item-card} Validation policy
:link: ../validation
:link-type: doc

Traceability, evidence bundles, context of use, and open independent-validation
work.
:::

:::{grid-item-card} Warning catalog
:link: ../warnings
:link-type: doc

Stable warning codes, mathematical meaning, remediation, and CLI exit behavior.
:::
::::

```{toctree}
:maxdepth: 1

architecture
../developer
documentation
../publishing
```
