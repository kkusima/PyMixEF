"""Sphinx configuration for the PyMixEF documentation."""

from __future__ import annotations

import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
ROOT = DOCS_DIR.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pymixef import __version__  # noqa: E402


project = "PyMixEF"
author = "PyMixEF contributors"
copyright = "2026, Kenneth L. Kusima, Ph.D."
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"
exclude_patterns = [
    "_build",
    "_static/brand/README.md",
    ".DS_Store",
    "Thumbs.db",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 4
myst_url_schemes = ("http", "https", "mailto")

autosectionlabel_prefix_document = True
autosummary_generate = True
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Tutorial snippets intentionally continue a single analysis across successive
# code blocks.  Preserve imports and assignments so codeautolink can resolve
# public PyMixEF calls in later steps.
codeautolink_concat_default = True

# Generate one searchable API page per public module on every build.  Private
# implementation modules remain available through source links but are not
# presented as stable API.
apidoc_modules = [
    {
        "path": "../src/pymixef",
        "destination": "api/generated",
        "exclude_patterns": [
            "../src/pymixef/_contracts.py",
            "../src/pymixef/_serialization.py",
            "../src/pymixef/_version.py",
            "../src/pymixef/__pycache__",
            "../src/pymixef/**/__pycache__",
        ],
        "separate_modules": True,
        "module_first": True,
        "max_depth": 4,
        "automodule_options": [
            "members",
            "show-inheritance",
            "undoc-members",
        ],
    }
]

html_theme = "pydata_sphinx_theme"
html_title = f"PyMixEF {release} documentation"
html_logo = "_static/brand/pymixef-logo.svg"
html_favicon = "_static/brand/pymixef-mark.svg"
html_static_path = ["_static"]
html_css_files = ["css/pymixef.css"]
html_js_files = [
    "js/search-highlight.js",
    "js/ux.js",
]
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = True
html_permalinks_icon = "¶"
html_baseurl = "https://pymixef.readthedocs.io/en/latest/"

html_theme_options = {
    "navbar_align": "left",
    "header_links_before_dropdown": 7,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "navigation_depth": 4,
    "collapse_navigation": True,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "search_bar_text": "Search concepts, APIs, and examples…",
    "show_version_warning_banner": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

nitpicky = False
show_warning_types = True
suppress_warnings: list[str] = []


def _exclude_root_compare_collision(
    app: object,
    docname: str,
    source: list[str],
) -> None:
    """Keep the root ``compare`` export from colliding with its module name."""

    if docname != "api/generated/pymixef":
        return
    source[0] = source[0].replace(
        "   :members:\n",
        "   :members:\n   :exclude-members: compare\n",
        1,
    )


def setup(app: object) -> dict[str, bool]:
    app.connect("source-read", _exclude_root_compare_collision)  # type: ignore[attr-defined]
    return {"parallel_read_safe": True, "parallel_write_safe": True}
