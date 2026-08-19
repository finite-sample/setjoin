"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())

# Repo-specific: the API docs are dense in numpy/pandas types, so resolve
# those references rather than leaving them as plain text.
intersphinx_mapping.update(  # noqa: F821
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    }
)

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
}
