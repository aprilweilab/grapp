# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "grapp"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
]
html_theme_options = {
    "external_links": [
        {
            "url": "https://grgl.readthedocs.io/en/latest/tutorials/",
            "name": "Tutorials",
        },
        {
            "url": "https://aprilweilab.github.io",
            "name": "Wei Lab",
        }
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/aprilweilab/grapp",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/grapp",
            "icon": "fa-custom fa-pypi",
        }
    ],
}

# Nasty workaround for RTD being annoying to test with. There is probably a better
# way to do this using .readthedocs.yaml
thisdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(thisdir, ".."))
