# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "comp2comp"
copyright = "2023, StanfordMIMI"
author = "StanfordMIMI"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Adapted from https://github.com/pyvoxel/pyvoxel

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
    "sphinx.ext.githubpages",
    "m2r2",
]

autosummary_generate = True
autosummary_imported_members = True

bibtex_bibfiles = ["references.bib"]

templates_path = ["_templates"]
exclude_patterns = []


pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "Comp2Compdoc"
html_static_path = ["_static"]

intersphinx_mapping = {"numpy": ("https://numpy.org/doc/stable/", None)}
html_theme_options = {"navigation_depth": 2}

source_suffix = [".rst", ".md"]

todo_include_todos = True
napoleon_use_ivar = True
napoleon_google_docstring = True
html_show_sourcelink = False
