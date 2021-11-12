# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

import mock

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# to support markdown
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath("../"))

DEPLOY = os.environ.get("READTHEDOCS") == "True"


# -- Project information -----------------------------------------------------

try:
    import keras  # noqa
    import tensorflow  # noqa
except ImportError:
    for m in []:
        sys.modules[m] = mock.Mock(name=m)

for m in ["cv2", "scipy", "scipy.stats"]:
    sys.modules[m] = mock.Mock(name=m)
sys.modules["cv2"].__version__ = "3.4"

import abctseg  # isort: skip

project = "abCTSeg"
copyright = "2020, Arjun Desai"
author = "Arjun Desai"

version = abctseg.__version__
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "1.7"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

# -- Configurations for plugins ------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

if DEPLOY:
    intersphinx_timeout = 10
else:
    # skip this when building locally
    intersphinx_timeout = 0.1
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_parsers = {".md": CommonMarkParser}

source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build", "README.md"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "abctseg2doc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "abctseg.tex",
        "abctseg Documentation",
        "abctseg contributors",
        "manual",
    )
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "abctseg", "abctseg Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "abctseg",
        "abctseg Documentation",
        author,
        "medsegpy",
        "One line description of project.",
        "Miscellaneous",
    )
]


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


_DEPRECATED_NAMES = set()


def autodoc_skip_member(app, what, name, obj, skip, options):
    # we hide something deliberately
    if getattr(obj, "__HIDE_SPHINX_DOC__", False):
        return True
    # Hide some names that are deprecated or not intended to be used
    if name in _DEPRECATED_NAMES:
        return True
    return None


def url_resolver(url):
    if ".html" not in url:
        url = url.replace("../", "")
        return (
            "https://github.com/facebookresearch/detectron2/blob/master/" + url
        )
    else:
        if DEPLOY:
            return "http://detectron2.readthedocs.io/" + url
        else:
            return "/" + url


def setup(app):
    from recommonmark.transform import AutoStructify

    app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect('autodoc-skip-member', autodoc_skip_member)
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": url_resolver,
            "auto_toc_tree_section": "Contents",
            "enable_math": True,
            "enable_inline_math": True,
            "enable_eval_rst": True,
            "enable_auto_doc_ref": True,
        },
        True,
    )
    app.add_transform(AutoStructify)
