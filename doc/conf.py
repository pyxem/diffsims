# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime
import inspect
import os
from os.path import relpath, dirname
import re
import sys

from diffsims import __author__, __file__, __version__

# -- Project information -----------------------------------------------------

project = "diffsims"
copyright = f"2018-{str(datetime.now().year)}, {__author__}"
author = __author__

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

# Create links to references within diffsims' documentation to these packages
intersphinx_mapping = {
    "diffpy.structure": ("https://www.diffpy.org/diffpy.structure", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "orix": ("https://orix.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML theming: pydata-sphinx-theme
# https://pydata-sphinx-theme.readthedocs.io
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/pyxem/diffsims",
    "header_links_before_dropdown": 6,
    "logo": {"alt_text": project, "text": project},
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "pyxem",
    "github_repo": "diffsims",
    "github_version": "main",
    "doc_path": "doc",
}
html_static_path = ["_static"]

# Logo
html_logo = "_static/pyxem_logo.png"
html_favicon = "_static/pyxem_logo.png"

# Syntax highlighting
pygments_style = "friendly"


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.

    This is taken from SciPy's Sphinx configuration file.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(__file__), ".."))
    fn = relpath(fn, start=startdir).replace(os.path.sep, "/")

    if fn.startswith("diffsims/"):
        m = re.match(r"^.*dev0\+([a-f\d]+)$", __version__)
        pre_link = "https://github.com/pyxem/diffsims/blob/"
        if m:
            return pre_link + "%s/%s%s" % (m.group(1), fn, linespec)
        elif "dev" in __version__:
            return pre_link + "main/%s%s" % (fn, linespec)
        else:
            return pre_link + "v%s/%s%s" % (__version__, fn, linespec)
    else:
        return None


# sphinx.ext.autodoc
# ------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"
autodoc_default_options = {
    "show-inheritance": True,
}

# -- Sphinx-Gallery---------------
# https://sphinx-gallery.github.io
sphinx_gallery_conf = {
    "backreferences_dir": "reference/generated",
    "doc_module": ("diffsims",),
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    "filename_pattern": "^((?!sgskip).)*$",  # pattern to define which will be executed
    "ignore_pattern": "_sgskip.py",  # pattern to define which will not be executed
    "reference_url": {"diffsims": None},
}
