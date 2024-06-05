============
Installation
============

diffsims can be installed from `Anaconda <https://anaconda.org/conda-forge/diffsims>`_, the
`Python Package Index <https://pypi.org/project/diffsims>`_ (``pip``), or from source,
and supports Python >= 3.8.


With pip
========

diffsims is availabe from the Python Package Index (PyPI), and can therefore be installed
with `pip <https://pip.pypa.io/en/stable>`__. To install, run the following::

    pip install diffsims

To update diffsims to the latest release::

    pip install --upgrade diffsims

To install a specific version of diffsims (say version 0.5.1)::

    pip install diffsims==0.5.1


With Anaconda
=============

To install with Anaconda, we recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`__.
To create an environment and activate it, run the following::

   conda create --name diffsims-env python=3.10
   conda activate diffsims-env

If you prefer a graphical interface to manage packages and environments, you can install
the `Anaconda distribution <https://docs.continuum.io/anaconda>`__ instead.

To install::

    conda install diffsims --channel conda-forge

To update diffsims to the latest release::

    conda update diffsims

To install a specific version of diffsims (say version 0.5.1)::

    conda install diffsims==0.5.1 -c conda-forge


.. _install-from-source:

From source
===========

The source code is hosted on `GitHub <https://github.com/pyxem/diffsims>`__. One way to
install diffsims from source is to clone the repository from `GitHub
<https://github.com/pyxem/diffsims>`__, and install with ``pip``::

    git clone https://github.com/pyxem/diffsims.git
    cd diffsims
    pip install --editable .

The source can also be downloaded as tarballs or zip archives via links like
``https://github.com/pyxem/diffsims/archive/v<major.minor.patch>/
diffsims-<major.minor.patch>.tar.gz``, where the version ``<major.minor.patch>`` can be
e.g. ``0.5.1``, and ``tar.gz`` can be exchanged with ``zip``.

.. _https://github.com/pyxem/diffsims/archive/v<major.minor.patch>/diffsims-<major.minor.patch>.tar.gz: https://github.com/pyxem/diffsims/archive/v<major.minor.patch>/diffsims-<major.minor.patch>.tar.gz