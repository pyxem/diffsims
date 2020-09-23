========
diffsims
========

.. toctree::
    :hidden:

    reference.rst

.. include:: ../README.rst

Contributing
------------

diffsims is a community maintained project. We welcome contributions in the form of bug
reports, feature requests, code, documentation, and more. `These guidelines
<https://github.com/pyxem/diffsims/blob/master/CONTRIBUTING.rst>`_ describe how best to
contribute to.

Related projects
----------------

Related, open-source projects that users of orix might find useful:

- `pyxem <https://github.com/pyxem/pyxem>`_: Python library for multi-dimensional
  diffraction microscopy.
- `orix <https://github.com/pyxem/orix>`_: Python library for handling crystal
  orientation mapping data.
- `kikuchipy <https://kikuchipy.org>`_: Python library for processing and analysis of
  electron backscatter diffraction (EBSD) patterns.

Installation
------------

diffsims can be installed from `Anaconda <https://anaconda.org/conda-forge/diffsims>`_,
the `Python Package Index <https://pypi.org/project/diffsims>`_ (``pip``), or from
source, and supports Python >= 3.6.

We recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution`_::

   $ conda create --name diffsims-env
   $ conda activate diffsims-env

If you prefer a graphical interface to manage packages and environments, install the
`Anaconda distribution`_ instead.

.. _Miniconda distribution: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda distribution: https://docs.continuum.io/anaconda/

Anaconda
~~~~~~~~

Anaconda provides the easiest installation. In the Anaconda Prompt, terminal or Command
Prompt, install with::

    $ conda install diffsims --channel conda-forge

If you at a later time need to update the package::

    $ conda update diffsims

pip
~~~

To install with ``pip``, run the following in the Anaconda Prompt, terminal or Command
Prompt::

    $ pip install diffsims

If you at a later time need to update the package::

    $ pip install --upgrade diffsims

Install from source
~~~~~~~~~~~~~~~~~~~

To install diffsims from source, clone the repository from `GitHub
<https://github.com/pyxem/diffsims>`_::

    $ git clone https://github.com/pyxem/diffsims.git
    $ cd diffsims
    $ pip install --editable .

Changelog
---------

Changes with each release can be viewed on the `GitHub release page
<https://github.com/pyxem/diffsims/releases>`_.
