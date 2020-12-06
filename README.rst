|build_status|_ |Coveralls|_ |docs|_ |pypi_version|_ |black|_ |doi|_

.. |build_status| image:: https://github.com/pyxem/diffsims/workflows/build/badge.svg
.. _build_status: https://github.com/pyxem/diffsims/actions

.. |Coveralls| image:: https://coveralls.io/repos/github/pyxem/diffsims/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/pyxem/diffsims?branch=master

.. |docs| image:: https://readthedocs.org/projects/diffsims/badge/?version=latest
.. _docs: https://diffsims.readthedocs.io/en/latest

.. |pypi_version| image:: http://img.shields.io/pypi/v/diffsims.svg?style=flat
.. _pypi_version: https://pypi.python.org/pypi/diffsims

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _black: https://github.com/psf/black

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3337900.svg
.. _doi: https://doi.org/10.5281/zenodo.3337900

diffsims is an open-source python library for simulating diffraction.

If simulations performed using diffsims form a part of published work please cite the DOI at the top of this page.

diffsims is released under the GPL v3 license.


Installation
------------

diffsims requires python 3 and conda - we suggest using the python 3 version of `Miniconda <https://conda.io/miniconda.html>`__ and creating a new environment for diffsims using the following commands in the anaconda prompt:::

      $ conda create -n diffsims
      $ conda activate diffsims

The recommended way to install diffsims is then from conda-forge using:::

      $ conda install -c conda-forge diffsims

Note that diffsims is also available via pip:::

      $ pip install diffsims
