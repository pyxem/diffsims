# -*- coding: utf-8 -*-
# Copyright 2017-2024 The diffsims developers
#
# This file is part of diffsims.
#
# diffsims is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffsims is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffsims.  If not, see <http://www.gnu.org/licenses/>.

from itertools import chain
from setuptools import setup, find_packages

exec(open("diffsims/release_info.py").read())  # grab version info

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
# fmt: off
extra_feature_requirements = {
    "doc": [
        "numpydoc",
        "pydata-sphinx-theme",
        "sphinx                 >= 3.0.2",
        "sphinx-design",
        "sphinx_gallery",
    ],
    "tests": [
        "coverage               >= 5.0",
        "pytest                 >= 5.4",
        "pytest-cov             >= 2.8.1",
        "pytest-xdist",
    ],
}
extra_feature_requirements["dev"] = [
    "black                      >= 23.1",
    "manifix",
    "pre-commit                 >= 1.16"
] + list(chain(*list(extra_feature_requirements.values())))
# fmt: on

setup(
    name=name,
    version=version,
    description="Diffraction Simulations in Python",
    author=author,
    author_email=email,
    license=license,
    url="https://github.com/pyxem/diffsims",
    long_description=open("README.rst").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        (
            "License :: OSI Approved :: GNU General Public License v3 or later "
            "(GPLv3+)"
        ),
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    extras_require=extra_feature_requirements,
    install_requires=[
        "diffpy.structure   >= 3.0.2",
        "matplotlib         >= 3.3",
        "numba",
        "numpy              >= 1.17.3",
        "orix               >= 0.12.1",
        "psutil",
        "scipy              >= 1.8",
        "tqdm               >= 4.9",
        "transforms3d",
    ],
    python_requires=">=3.6",
    package_data={
        "": ["LICENSE", "README.rst", "readthedocs.yaml"],
        "diffsims": ["*.py"],
    },
)
