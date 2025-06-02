# -*- coding: utf-8 -*-
# Copyright 2017-2025 The diffsims developers
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

import pytest
import numpy as np

from diffsims.generators.library_generator import (
    VectorLibraryGenerator,
    _generate_lookup_table,
)

from diffsims.libraries.vector_library import load_VectorLibrary
from diffsims.libraries.structure_library import StructureLibrary


@pytest.fixture
def get_library(default_structure):
    structure_library = StructureLibrary(
        ["Phase"], [default_structure], [[(0, 0, 0), (0, 0.2, 0)]]
    )
    vlg = VectorLibraryGenerator(structure_library)
    return vlg.get_vector_library(0.5)


def test_library_io(get_library, pickle_temp_file):
    get_library.pickle_library(pickle_temp_file)
    loaded_library = load_VectorLibrary(pickle_temp_file, safety=True)
    # we can't check that the entire libraries are the same as the memory
    # location of the 'Sim' changes
    np.testing.assert_allclose(
        get_library["Phase"]["measurements"], loaded_library["Phase"]["measurements"]
    )
    np.testing.assert_allclose(
        get_library["Phase"]["indices"], loaded_library["Phase"]["indices"]
    )


def test_unsafe_loading(get_library, pickle_temp_file):
    get_library.pickle_library(pickle_temp_file)
    with pytest.raises(RuntimeError):
        _ = load_VectorLibrary(pickle_temp_file)


def test_generate_lookup_table(default_structure):
    lattice = default_structure.lattice.reciprocal()
    _ = _generate_lookup_table(lattice, 0.5, unique=True)
    _ = _generate_lookup_table(lattice, 0.5, unique=False)
