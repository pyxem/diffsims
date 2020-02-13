# -*- coding: utf-8 -*-
# Copyright 2017-2020 The diffsims developers
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

from diffsims.libraries.structure_library import StructureLibrary


def test_from_orientations_method():
    identifiers = ['a', 'b']
    structures = [1, 2]
    orientations = [3, 4]
    library = StructureLibrary.from_orientation_lists(identifiers, structures, orientations)
    np.testing.assert_equal(library.identifiers, identifiers)
    np.testing.assert_equal(library.structures, structures)
    np.testing.assert_equal(library.orientations, orientations)
    np.testing.assert_equal(library.struct_lib['a'], (1, 3))
    np.testing.assert_equal(library.struct_lib['b'], (2, 4))


def test_from_systems_methods():
    identifiers = ['a', 'b']
    structures = [1, 2]
    systems = ['cubic', 'hexagonal']
    library = StructureLibrary.from_crystal_systems(identifiers, structures, systems, resolution=2, equal='angle')
    assert len(library.struct_lib['a'][1]) < len(library.struct_lib['b'][1])  # cubic is less area the hexagonal


@pytest.mark.parametrize('identifiers, structures, orientations', [
    (['a'], [1, 2], [3, 4]),
    (['a'], [1], [3, 4]),
])
@pytest.mark.xfail(raises=ValueError)
def test_constructor_parameter_validation_errors(identifiers, structures, orientations):
    StructureLibrary(identifiers, structures, orientations)
