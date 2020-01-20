# -*- coding: utf-8 -*-
# Copyright 2017-2019 The diffsims developers
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

import numpy as np

from diffsims.generators.structure_library_generator import StructureLibraryGenerator


def test_orientations_from_list():
    expected_orientations = [(0, 0, 0), (0, 90, 0)]
    structure_library_generator = StructureLibraryGenerator([
        ('a', None, [(0,0,0)]),
        ('b', None, [(0,5,5),(0,0,10)])
    ])
    structure_library = structure_library_generator.get_library()
    assert structure_library.identifiers == ['a', 'b']
    assert structure_library.structures == [None, None]
