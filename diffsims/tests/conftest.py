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

import pytest
import diffpy.structure
import numpy as np
from transforms3d.euler import euler2mat

from diffsims.libraries.vector_library import DiffractionVectorLibrary

@pytest.fixture
def default_structure():
    """An atomic structure represetned using diffpy
    """
    latt = diffpy.structure.lattice.Lattice(3,3,5,90,90,120)
    atom = diffpy.structure.atom.Atom(atype='Ni',xyz=[0,0,0],lattice=latt)
    hexagonal_structure = diffpy.structure.Structure(atoms=[atom],lattice=latt)
    return hexagonal_structure

@pytest.fixture
def vector_library():
    library = DiffractionVectorLibrary()
    library['A'] = {
        'indices': np.array([
            [[0, 2, 0], [1, 0, 0]],
            [[1, 2, 3], [0, 2, 0]],
            [[1, 2, 3], [1, 0, 0]],
        ]),
        'measurements': np.array([
            [2, 1, np.pi / 2],
            [np.sqrt(14), 2, 1.006853685],
            [np.sqrt(14), 1, 1.300246564],
        ])
    }
    lattice = diffpy.structure.Lattice(1, 1, 1, 90, 90, 90)
    library.structures = [
        diffpy.structure.Structure(lattice=lattice)
    ]
    return library
