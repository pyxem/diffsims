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
import os
import numpy as np
import diffpy.structure

from diffsims.generators.library_generator import VectorLibraryGenerator

from diffsims.libraries.structure_library import StructureLibrary


@pytest.fixture
def get_library(default_structure):
    structure_library = StructureLibrary(['Phase'], [default_structure], [[(0, 0, 0), (0, 0.2, 0)]])
    vlg = VectorLibraryGenerator(structure_library)
    return vlg.get_vector_library(0.5)
