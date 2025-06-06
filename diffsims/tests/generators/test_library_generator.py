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

from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.libraries.diffraction_library import DiffractionLibrary
from diffsims.libraries.structure_library import StructureLibrary


@pytest.fixture
def diffraction_calculator():
    return DiffractionGenerator(300.0)


@pytest.fixture
def library_generator(diffraction_calculator):
    return DiffractionLibraryGenerator(diffraction_calculator)


@pytest.fixture
def structure_library(default_structure):
    return StructureLibrary(["Si"], [default_structure], [[(0, 0, 0), (0.1, 0.1, 0)]])


class TestDiffractionLibraryGenerator:
    @pytest.mark.parametrize(
        "calibration, reciprocal_radius, half_shape, with_direct_beam",
        [
            (0.017, 2.4, (72, 72), False),
        ],
    )
    def test_get_diffraction_library(
        self,
        library_generator: DiffractionLibraryGenerator,
        structure_library,
        calibration,
        reciprocal_radius,
        half_shape,
        with_direct_beam,
    ):
        library = library_generator.get_diffraction_library(
            structure_library,
            calibration,
            reciprocal_radius,
            half_shape,
            with_direct_beam,
        )
        assert isinstance(library, DiffractionLibrary)
