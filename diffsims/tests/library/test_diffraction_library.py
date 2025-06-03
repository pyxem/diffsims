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

import numpy as np
import pytest

from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.libraries.diffraction_library import load_DiffractionLibrary
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.sims.diffraction_simulation import DiffractionSimulation


@pytest.fixture
def get_library(default_structure):
    diffraction_calculator = DiffractionGenerator(300.0)
    dfl = DiffractionLibraryGenerator(diffraction_calculator)
    structure_library = StructureLibrary(
        ["Phase"], [default_structure], [np.array([(0, 0, 0), (0, 0.2, 0)])]
    )

    return dfl.get_diffraction_library(structure_library, 0.017, 2.4, (72, 72))


def test_get_library_entry_assertionless(get_library):
    assert isinstance(get_library.get_library_entry()["Sim"], DiffractionSimulation)
    assert isinstance(
        get_library.get_library_entry(phase="Phase")["Sim"], DiffractionSimulation
    )
    assert isinstance(
        get_library.get_library_entry(phase="Phase", angle=(0, 0, 0))["Sim"],
        DiffractionSimulation,
    )


def test_get_library_small_offset(get_library):
    alpha = get_library.get_library_entry(phase="Phase", angle=(0, 0, 0))["intensities"]
    beta = get_library.get_library_entry(phase="Phase", angle=(1e-8, 0, 0))[
        "intensities"
    ]
    assert np.allclose(alpha, beta)


def test_library_io(get_library, pickle_temp_file):
    get_library.pickle_library(pickle_temp_file)
    loaded_library = load_DiffractionLibrary(pickle_temp_file, safety=True)
    # We can't check that the entire libraries are the same as the memory
    # location of the 'Sim' changes
    for i in range(len(get_library["Phase"]["orientations"])):
        np.testing.assert_allclose(
            get_library["Phase"]["orientations"][i],
            loaded_library["Phase"]["orientations"][i],
        )
        np.testing.assert_allclose(
            get_library["Phase"]["intensities"][i],
            loaded_library["Phase"]["intensities"][i],
        )
        np.testing.assert_allclose(
            get_library["Phase"]["pixel_coords"][i],
            loaded_library["Phase"]["pixel_coords"][i],
        )


def test_angle_but_no_phase(get_library):
    # we have given an angle but no phase
    with pytest.raises(ValueError, match="To select a certain angle you must first "):
        assert isinstance(
            get_library.get_library_entry(angle=(0, 0, 0))["Sim"], DiffractionSimulation
        )


def test_unknown_library_entry(get_library):
    # The angle we have asked for is not in the library
    with pytest.raises(ValueError, match="It appears that no library entry lies"):
        assert isinstance(
            get_library.get_library_entry(phase="Phase", angle=(1e-1, 0, 0))["Sim"],
            DiffractionSimulation,
        )


def test_unsafe_loading(get_library, pickle_temp_file):
    with pytest.raises(RuntimeError, match="Unpickling is risky, turn safety to True "):
        get_library.pickle_library(pickle_temp_file)
        _ = load_DiffractionLibrary(pickle_temp_file)
