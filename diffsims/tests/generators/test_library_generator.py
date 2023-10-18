# -*- coding: utf-8 -*-
# Copyright 2017-2023 The diffsims developers
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

from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.libraries.diffraction_library import DiffractionLibrary
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.crystallography import CrystalPhase





class TestDiffractionLibraryGenerator:
    def get_phase(self):
        """
        We construct an Fd-3m silicon (with lattice parameter 5.431 as a default)
        """
        a = 5.431
        latt = diffpy.structure.lattice.Lattice(a, a, a, 90, 90, 90)
        # TODO - Make this construction with internal diffpy syntax
        atom_list = []
        for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
            x, y, z = coords[0], coords[1], coords[2]
            atom_list.append(
                diffpy.structure.atom.Atom(atype="Si", xyz=[x, y, z], lattice=latt)
            )  # Motif part A
            atom_list.append(
                diffpy.structure.atom.Atom(
                    atype="Si", xyz=[x + 0.25, y + 0.25, z + 0.25], lattice=latt
                )
            )  # Motif part B

        structure = diffpy.structure.Structure(atoms=atom_list, lattice=latt)
        return CrystalPhase(structure=structure, space_group=227)

    def setup(self):
        calc = DiffractionGenerator(300.0)
        phase = self.get_phase()
        lib = StructureLibrary(names=["Si",],
                               phases=[phase,],
                               orientations=[phase.constrained_rotation(),]
                               )
        self.gen = DiffractionLibraryGenerator(electron_diffraction_calculator=calc,
                                    structure_library=lib,
                                    reciprocal_radius=1.0,
                                    calibration=0.1,
                                    half_shape=72)

    def test_setup(self):
        assert isinstance(self.gen, DiffractionLibraryGenerator)

    def test_calculate_library(self):
        lib = self.gen.calculate_library()
        assert isinstance(lib, DiffractionLibrary)