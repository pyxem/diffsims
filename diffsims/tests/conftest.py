# -*- coding: utf-8 -*-
# Copyright 2017-2021 The diffsims developers
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
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase

from diffsims.generators.diffraction_generator import DiffractionGenerator


@pytest.fixture
def default_structure():
    """An atomic structure represented using diffpy """
    latt = Lattice(3, 3, 5, 90, 90, 120)
    atom = Atom(atype="Ni", xyz=[0, 0, 0], lattice=latt)
    hexagonal_structure = Structure(atoms=[atom], lattice=latt)
    return hexagonal_structure


@pytest.fixture
def default_simulator():
    accelerating_voltage = 300
    return DiffractionGenerator(accelerating_voltage)


@pytest.fixture
def nickel_phase():
    return Phase(
        name="nickel",
        space_group=225,
        structure=Structure(
            lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
            atoms=[Atom(xyz=[0, 0, 0], atype="Ni", Uisoequiv=0.006332)],
        ),
    )


@pytest.fixture
def ferrite_phase():
    return Phase(
        name="ferrite",
        space_group=229,
        structure=Structure(
            lattice=Lattice(2.8665, 2.8665, 2.8665, 90, 90, 90),
            atoms=[
                Atom(xyz=[0, 0, 0], atype="Fe", Uisoequiv=0.006332),
                Atom(xyz=[0.5, 0.5, 0.5], atype="Fe", Uisoequiv=0.006332),
            ],
        ),
    )


@pytest.fixture
def silicon_carbide_phase():
    """Silicon Carbide 4H polytype (hexagonal, space group 186)."""
    return Phase(
        space_group=186,
        structure=Structure(
            lattice=Lattice(3.073, 3.073, 10.053, 90, 90, 120),
            atoms=[
                Atom(atype="Si", xyz=[0, 0, 0]),
                Atom(atype="Si", xyz=[0.33, 0.667, 0.25]),
                Atom(atype="C", xyz=[0, 0, 0.188]),
                Atom(atype="C", xyz=[0.333, 0.667, 0.438]),
            ],
        ),
    )
