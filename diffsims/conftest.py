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

from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase
import pytest

from diffsims.crystallography import ReciprocalLatticeVector
from diffsims.generators.diffraction_generator import DiffractionGenerator


@pytest.fixture
def default_structure():
    """An atomic structure represented using diffpy"""
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
    """Ni phase with space group 225 and a = 3.5236 Å."""
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
    """Ferrite phase with space group 229 and a = 2.8665 Å."""
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
    """Silicon Carbide 4H polytype phase with space group 186 and
    a = b = 3.073 Å and c = 10.053 Å.
    """
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


@pytest.fixture
def tetragonal_phase():
    """Tetragonal phase with space group 4 and a = b = 0.5 Å and
    c = 1 Å.
    """
    return Phase(
        point_group=4, structure=Structure(lattice=Lattice(0.5, 0.5, 1, 90, 90, 90))
    )


@pytest.fixture(autouse=True)
def add_reciprocal_lattice_vector_al(doctest_namespace):
    phase = Phase(
        "al",
        space_group=225,
        structure=Structure(
            lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
            atoms=[Atom("Al", [0, 0, 1])],
        ),
    )
    rlv = ReciprocalLatticeVector(phase, hkl=[[1, 1, 1], [2, 0, 0]])
    doctest_namespace["rlv"] = rlv
