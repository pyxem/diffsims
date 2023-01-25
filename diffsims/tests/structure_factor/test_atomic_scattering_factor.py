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

from copy import deepcopy

from diffpy.structure import Atom, Structure, Lattice
import numpy as np
import pytest

from diffsims.structure_factor import (
    get_kinematical_atomic_scattering_factor,
    get_doyleturner_atomic_scattering_factor,
)
from diffsims.structure_factor.atomic_scattering_parameters import ELEMENTS


@pytest.mark.parametrize(
    "element, occupancy, displacement_factor, scattering_parameter, desired_factor",
    [
        ("fe", 1, 0.5, 0.17442875, 6.306987),
        ("fe", 1, 0.5, 0.90635835, 5.911219e-14),
        ("ni", 1, 0.5, 0.14190033, 10.964046),
        ("ni", 0.5, 0.5, 0.14190033, 5.482023),
    ],
)
def test_get_kinematical_atomic_scattering_factor(
    element,
    occupancy,
    displacement_factor,
    scattering_parameter,
    desired_factor,
):
    atom = Atom(atype=element, occupancy=occupancy, Uisoequiv=displacement_factor)
    factor1 = get_kinematical_atomic_scattering_factor(
        atom=atom,
        scattering_parameter=scattering_parameter,
    )
    atom.element = ELEMENTS.index(element) + 1
    factor2 = get_kinematical_atomic_scattering_factor(
        atom=atom,
        scattering_parameter=scattering_parameter,
    )

    assert np.allclose(factor1, desired_factor)
    assert np.allclose(factor2, desired_factor)


nickel = Structure(
    lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
    atoms=[Atom(xyz=[0, 0, 0], atype="ni")],
)
ferrite = Structure(
    lattice=Lattice(2.8665, 2.8665, 2.8665, 90, 90, 90),
    atoms=[Atom(xyz=[0, 0, 0], atype="fe")],
)
# Silicon Carbide 4H polytype (hexagonal, space group 186)
sic4h = Structure(
    lattice=Lattice(3.073, 3.073, 10.053, 90, 90, 120),
    atoms=[
        Atom(atype="si", xyz=[0, 0, 0]),
        Atom(atype="si", xyz=[0.33, 0.667, 0.25]),
        Atom(atype="c", xyz=[0, 0, 0.188]),
        Atom(atype="c", xyz=[0.333, 0.667, 0.438]),
    ],
)


@pytest.mark.parametrize(
    "structure, displacement_factor, scattering_parameter, desired_factor",
    [
        (ferrite, 0.5, 0.17442875, 2.422658),
        (ferrite, 0.5, 0.90635835, 9.172283e-15),
        (ferrite, 0, 0.90635835, 1.114444),
        (nickel, 0.5, 0.14190033, 2.186904),
        (sic4h, 0, 0.19894559, 1.512963),
    ],
)
def test_get_doyleturner_atomic_scattering_factor(
    structure,
    displacement_factor,
    scattering_parameter,
    desired_factor,
):
    atom = deepcopy(structure[0])
    atom.Uisoequiv = displacement_factor

    factor1 = get_doyleturner_atomic_scattering_factor(
        atom=atom,
        scattering_parameter=scattering_parameter,
        unit_cell_volume=structure.lattice.volume,
    )
    assert np.allclose(factor1, desired_factor)

    atom.element = ELEMENTS.index(atom.element) + 1
    factor2 = get_doyleturner_atomic_scattering_factor(
        atom=atom,
        scattering_parameter=scattering_parameter,
        unit_cell_volume=structure.lattice.volume,
    )
    assert np.allclose(factor1, factor2)
