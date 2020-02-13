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
import diffpy.structure
import numpy as np
from transforms3d.euler import euler2mat

from diffsims.libraries.vector_library import DiffractionVectorLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator

@pytest.fixture
def default_structure():
    """An atomic structure represented using diffpy
    """
    latt = diffpy.structure.lattice.Lattice(3,3,5,90,90,120)
    atom = diffpy.structure.atom.Atom(atype='Ni',xyz=[0,0,0],lattice=latt)
    hexagonal_structure = diffpy.structure.Structure(atoms=[atom],lattice=latt)
    return hexagonal_structure

@pytest.fixture
def default_simulator():
    accelerating_voltage = 300
    max_excitation_error = 1e-2
    return DiffractionGenerator(accelerating_voltage,max_excitation_error)


@pytest.fixture()
def random_eulers():
    """ Using [0,360] [0,180] and [0,360] as ranges """
    alpha = np.random.rand(100) * 360
    beta  = np.random.rand(100) * 180
    gamma = np.random.rand(100) * 360
    eulers = np.asarray((alpha,beta,gamma)).T
    return eulers

@pytest.fixture()
def random_quats():
    """ Unnormalised"""
    q_rand = np.random.random(size=(1000,4))*7
    return q_rand

@pytest.fixture()
def random_axangles():
    """ Unnormalised axes, & rotation between -pi and 2 pi """
    axangle_rand = (np.random.random(size=(1000,4)) * 3 * np.pi) - np.pi
    return axangle_rand
