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

from diffsims.generators.zap_map_generator import get_rotation_from_z

@pytest.fixture
def cubic():
    """An atomic structure represented using diffpy
    """
    latt = diffpy.structure.lattice.Lattice(3,3,3,90,90,90)
    atom = diffpy.structure.atom.Atom(atype='Ni',xyz=[0,0,0],lattice=latt)
    return diffpy.structure.Structure(atoms=[atom],lattice=latt)

@pytest.mark.parametrize("sample_system",[cubic])
def test_zero_rotation_cases(sample_system):
    r_test = get_rotation_from_z(sample_system,[0,0,2])
    assert r_test == (0,0,0)


@pytest.mark.parametrize("sample_system",[cubic])
class TestOrthonormals:

    def test_rotation_to_x_axis(self,sample_system):
        r_to_x = get_rotation_from_z(sample_system,[0,1,0])
        assert np.allclose(r_test,(0,90,0))

    def test_rotation_to_y_axis(self,sample_system):
        r_to_y = get_rotation_from_z(sample_system,[1,0,0])
        assert np.allclose(r_test,(90,90,0))

    def test_rotations_to_yz(self,sample_system):
        """ We rotate from z towards y, in the cubic case the angle
        will be 45, ---- """
        r_to_yz = get_rotation_from_z(sample_system,[0,1,1])
        cos_angle = np.cos(np.deg2rad(r_to_yz[2]))
        cos_lattice = sample_system.b / sample_system.c
        assert cos_angle == cos_lattice

    def test_rotation_to_111(self,sample_system):
        """ Cubic case is known Z for 45° and around X by 54.74° """
        pass

class TestHexagonal:
    """ Results are taken from """

    def test_rotation_to_streographic_corner_a(self):
        pass
    def test_rotation_to_streographic_corner_b(self):
        pass
