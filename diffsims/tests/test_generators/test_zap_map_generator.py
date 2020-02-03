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

@pytest.mark.parametrize("sample_system",[cubic])
def test_zero_rotation_cases(sample_system):
    r_test = get_rotation_from_z(sample_system,[0,0,2])
    assert r_test == (0,0,0)

class TestOrthonormals:

    def test_rotation_to_x_axis(self):
        r_to_x = get_rotation_from_z(sample_system,[0,1,0])
        assert r_test ==

    def test_rotation_to_y_axis(self):
        r_to_y = get_rotation_from_z(sample_system,[1,0,0])
        assert r_test

    def test_rotations_to_yz(self):
        """ We rotate from z towards y, in the cubic case the angle
        will be 45, ---- """
        r_to_yz = get_rotation_from_z(sample_system,[0,1,1])
        cos_angle = np.cos(np.deg2rad(r_to_yz[2]))
        cos_lattice = sample_system.b / sample_system.c
        assert cos_angle == cos_lattice

    def test_rotation_to_111(self):
        """ Cubic case is known Z for 45° and around X by 54.74° """

class TestHexagonal:
    """ Results are taken from """

    def test_rotation_to_streographic_corner_a():
        pass
    def test_rotation_to_streographic_corner_b():
        pass
