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
import numpy as np

from diffsims.utils.rotation_conversion_utils import AxAngle,Euler, vectorised_axangle_to_correct_range, convert_axangle_to_correct_range
from diffsims.utils.fundemental_zone_utils import get_proper_point_group_string, reduce_to_fundemental_zone,
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz, vectorised_qmult, \
                                          _create_advanced_linearly_spaced_array_in_rzxz
                                          
from transforms3d.quaternions import qmult

""" These tests check that AxAngle and Euler behave in good ways """

class TestAxAngle:
    @pytest.fixture()
    def good_array(self):
        return np.asarray([[1, 0, 0, 1],
                           [0, 1, 0, 1.1]])

    def test_good_array__init__(self, good_array):
        assert isinstance(AxAngle(good_array), AxAngle)

    def test_remove_large_rotations(self, good_array):
        axang = AxAngle(good_array)
        axang.remove_large_rotations(1.05)  # removes 1 rotations
        assert axang.data.shape == (1, 4)

    @pytest.mark.xfail(raises=ValueError, strict=True)
    class TestCorruptingData:
        @pytest.fixture()
        def axang(self, good_array):
            return AxAngle(good_array)

        def test_bad_shape(self, axang):
            axang.data = axang.data[:, :2]
            axang._check_data()

        def test_dumb_angle(self, axang):
            axang.data[0, 3] = -0.5
            axang._check_data()

        def test_denormalised(self, axang):
            axang.data[:, 0] = 3
            axang._check_data()


class TestEuler:
    @pytest.fixture()
    def good_array(self):
        return np.asarray([[32, 80, 21],
                           [40, 10, 11]])

    def test_good_array__init__(self, good_array):
        assert isinstance(Euler(good_array), Euler)

    @pytest.mark.xfail(raises=ValueError, strict=True)
    class TestCorruptingData:
        @pytest.fixture()
        def euler(self, good_array):
            return Euler(good_array)

        def test_bad_shape(self, euler):
            euler.data = euler.data[:, :2]
            euler._check_data()

        def test_dumb_angle(self, euler):
            euler.data[0, 0] = 700
            euler._check_data()


def test_interconversion_euler_axangle():
    """
    This function checks (with random numbers) that .to_Axangle() and .to_Euler()
    go back and forth correctly
    """
    axes = np.random.random_sample((1000, 3))
    axes = np.divide(axes, np.linalg.norm(axes, axis=1).reshape(1000, 1))
    assert np.allclose(np.linalg.norm(axes, axis=1), 1)  # check for input normalisation
    angles = np.multiply(np.random.random_sample((1000, 1)), np.pi)
    axangle = AxAngle(np.concatenate((axes, angles), axis=1))
    transform = AxAngle(np.concatenate((axes, angles), axis=1))
    e = transform.to_Euler(axis_convention='rzxz')
    transform_back = e.to_AxAngle()
    assert isinstance(transform_back, AxAngle)
    assert np.allclose(transform_back.data, axangle.data)

""" These these test vectorizations """

@pytest.fixture()
def random_quats():
    q_rand = np.random.random(size=(1000,4))*7
    return q_rand

def test_qmult_vectorisation(random_quats):
    q1 = np.asarray([2,3,4,5])
    fast = vectorised_qmult(q1,random_quats)

    stored_quat = np.ones_like(random_quats)
    for i, row in enumerate(random_quats):
        stored_quat[i] = qmult(q1, row)

    assert np.allclose(fast,stored_quat)

@pytest.fixture()
def random_axangles():
    random_axangles = np.random.random(size=(1000,4))*2*np.pi
    return random_axangles

def test_convert_to_correct_range_vectorisation(random_axangles):
    fast = vectorised_axangle_to_correct_range(random_axangles)
    stored_axangles = np.empty_like(random_axangles)
    for i,row in enumerate(random_axangles):
        temp_vect, temp_angle = convert_axangle_to_correct_range(row[:3],row[3])
        for j in [0, 1, 2]:
            stored_axangles[i, j] = temp_vect[j]
            stored_axangles[i, 3] = temp_angle  # in radians!

    assert np.allclose(fast,stored_axangles)

""" These are more general gridding util tests """

def test_linearly_spaced_array_in_rzxz():
    """ From definition, a resolution of 3.75 will give us:
        Two sides of length = 96
        One side of length  = 48
        And thus a total of 96 * 96 * 48 = 442368 points
    """
    grid = create_linearly_spaced_array_in_rzxz(resolution=3.75)
    assert isinstance(grid, Euler)
    assert grid.axis_convention == 'rzxz'
    assert grid.data.shape == (442368, 3)


""" These tests check the fundemental_zone section of the code """


def test_select_fundemental_zone():
    """ Makes sure all the ints from 1 to 230 give answers """
    for _space_group in np.arange(1, 231):
        fz_string = get_proper_point_group_string(_space_group)
        assert fz_string in ['1', '2', '222', '3', '32', '6', '622', '4', '422', '432', '23']

""" Below here are some misc tests """
