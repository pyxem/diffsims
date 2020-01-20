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

from transforms3d.euler import euler2quat, quat2axangle, axangle2euler
from diffsims.utils.rotation_conversion_utils import vectorised_euler2quat,vectorised_quat2axangle,vectorised_axangle2euler,\
                                                     Euler,AxAngle

def test_vectorised_euler2quat(random_eulers):
    fast = vectorised_euler2quat(random_eulers,'rzxz')

    stored_quats = np.ones((random_eulers.shape[0],4))
    for i,row in enumerate(random_eulers):
        temp_quat = euler2quat(row[0],row[1],row[2],'rzxz')
        for j in [0,1,2,3]:
            stored_quats[i,j] = temp_quat[j]

    assert np.allclose(fast,stored_quats)

def test_vectorised_quat2axangle(random_quats):
    fast = vectorised_quat2axangle(random_quats)

    stored_axangles = np.ones((random_quats.shape[0],4))
    for i,row in enumerate(random_quats):
        temp_ax    = quat2axangle(row)[0]
        temp_angle = quat2axangle(row)[1]
        for j in [0,1,2]:
            stored_axangles[i,j] = temp_ax[j]
        stored_axangles[i,3] = temp_angle #in radians!

    assert np.allclose(fast,stored_axangles)

def test_vectorised_axangle2euler(random_axangles):
    fast = vectorised_axangle2euler(random_axangles,'rzxz')

    stored_euler = np.ones((random_axangles.shape[0], 3))
    for i, row in enumerate(random_axangles):
        a_array = axangle2euler(row[:3], row[3],'rzxz')
        for j in [0, 1, 2]:
            stored_euler[i, j] = a_array[j]

    assert np.allclose(fast,stored_euler)

""" These tests check that AxAngle and Euler behave in good ways """

def test_interconversion_euler_axangle(random_axangles):
    """
    This function checks (with random numbers) that .to_Axangle() and .to_Euler()
    go back and forth correctly
    """
    z = random_axangles.copy()
    z[:,:3] = np.divide(random_axangles[:,:3],np.linalg.norm(random_axangles[:,:3],axis=1).reshape(z.shape[0],1)) #normalise
    axangle = AxAngle(z)
    e = axangle.to_Euler(axis_convention='rzxz')
    transform_back = e.to_AxAngle()
    assert isinstance(transform_back, AxAngle)
    assert np.allclose(transform_back.data, axangle.data)

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
