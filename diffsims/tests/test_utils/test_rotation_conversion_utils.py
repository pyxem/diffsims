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
from diffsims.utils.rotation_conversion_utils import vectorised_euler2quat,vectorised_quat2axangle,vectorised_axangle2euler


@pytest.fixture()
def random_eulers():
    """ Using [0,360] [0,180] and [0,360] as ranges """
    alpha = np.random.rand(100) * 360
    beta  = np.random.rand(100) * 180
    gamma = np.random.rand(100) * 360
    eulers = np.asarray((alpha,beta,gamma)).T
    return np.deg2rad(eulers)

def test_vectorised_euler2quat(random_eulers):
    fast = vectorised_euler2quat(random_eulers,'rzxz')

    stored_quats = np.ones((random_eulers.shape[0],4))
    for i,row in enumerate(random_eulers):
        temp_quat = euler2quat(row[0],row[1],row[2],'rzxz')
        for j in [0,1,2,3]:
            stored_quats[i,j] = temp_quat[j]

    assert np.allclose(fast,stored_quats)

@pytest.fixture()
def random_quats():
    q_rand = np.random.random(size=(1000,4))*7
    return q_rand

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

@pytest.fixture()
def random_axangles():
    axangle_rand = np.random.random(size=(1000,4)) * np.pi
    return axangle_rand

def test_vectorised_axangle2euler(random_axangles):
    fast = vectorised_axangle2euler(random_axangles,'rzxz')

    stored_euler = np.ones((random_axangles.shape[0], 3))
    for i, row in enumerate(random_axangles):
        a_array = axangle2euler(row[:3], row[3],'rzxz')
        for j in [0, 1, 2]:
            stored_euler[i, j] = a_array[j]

    #can compare directly in radians:
    assert np.allclose(fast,stored_euler)
