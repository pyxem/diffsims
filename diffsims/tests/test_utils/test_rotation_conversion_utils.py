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
import numpy as np

from transforms3d.euler import euler2quat, quat2axangle, axangle2euler
from diffsims.utils.rotation_conversion_utils import vectorised_euler2quat, vectorised_quat2axangle, vectorised_axangle2euler,\
    vectorised_axangle_to_correct_range, convert_axangle_to_correct_range, \
    Euler, AxAngle


@pytest.mark.parametrize("axis_convention", ['rzxz', 'szxz'])
def test_vectorised_euler2quat(random_eulers, axis_convention):
    fast = vectorised_euler2quat(random_eulers, axis_convention)

    stored_quats = np.ones((random_eulers.shape[0], 4))
    for i, row in enumerate(random_eulers):
        temp_quat = euler2quat(row[0], row[1], row[2], axis_convention)
        for j in [0, 1, 2, 3]:
            stored_quats[i, j] = temp_quat[j]

    assert np.allclose(fast, stored_quats)


def test_vectorised_quat2axangle(random_quats):
    fast = vectorised_quat2axangle(random_quats)

    stored_axangles = np.ones((random_quats.shape[0], 4))
    for i, row in enumerate(random_quats):
        temp_ax = quat2axangle(row)[0]
        temp_angle = quat2axangle(row)[1]
        for j in [0, 1, 2]:
            stored_axangles[i, j] = temp_ax[j]
        stored_axangles[i, 3] = temp_angle  # in radians!

    assert np.allclose(fast, stored_axangles)


@pytest.mark.parametrize("axis_convention", ['rzxz', 'szxz'])
def test_vectorised_axangle2euler(random_axangles, axis_convention):
    fast = vectorised_axangle2euler(random_axangles, axis_convention)

    stored_euler = np.ones((random_axangles.shape[0], 3))
    for i, row in enumerate(random_axangles):
        a_array = axangle2euler(row[:3], row[3], axis_convention)
        for j in [0, 1, 2]:
            stored_euler[i, j] = a_array[j]

    assert np.allclose(fast, stored_euler)


def test_convert_to_correct_range_vectorisation(random_axangles):
    fast = vectorised_axangle_to_correct_range(random_axangles)
    stored_axangles = np.empty_like(random_axangles)
    for i, row in enumerate(random_axangles):
        temp_vect, temp_angle = convert_axangle_to_correct_range(row[:3], row[3])
        for j in [0, 1, 2]:
            stored_axangles[i, j] = temp_vect[j]
            stored_axangles[i, 3] = temp_angle  # in radians!

    assert np.all(stored_axangles[:, 3] >= 0)
    assert np.all(stored_axangles[:, 3] < np.pi)
    assert np.allclose(fast, stored_axangles, atol=1e-3)


""" These tests check that AxAngle and Euler behave in good ways """


def test_interconversion_euler_axangle(random_axangles):
    """
    This function checks (with random numbers) that .to_Axangle() and .to_Euler()
    go back and forth correctly
    """
    z = random_axangles.copy()
    z[:, :3] = np.divide(random_axangles[:, :3], np.linalg.norm(
        random_axangles[:, :3], axis=1).reshape(z.shape[0], 1))  # normalise
    axangle = AxAngle(z)
    e = axangle.to_Euler(axis_convention='rzxz')
    transform_back = e.to_AxAngle()
    assert isinstance(transform_back, AxAngle)
    assert np.allclose(transform_back.data, axangle.data)


def test_slow_to_euler_case(random_eulers):
    """
    This function checks that to_Axangle runs with rarer conventions on the eulers
    """
    e = Euler(random_eulers, 'sxyz')
    axangle = e.to_AxAngle()
    assert isinstance(axangle, AxAngle)


@pytest.mark.xfail(strict=True, raises=ValueError)
class TestsThatFail:
    def test_odd_convention_euler2quat(self, random_eulers):
        vectorised_euler2quat(random_eulers, axes='sxyz')

    def test_odd_convention_mat2euler(self):
        ax = AxAngle(np.asarray([[1, 0, 0, 0.2]]))
        ax.to_Euler(axis_convention='sxyz')

    def test_inf_quat(self):
        qdata = np.asarray([[0, np.inf, 1, 1]])
        edata = vectorised_quat2axangle(qdata)

    def test_small_quat(self):
        qdata = np.asarray([[1e-8, 1e-8, 1e-8, 1e-8]])
        edata = vectorised_quat2axangle(qdata)


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

    def test_to_rotation_list_no_round(self, good_array):
        l = Euler(good_array).to_rotation_list(round_to=None)
        assert isinstance(l, list)
        assert isinstance(l[0], tuple)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_warning_code(self, good_array):
        euler = Euler(good_array)
        euler.data[:, :] = 1e-5
        euler._check_data()

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
