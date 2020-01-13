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

from diffsims.utils.gridding_utils import AxAngle, Euler, create_linearly_spaced_array_in_rzxz,select_fundemental_zone,reduce_to_fundemental_zone,_create_advanced_linearly_spaced_array_in_rzxz

def test_linearly_spaced_array_in_rzxz():
    """ From definition, a resolution of 3.75 will give us:
        Two sides of length = 96
        One side of length  = 48
        And thus a total of 96 * 96 * 48 = 442368 points
    """
    grid = create_linearly_spaced_array_in_rzxz(resolution=3.75)
    assert isinstance(grid,Euler)
    assert grid.axis_convention == 'szxz'
    assert grid.data.shape == (442368,3)

def test_advanced_rzxz_gridding():
    """

    """
    def process_angles(raw_angles,max_rotation):
        raw_angles = raw_angles.to_AxAngle()
        return raw_angles.remove_large_rotations(max_rotation)

    long_true_way = process_angles(create_linearly_spaced_array_in_rzxz(1),20)
    quick_way     = process_angles(_create_advanced_linearly_spaced_array_in_rzxz(1,20),20)
    assert long_true_way.shape == quick_way.shape











""" These tests check the fundemental_zone section of the code """

def test_select_fundemental_zone():
    """ Makes sure all the ints from 1 to 230 give answers """
    for _space_group in np.arange(1,231):
        fz_string = select_fundemental_zone(_space_group)
        assert fz_string in ['1','2','222','3','32','6','622','4','422','432','23']

""" These tests check that AxAngle and Euler behave in good ways """

class TestAxAngle:
    @pytest.fixture()
    def good_array(self):
        return np.asarray([[1,0,0,1],
                             [0,1,0,1.1]])

    def test_good_array__init__(self,good_array):
        assert isinstance(AxAngle(good_array),AxAngle)

    def test_remove_large_rotations(self,good_array):
        axang = AxAngle(good_array)
        axang.remove_large_rotations(1.05) #removes 1 rotations
        assert axang.data.shape == (1,4)

    @pytest.mark.xfail(raises = ValueError, strict=True)
    class TestCorruptingData:
        @pytest.fixture()
        def axang(self,good_array):
            return AxAngle(good_array)

        def test_bad_shape(self,axang):
            axang.data = axang.data[:,:2]
            axang._check_data()

        def test_dumb_angle(self,axang):
            axang.data[0,3] = -0.5
            axang._check_data()

        def test_denormalised(self,axang):
            axang.data[:,0] = 3
            axang._check_data()

class TestEuler:
    @pytest.fixture()
    def good_array(self):
        return np.asarray([[32,80,21],
                           [40,10,11]])
    def test_good_array__init__(self,good_array):
        assert isinstance(Euler(good_array),Euler)

    def test_toAxangle(self,good_array):
        """ Conventions are grim, so only test the code elements """
        axang = Euler(good_array,axis_convention='szxz').to_AxAngle()
        assert isinstance(axang,AxAngle)
        axang._check_data()

    @pytest.mark.xfail(raises = ValueError, strict=True)
    class TestCorruptingData:
        @pytest.fixture()
        def euler(self,good_array):
            return Euler(good_array)

        def test_bad_shape(self,euler):
            euler.data = euler.data[:,:2]
            euler._check_data()

        def test_dumb_angle(self,euler):
            euler.data[0,0] = 700
            euler._check_data()

def test_interconversion_euler_axangle():
    """
    This function checks (with random numbers) that .to_Axangle() and .to_Euler()
    go back and forth correctly
    """
    axes = np.random.random_sample((1000,3))
    axes = np.divide(axes,np.linalg.norm(axes,axis=1).reshape(1000,1))
    assert np.allclose(np.linalg.norm(axes,axis=1),1) #check for input normalisation
    angles = np.multiply(np.random.random_sample((1000,1)),np.pi)
    axangle   = AxAngle(np.concatenate((axes,angles),axis=1))
    transform = AxAngle(np.concatenate((axes,angles),axis=1))
    e = transform.to_Euler(axis_convention='szxz')
    transform_back = e.to_AxAngle()
    assert np.allclose(transform_back.data,axangle.data)
