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
from diffsims.utils.fundemental_zone_utils import get_proper_point_group_string, reduce_to_fundemental_zone
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz, vectorised_qmult, \
                                          _create_advanced_linearly_spaced_array_in_rzxz

from transforms3d.quaternions import qmult

""" These these test vectorizations """

def test_qmult_vectorisation(random_quats):
    q1 = np.asarray([2,3,4,5])
    fast = vectorised_qmult(q1,random_quats)

    stored_quat = np.ones_like(random_quats)
    for i, row in enumerate(random_quats):
        stored_quat[i] = qmult(q1, row)

    assert np.allclose(fast,stored_quat)

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
