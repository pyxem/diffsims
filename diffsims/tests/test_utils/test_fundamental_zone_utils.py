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

from diffsims.utils.fundemental_zone_utils import get_proper_point_group_string, reduce_to_fundemental_zone, numpy_bounding_plane

""" These tests check the fundemental_zone section of the code """

def test_select_fundemental_zone():
    """ Makes sure all the ints from 1 to 230 give answers """
    for _space_group in np.arange(1, 231):
        fz_string = get_proper_point_group_string(_space_group)
        assert fz_string in ['1', '2', '222', '3', '32', '6', '622', '4', '422', '432', '23']

@pytest.mark.xfail(strict=True)
def test_edge_case_numpy_bounding_plane():
    z = np.asarray([1,1,1,np.inf])
    numpy_bounding_plane(data=z,vector=[1,1,1],distance=1)
