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
from diffsims.utils.gridding import get_local_grid, get_grid_around_beam_direction
from diffsims.utils.gridding_utils import Euler


def test_get_local_grid():
    grid = get_local_grid((0,0,0),10,2)
    assert isinstance(grid,list)
    assert isinstance(grid[0],tuple)

def test_get_grid_around_beam_direction():
    grid_simple = get_grid_around_beam_direction((0,0,0),1,(0, 360))
    assert isinstance(grid_simple,list)
    assert isinstance(grid_simple[0],tuple)
    assert len(grid_simple) == 360

    grid_rotated = get_grid_around_beam_direction((0,90,0),1,(0, 360))
    assert isinstance(grid_simple,list)
    assert isinstance(grid_simple[0],tuple)
    assert len(grid_simple) == 360

    axangle = Euler(np.asarray(grid_rotated),axis_convention='rzxz').to_AxAngle()
    assert np.allclose(axangle.data[1:,0][axangle.data[1:,3] > 0],0)
    assert np.allclose(axangle.data[1:,1][axangle.data[1:,3] > 0],1)
    assert np.allclose(axangle.data[1:,2][axangle.data[1:,3] > 0],0)
