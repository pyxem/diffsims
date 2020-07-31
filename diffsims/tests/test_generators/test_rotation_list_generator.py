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
from diffsims.generators.rotation_list_generators import (
    get_local_grid,
    get_grid_around_beam_direction,
    get_fundamental_zone_grid
    )

@pytest.mark.skip(reason="Not implemented")
def test_calls_to_orix():
    _ = get_local_grid((0,0,0),31,15)
    _ = get_fundamental_zone_grid(space_group_number, resolution=20)

def test_get_grid_around_beam_direction():
    grid_simple = get_grid_around_beam_direction([1, 1, 1], 1, (0, 360))
    assert isinstance(grid_simple, list)
    assert isinstance(grid_simple[0], tuple)
    assert len(grid_simple) == 360
