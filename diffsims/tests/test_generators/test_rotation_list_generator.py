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
    get_fundamental_zone_grid,
    get_beam_directions_grid,
)


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param(
            get_local_grid(resolution=30, center=None, grid_width=35),
            marks=pytest.mark.xfail(reason="Downstream bug"),
        ),
        get_fundamental_zone_grid(space_group=20, resolution=20),
    ],
)
def test_get_grid(grid):
    assert isinstance(grid, list)
    assert len(grid) > 0
    assert isinstance(grid[0], tuple)


@pytest.mark.xfail(reason="Functionality removed")
def test_get_grid_around_beam_direction():
    grid_simple = get_grid_around_beam_direction([1, 1, 1], 1, (0, 360))
    assert isinstance(grid_simple, list)
    assert isinstance(grid_simple[0], tuple)
    assert len(grid_simple) == 360


@pytest.mark.parametrize(
    "crystal_system",
    [
        "cubic",
        "hexagonal",
        "trigonal",
        "tetragonal",
        "orthorhombic",
        "monoclinic",
        "triclinic",
    ],
)
def test_get_beam_directions_grid(crystal_system):
    for equal in ["angle", "area"]:
        _ = get_beam_directions_grid(crystal_system, 5, equal=equal)
