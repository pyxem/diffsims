# -*- coding: utf-8 -*-
# Copyright 2017-2023 The diffsims developers
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
from diffsims.generators.sphere_mesh_generators import (
    get_uv_sphere_mesh_vertices,
    get_cube_mesh_vertices,
    get_icosahedral_mesh_vertices,
    get_random_sphere_vertices,
    beam_directions_grid_to_euler,
    _normalize_vectors,
    _get_first_nearest_neighbors,
    _get_angles_between_nn_gridpoints,
    _get_max_grid_angle,
)






def test_vectors_to_euler():
    grid = _normalize_vectors(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]
        )
    )
    ang = np.array(
        [
            [0, 90, 90],
            [0, 90, 0],
            [0, 45, 0],
            [0, 45, 90],
        ]
    )
    assert np.allclose(ang, beam_directions_grid_to_euler(grid))
