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


def test_random_sphere_mesh():
    grid = get_random_sphere_vertices(1)
    assert grid.shape[0] == 10313
    assert grid.shape[1] == 3


def test_seed_for_random_sphere_mesh():
    grid_7 = get_random_sphere_vertices(resolution=3, seed=7)
    grid_7_again = get_random_sphere_vertices(resolution=3, seed=7)
    grid_8 = get_random_sphere_vertices(resolution=3, seed=8)
    assert np.allclose(grid_7, grid_7_again)
    assert not np.allclose(grid_7, grid_8)


def test_get_uv_sphere_mesh_vertices():
    grid = get_uv_sphere_mesh_vertices(10)
    np.testing.assert_almost_equal(np.sum(grid), 0)
    assert grid.shape[0] == 614
    assert grid.shape[1] == 3
    np.testing.assert_almost_equal(np.sum(grid), 0)
    grid_unique = np.unique(grid, axis=0)
    assert grid.shape[0] == grid_unique.shape[0]


@pytest.mark.parametrize(
    "grid_type,expected_len",
    [
        ("normalized", 866),
        ("spherified_edge", 602),
        ("spherified_corner", 866),
    ],
)
def test_get_cube_mesh_vertices(grid_type, expected_len):
    grid = get_cube_mesh_vertices(10, grid_type=grid_type)
    assert grid.shape[0] == expected_len
    assert grid.shape[1] == 3
    np.testing.assert_almost_equal(np.sum(grid), 0)
    grid_unique = np.unique(grid, axis=0)
    assert grid.shape[0] == grid_unique.shape[0]
    test_vectors = np.round(
        _normalize_vectors(np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [-1, 1, -1]])), 13
    ).tolist()
    grid = np.round(grid, 13)
    for i in test_vectors:
        assert i in grid.tolist()


def test_get_cube_mesh_vertices_exception():
    with pytest.raises(Exception):
        get_cube_mesh_vertices(10, "non_existant")


def test_first_nearest_neighbors():
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
    fnn = _normalize_vectors(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
    )
    angles = np.array([45, 45, 45, 45])
    fnn_test = _get_first_nearest_neighbors(grid)
    angles_test = _get_angles_between_nn_gridpoints(grid)
    assert np.allclose(fnn, fnn_test)
    assert np.allclose(angles, angles_test)
    assert _get_max_grid_angle(grid) == 45.0


def test_icosahedral_grid():
    grid = get_icosahedral_mesh_vertices(10)
    assert grid.shape[0] == 642
    assert grid.shape[1] == 3
    np.testing.assert_almost_equal(np.sum(grid), 0)
    grid_unique = np.unique(grid, axis=0)
    assert grid.shape[0] == grid_unique.shape[0]


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
