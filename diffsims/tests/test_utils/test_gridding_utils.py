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

from diffsims.utils.rotation_conversion_utils import AxAngle, Euler
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz, vectorised_qmult, \
    _create_advanced_linearly_spaced_array_in_rzxz, get_beam_directions

from transforms3d.quaternions import qmult

""" These these test vectorizations """


def test_qmult_vectorisation(random_quats):
    q1 = np.asarray([2, 3, 4, 5])
    fast = vectorised_qmult(q1, random_quats)

    stored_quat = np.ones_like(random_quats)
    for i, row in enumerate(random_quats):
        stored_quat[i] = qmult(q1, row)

    assert np.allclose(fast, stored_quat)


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


""" This tests get_beam_directions """


@pytest.mark.parametrize("crystal_system,expected_corners",
                         [
                             ['monoclinic', [(0, 0, 1), (0, 1, 0), (0, -1, 0)]],
                             ['orthorhombic', [(0, 0, 1), (1, 0, 0), (0, 1, 0)]],
                             ['tetragonal', [(0, 0, 1), (1, 0, 0), (1, 1, 0)]],
                             ['cubic', [(0, 0, 1), (1, 0, 1), (1, 1, 1)]],
                             ['hexagonal', [(0, 0, 1), (2, 1, 0), (1, 1, 0)]],
                             ['trigonal', [(0, 0, 1), (-2, -1, 0), (1, 1, 0)]]
                         ])
def test_get_beam_directions_equal_angle(crystal_system, expected_corners):
    z = get_beam_directions(crystal_system, 1, 'angle')
    assert np.allclose(np.linalg.norm(z, axis=1), 1)
    for corner in expected_corners:
        norm_corner = np.divide(corner, np.linalg.norm(corner))
        assert np.any(np.isin(z, norm_corner))


@pytest.mark.parametrize("crystal_system,expected_corners",
                         [
                             ['monoclinic', [(0, 0, 1), (0, 1, 0), (0, -1, 0)]],
                             ['orthorhombic', [(0, 0, 1), (1, 0, 0), (0, 1, 0)]],
                             ['tetragonal', [(0, 0, 1), (1, 0, 0), (1, 1, 0)]],
                             ['cubic', [(0, 0, 1), (1, 0, 1), (1, 1, 1)]],
                             ['hexagonal', [(0, 0, 1), (2, 1, 0), (1, 1, 0)]],
                             ['trigonal', [(0, 0, 1), (-2, -1, 0), (1, 1, 0)]]
                         ])
def test_get_beam_directions_equal_area(crystal_system, expected_corners):
    z = get_beam_directions(crystal_system, 1, equal='area')
    assert np.allclose(np.linalg.norm(z, axis=1), 1)
    for corner in expected_corners:
        norm_corner = np.divide(corner, np.linalg.norm(corner))
        assert np.any(np.isin(z, norm_corner))


@pytest.mark.parametrize("crystal_system", ['cubic', 'hexagonal'])
def test_equal_area_same_as_equal_angle(crystal_system):
    z_angle = get_beam_directions(crystal_system, 1, equal='angle')
    z_area = get_beam_directions(crystal_system, 1, equal='area')
    assert np.all(z_angle.shape == z_area.shape)


def test_beam_directions_cubic():
    # Following "Orientation precision of TEM-based orientation mapping techniques" - Morawiec et al, Ultramicroscopy 136,2014
    z = get_beam_directions('cubic', 1.6)
    assert z.shape[0] > 950
    assert z.shape[0] < 1050
