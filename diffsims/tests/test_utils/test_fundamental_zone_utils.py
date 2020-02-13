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

from diffsims.utils.fundamental_zone_utils import get_proper_point_group_string, reduce_to_fundamental_zone,\
    numpy_bounding_plane, cyclic_group, dihedral_group, \
    remove_out_of_domain_rotations,\
    generate_mask_from_rodrigues_frank
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz
from diffsims.utils.rotation_conversion_utils import Euler, AxAngle

""" These tests check the fundamental_zone section of the code """


@pytest.fixture()
def sparse_rzxz_grid():
    z = create_linearly_spaced_array_in_rzxz(5)
    axangle = z.to_AxAngle()
    return axangle


def assert_volume_changes_obeyed(r, start_size, volume):
    """
    Confirms that 'r' is within 10% of the predicted size based on theory [1]

    Parameters
    ----------
    r : np.array [n,4]
        Kept orientations in r-frank representation
    start_size : int
        The number of elements originally considered
    volume : float
        The fraction of axis-angle space volume occupied by the target zone

    Notes
    -----
    [1] Page 103 - Orientations and Rotations - A. Morawiec
    """
    assert r.data.shape[0] > 0
    assert r.data.shape[0] < (start_size * volume * 1.1)
    assert r.data.shape[0] > (start_size * volume * 0.9)


""" Tests of internal functionality """


def test_select_fundamental_zone():
    """ Makes sure all the ints from 1 to 230 give answers """
    # Not done with parametrize to avoid clogging the output log (it's 230 tests if you do it that way)
    for _space_group in np.arange(1, 231):
        fz_string = get_proper_point_group_string(_space_group)
        assert fz_string in ['1', '2', '222', '3', '32', '6', '622', '4', '422', '432', '23']


@pytest.mark.parametrize("point_group_str", ['432', '222', '23', '1'])
def test_remove_out_of_domain_rotations(sparse_rzxz_grid, point_group_str):
    """ Tests that cutting rotations bigger than a certain angle behaves as expected"""
    start_size = sparse_rzxz_grid.data.shape[0]
    ax = remove_out_of_domain_rotations(sparse_rzxz_grid, point_group_str)
    assert ax.data.shape[0] > 0

    if point_group_str == '432':
        assert ax.data.shape[0] < (start_size / 2)
    elif point_group_str == '222' or point_group_str == '23':
        assert ax.data.shape[0] < (start_size * 0.8)
    elif point_group_str == '1':
        assert start_size == ax.data.shape[0]


@pytest.mark.parametrize("point_group_str", ['2', '222', '23', '432'])
def test_generate_mask_from_rodrigues_frank(sparse_rzxz_grid, point_group_str):
    """ Tests that mask generation from rf parameterizations behaves as hoped """
    start_size = sparse_rzxz_grid.data.shape[0]
    mask = generate_mask_from_rodrigues_frank(sparse_rzxz_grid, point_group_str)
    assert np.sum(mask) > 0

    if point_group_str == '2':
        assert np.sum(mask) > (start_size / 2.2)
        assert np.sum(mask) < (start_size / 1.8)
    elif point_group_str == '432':
        assert np.sum(mask) < (start_size / 2)
    elif point_group_str == '222' or point_group_str == '23':
        assert np.sum(mask) < (start_size * 0.8)


@pytest.mark.xfail(strict=True)
def test_edge_case_numpy_bounding_plane():
    """ Confirms you can't have infinite bounding planes """
    z = np.asarray([1, 1, 1, np.inf])
    numpy_bounding_plane(data=z, vector=[1, 1, 1], distance=1)


@pytest.mark.parametrize("fz_string", ['1', '2', '222', '3', '32', '6', '622', '4', '422', '432', '23'])
def test_non_zero_returns(sparse_rzxz_grid, fz_string):
    """ All code paths must return some orientations """
    reduced_data = reduce_to_fundamental_zone(sparse_rzxz_grid, fz_string)
    assert reduced_data.data.shape[0] > 0


""" Cyclic case """


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_cyclic(order):
    """ For z axis rotations, included size scaling should go with 'order' """
    axis = np.hstack((np.zeros((2000, 2)), np.ones((2000, 1))))  # z
    rf = np.tan(np.linspace(0, np.pi / 2, 2000))  # Uniform spacing in angle space
    z = np.hstack((axis, rf.reshape(-1, 1)))
    reduced = cyclic_group(z, order)
    assert np.allclose(np.sum(reduced), 2000 / order, atol=3)


@pytest.mark.parametrize("fz_string", ['1', '2', '3', '4', '6'])
def test_orthogonal_linear_case_for_cyclic_group(fz_string):
    """ Rotations about the x direction are never removed by the effects of the cyclic group """
    axis = np.hstack((np.ones((2000, 1)), np.zeros((2000, 2))))  # x
    angle = np.linspace(-np.pi, np.pi, 2000)
    along_x_axis = AxAngle(np.hstack((axis, angle.reshape(-1, 1))))
    reduced = reduce_to_fundamental_zone(along_x_axis, fz_string)
    assert reduced.data.shape[0] == along_x_axis.data.shape[0]


@pytest.mark.parametrize("order", ['1', '2', '3', '4', '6'])
def test_cyclic_groups(sparse_rzxz_grid, order):
    """ Asserts volume changes broadly as expected """
    start_size = sparse_rzxz_grid.data.shape[0]
    r = reduce_to_fundamental_zone(sparse_rzxz_grid, order)
    volume = 1 / int(order)
    assert_volume_changes_obeyed(r, start_size, volume)


""" Dihedral case """


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_x_direction_dihedral(order):
    """ All dihedral groups have the same effect on x-rotations """
    axis = np.hstack((np.ones((2000, 1)), np.zeros((2000, 2))))
    rf = np.tan(np.linspace(0, np.pi / 2, 2000))
    z = np.hstack((axis, rf.reshape(-1, 1)))
    reduced = dihedral_group(z, order)
    assert np.allclose(np.sum(reduced), 2000 / 2, atol=3)


@pytest.mark.parametrize("point_group_str", ['222', '32', '422', '622'])
def test_dihedral_groups(sparse_rzxz_grid, point_group_str):
    """ Asserts volume changes broadly as expected """
    order = int(point_group_str[0])
    volume = 1 / (2 * order)  # From page 103 of Morawiec
    start_size = sparse_rzxz_grid.data.shape[0]
    r = reduce_to_fundamental_zone(sparse_rzxz_grid, point_group_str)
    assert_volume_changes_obeyed(r, start_size, volume)


""" Cubic case """


@pytest.fixture()
def dense_rzxz_grid():
    z = create_linearly_spaced_array_in_rzxz(1)
    axangle = z.to_AxAngle()
    return axangle


def test_tetragonal_group(dense_rzxz_grid):
    """ Tests volume lies with the two bounding spheres """
    full_volume = (4 / 3) * np.pi * (180)**3
    outer_volume = (4 / 3) * np.pi * (90)**3
    r_inner = 90 * np.sqrt(1 / 3)  # See https://en.wikipedia.org/wiki/Octahedron
    inner_volume = (4 / 3) * np.pi * (r_inner)**3
    start_size = dense_rzxz_grid.data.shape[0]
    r = reduce_to_fundamental_zone(dense_rzxz_grid, '23')
    observed_ratio = r.data.shape[0] / start_size
    assert observed_ratio < outer_volume / full_volume
    assert observed_ratio > inner_volume / full_volume
