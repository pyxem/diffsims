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

"""
Provides users with a range of gridding functions
"""

import numpy as np
import warnings
from itertools import product

from transforms3d.euler import euler2axangle, axangle2euler

from diffsims.utils.rotation_conversion_utils import Euler
from diffsims.utils.fundamental_zone_utils import (
    get_proper_point_group_string,
    reduce_to_fundamental_zone,
)
from diffsims.utils.gridding_utils import (
    create_linearly_spaced_array_in_rzxz,
    rotate_axangle,
    _create_advanced_linearly_spaced_array_in_rzxz,
    get_beam_directions,
    beam_directions_to_euler_angles,
)


def _returnable_eulers_from_axangle(grid, axis_convention, round_to):
    """ Converts a grid of orientations in axis-angle space to Euler
    angles following a user specified convention and rounding."""
    eulers = grid.to_Euler(axis_convention=axis_convention)
    rotation_list = eulers.to_rotation_list(round_to=round_to)
    return rotation_list


def get_fundamental_zone_grid(space_group_number, resolution):
    """
    Creates a rotation list for the rotations within the fundamental zone of a given space group.

    Parameters
    ----------
    space_group_number : int
        Between 1 and 230

    resolution : float
        The 'resolution' of the grid (degrees)

    Returns
    -------
    rotation_list : list of tuples

    Notes
    -----
    """
    #raise Deprecation warning
    #g get grid from orix
    #convert_to_rotation_list()
    return None

def get_local_grid(center, max_rotation, resolution):
    """
    Creates a rotation list for the rotations within max_rotation of center at a given rotation.

    Parameters
    ----------
    center : tuple
        The orientation that acts as the center of the grid, as euler angles specified in the
        'rzxz' convention (degrees)

    max_rotation : float
        The largest rotation away from 'center' that should be included in the grid (degrees)

    resolution : float
        The 'resolution' of the grid (degrees)

    Returns
    -------
    rotation_list : list of tuples
    """
    #raise Deprecation warning
    #g get grid from orix
    #convert_to_rotation_list()
    return None


def get_grid_around_beam_direction(beam_rotation, resolution, angular_range=(0, 360)):
    """
    Creates a rotation list of rotations for which the rotation is about given beam direction

    Parameters
    ----------
    beam_rotation : tuple
        A desired beam direction as a rotation (rzxz eulers), usually found via get_rotation_from_z_to_direction

    resolution : float
        The resolution of the grid (degrees)

    angular_range : tuple
        The minimum (included) and maximum (excluded) rotation around the beam direction to be included

    Returns
    -------
    rotation_list : list of tuples

    Example
    -------
    >>> from diffsims.generators.zap_map_generator import get_rotation_from_z_to_direction
    >>> beam_rotation = get_rotation_from_z_to_direction(structure,[1,1,1])
    >>> grid = get_grid_around_beam_direction(beam_rotation,1)
    """

    beam_rotation = np.deg2rad(beam_rotation)
    axangle = euler2axangle(
        beam_rotation[0], beam_rotation[1], beam_rotation[2], "rzxz"
    )
    euler_szxz = axangle2euler(axangle[0], axangle[1], "szxz")  # convert to szxz
    rotation_alpha, rotation_beta = np.rad2deg(euler_szxz[0]), np.rad2deg(euler_szxz[1])

    # see _create_advanced_linearly_spaced_array_in_rzxz for details
    steps_gamma = int(np.ceil((angular_range[1] - angular_range[0]) / resolution))
    alpha = np.asarray([rotation_alpha])
    beta = np.asarray([rotation_beta])
    gamma = np.linspace(
        angular_range[0], angular_range[1], num=steps_gamma, endpoint=False
    )
    z = np.asarray(list(product(alpha, beta, gamma)))
    raw_grid = Euler(
        z, axis_convention="szxz"
    )  # we make use of an uncommon euler angle set here for speed
    grid_rzxz = raw_grid.to_AxAngle().to_Euler(axis_convention="rzxz")
    rotation_list = grid_rzxz.to_rotation_list(round_to=2)
    return rotation_list
