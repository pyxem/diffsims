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

"""
Provides users with a range of gridding functions
"""

import numpy as np
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz, get_proper_point_group_string, \
                                          reduce_to_fundemental_zone, rotate_axangle, \
                                          _create_advanced_linearly_spaced_array_in_rzxz, Euler. \
                                          _get_rotation_to_beam_direction



def get_fundemental_zone_grid(space_group_number, resolution):
    """
    Parameters
    ----------

    space_group_number : int


    resolution : float
        The 'resolution' of the grid (degrees)

    Returns
    -------
    """
    zone_string = get_proper_point_group_string(space_group_number)

    raw_grid = create_linearly_spaced_array_in_rzxz(resolution)  # could cut the count down here
    raw_grid_ax_angle = raw_grid.to_AxAngle()
    """
    could use a conditional .remove_large_angles() here for speed.
    we know what are max angles are, so save some time by cutting out chunks
    see Figure 5 of "On 3 dimensional misorientation spaces"
    """
    fz_grid = reduce_to_fundemental_zone(raw_grid_axangle, zone_string)
    # convert to rzxz
    return None


def get_local_grid(center, max_rotation, resolution):
    """

    Parameters
    ----------
    center : 3 angle tuple
        The orientation that acts as the center of the grid, specified in the
        'rzxz' convention (degrees)

    max_rotation : float
        The largest rotation away from 'center' that should be included in the grid (degrees)

    resolution : float
        The 'resolution' of the grid (degrees)

    Returns
    -------
    rotation_list : list of tuples
    """
    raw_grid = _create_advanced_linearly_spaced_array_in_rzxz(resolution,360,max_rotation + 10,360)
    raw_grid_axangle = raw_grid.to_AxAngle()
    raw_grid_axangle.remove_large_rotations(np.deg2rad(max_rotation))
    if not np.all(np.asarray(center) == 0):
        raw_grid_axangle = rotate_axangle(raw_grid_axangle, center)
    eulers = raw_grid_axangle.to_Euler(axis_convention='rzxz')
    rotation_list = eulers.to_rotation_list(round_to=2)
    return rotation_list


def get_grid_around_beam_direction(beam_direction,resolution, angular_range=(0, 360)):
    """

    Parameters
    ----------
    beam_direction : [x,y,z]
        A desired beam direction

    resolution : float
        The 'resolution' of the grid (degrees)

    angular_range : tuple
        The minimum (included) and maximum (excluded) rotation around the beam direction to be included


    Returns
    -------
    rotation_list : list of tuples
    """
    from itertools import product

    rotation_alpha, rotation_beta = _get_rotation_to_beam_direction(beam_direction)
    # see _create_advanced_linearly_spaced_array_in_rzxz for details
    steps_gamma = int(np.ceil((angular_range[1] - angular_range[0])/resolution))
    alpha = np.asarray([rotation_alpha])
    beta =  np.asarray([rotation_beta])
    gamma = np.linspace(angular_range[0],angular_range[1], num=steps_gamma, endpoint=False)
    z = np.asarray(list(product(alpha, beta, gamma)))
    raw_grid = Euler(z, axis_convention='szxz')
    grid_rzxz = raw_grid.to_AxAngle().to_Euler(axis_convention='rzxz')
    rotation_list = grid_rzxz.to_rotation_list(round_to=2)
    return rotation_list
