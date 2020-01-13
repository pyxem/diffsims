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
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz, select_fundemental_zone, reduce_to_fundemental_zone, rotate_axangle


def get_fundemental_zone_grid(space_group_number,resolution,center=(0,0,0)):
    """
    Parameters
    ----------

    space_group_number : int


    resolution : float
        The 'resolution' of the grid (degrees)

    center : 3 angle tuple
        The orientation that acts as the center of the grid, specified in the
        'rzxz' convention (degrees)

    Returns
    -------
    """
    raw_grid = create_linearly_spaced_array_in_rzxz(resolution)
    raw_grid_ax_angle = raw_grid.to_AxAngle()
    fz = select_fundemental_zone(space_group_number)
    fz_grid = reduce_to_fundemental_zone(raw_grid_axangle,fz)
    # rotate to the center
    # convert to rzxz
    return None


def get_local_grid(center,max_rotation,resolution):
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
    """
    raw_grid = create_linearly_spaced_array_in_rzxz(resolution)
    raw_grid_axangle = raw_grid.to_AxAngle()
    raw_grid_axangle.remove_large_rotations(max_rotation)
    if np.any(center!=0):
        raw_grid_axangle = rotate_axangle(raw_grid_axangle,center)
    returnable_euler = raw_grid_axangle.to_Euler(axis_convention='rzxz')
    #figure out the final return style.
    return returnable_euler

def get_grid_around_beam_direction(beam_direction,resolution,angular_range=(0,360)):
    """

    Parameters
    ----------
    beam_direction : 3 angle tuple
        An orientation that acts as the center of the grid, specified in the
        'rzxz' convention (degrees)

    resolution : float
        The 'resolution' of the grid (degrees)

    angular_range : tuple
        The minimum (included) and maximum (excluded) rotation around the beam direction to be included


    Returns
    -------
    """
