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
import warnings

from diffsims.utils.rotation_conversion_utils import Euler
from diffsims.utils.fundemental_zone_utils import get_proper_point_group_string, reduce_to_fundemental_zone
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz,rotate_axangle, \
                                          _create_advanced_linearly_spaced_array_in_rzxz, \
                                          _get_rotation_to_beam_direction, get_beam_directions, beam_directions_to_euler_angles


def _returnable_eulers_from_axangle(grid,axis_convention,round_to):
    """ Converts a grid of orientations in axis-angle space to Euler angles following a user specified convention and rounding."""
    eulers = grid.to_Euler(axis_convention=axis_convention)
    rotation_list = eulers.to_rotation_list(round_to=round_to)
    return rotation_list

def get_fundemental_zone_grid(space_group_number, resolution):
    """
    Creates a rotation list for the rotations within the fundemental zone of a given space group.

    Parameters
    ----------
    space_group_number : int
        Between 1 and 230

    resolution : float
        The 'resolution' of the grid (degrees)

    Returns
    -------
    rotation_list : list of tuples
    """
    zone_string = get_proper_point_group_string(space_group_number)
    raw_grid = create_linearly_spaced_array_in_rzxz(resolution)  # see discussion in diffsims/#50
    raw_grid_axangle = raw_grid.to_AxAngle()
    fz_grid_axangle = reduce_to_fundemental_zone(raw_grid_axangle, zone_string)
    return _returnable_eulers_from_axangle(fz_grid_axangle,'rzxz',round_to=2)

def get_grid_streographic(crystal_system,resolution,equal='angle'):
    """
    Creates a rotation list by combining the minimum region of the streogram's beam directions
    with in plane rotations

    Parameters
    ----------
    crytal_system : string
        'cubic','hexagonal','tetragonal','orthorhombic','monoclinic','trigonal' add 'triclinc' which acts as 'none'

    resolution : float (angle in degrees)
        Nearest neighbour rotations are seperated by a distance of 'resolution'

    equal : 'angle' or 'area'
        See docstrings for diffsims.utils.gridding_utils.get_beam_directions
        
    Returns
    -------
    rotation_list : 
    """
    from itertools import product
    beam_directions_rzxz = beam_directions_to_euler_angles(get_beam_directions(crystal_system,resolution,equal=equal))
    beam_directions_szxz = beam_directions_rzxz.to_AxAngle().to_Euler(axis_convention='szxz') # convert to high speed convention

    # drop in all the inplane rotations to form z
    alpha = beam_directions.data[:,0]
    beta  = beam_directions.data[:,1]
    in_plane = np.arange(0,360,resolution)

    ipalpha  = np.asarray(list(product(alpha,np.asarray(in_plane))))
    ipbeta   = np.asarray(list(product(beta,np.asarray(in_plane))))
    z = np.hstack((ipalpha[:,0].reshape((-1,1)),ipbeta))

    raw_grid = Euler(z, axis_convention='szxz')
    grid_rzxz = raw_grid.to_AxAngle().to_Euler(axis_convention='rzxz') #convert back Bunge convention to return
    rotation_list = grid_rzxz.to_rotation_list(round_to=2)
    return rotation_list


def get_local_grid(center, max_rotation, resolution):
    """
    Creates a rotation list for the rotations within max_rotation of center at a given rotation.

    Parameters
    ----------
    center : 3 angle tuple
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
    raw_grid = _create_advanced_linearly_spaced_array_in_rzxz(resolution,360,max_rotation + 10,360)
    raw_grid_axangle = raw_grid.to_AxAngle()
    raw_grid_axangle.remove_large_rotations(np.deg2rad(max_rotation))
    if not np.all(np.asarray(center) == 0):
        raw_grid_axangle = rotate_axangle(raw_grid_axangle, center)

    return _returnable_eulers_from_axangle(raw_grid_axangle,'rzxz',round_to=2)


def get_grid_around_beam_direction(beam_direction,resolution, angular_range=(0, 360),cubic=False):
    """
    Creates a rotation list of rotations for which the rotation is about given beam direction

    Parameters
    ----------
    beam_direction : [x,y,z]
        A desired beam direction

    resolution : float
        The 'resolution' of the grid (degrees)

    angular_range : tuple
        The minimum (included) and maximum (excluded) rotation around the beam direction to be included

    cubic : bool (Default=False)
        This only works for cubic systems at the present, when False this raises a warning, set to
        True to supress said warning

    Returns
    -------
    rotation_list : list of tuples
    """
    from itertools import product

    if not cubic:
        warnings.warn("This code only works for cubic systems at present")
    rotation_alpha, rotation_beta = _get_rotation_to_beam_direction(beam_direction)
    # see _create_advanced_linearly_spaced_array_in_rzxz for details
    steps_gamma = int(np.ceil((angular_range[1] - angular_range[0])/resolution))
    alpha = np.asarray([rotation_alpha])
    beta =  np.asarray([rotation_beta])
    gamma = np.linspace(angular_range[0],angular_range[1], num=steps_gamma, endpoint=False)
    z = np.asarray(list(product(alpha, beta, gamma)))
    raw_grid = Euler(z, axis_convention='szxz') #we make use of an uncommon euler angle set here for speed
    grid_rzxz = raw_grid.to_AxAngle().to_Euler(axis_convention='rzxz')
    rotation_list = grid_rzxz.to_rotation_list(round_to=2)
    return rotation_list
