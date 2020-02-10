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
from diffsims.utils.fundamental_zone_utils import get_proper_point_group_string, reduce_to_fundamental_zone
from diffsims.utils.gridding_utils import create_linearly_spaced_array_in_rzxz, rotate_axangle, \
    _create_advanced_linearly_spaced_array_in_rzxz, \
    get_beam_directions, beam_directions_to_euler_angles


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
    """
    zone_string = get_proper_point_group_string(space_group_number)
    raw_grid = create_linearly_spaced_array_in_rzxz(resolution)  # see discussion in diffsims/#50
    raw_grid_axangle = raw_grid.to_AxAngle()
    fz_grid_axangle = reduce_to_fundamental_zone(raw_grid_axangle, zone_string)
    return _returnable_eulers_from_axangle(fz_grid_axangle, 'rzxz', round_to=2)


def get_grid_streographic(crystal_system, resolution, equal='angle'):
    """
    Creates a rotation list by determining the beam directions within the symmetry reduced
    region of the inverse pole figure, corresponding to the specified crystal system, and
    combining this with rotations about the beam direction at a given resolution.

    Parameters
    ----------
    crytal_system : str
        'cubic','hexagonal','trigonal','tetragonal','orthorhombic','monoclinic' and 'triclinic'

    resolution : float
        The maximum misorientation between rotations in the list, as defined according to
        the parameter 'equal'. Specified as an angle in degrees.
    equal : str
        'angle' or 'area'. If 'angle', the misorientation is calculated between each beam direction
        and its nearest neighbour(s). If 'area', the density of points is as in the equal angle case
        but each point covers an equal area.

    Returns
    -------
    rotation_list : list of tuples
        List of rotations
    """
    beam_directions_rzxz = beam_directions_to_euler_angles(get_beam_directions(crystal_system, resolution, equal=equal))
    beam_directions_szxz = beam_directions_rzxz.to_AxAngle().to_Euler(axis_convention='szxz')  # convert to high speed convention

    # drop in all the inplane rotations to form z
    alpha = beam_directions_szxz.data[:, 0]
    beta = beam_directions_szxz.data[:, 1]
    in_plane = np.arange(0, 360, resolution)

    ipalpha = np.asarray(list(product(alpha, np.asarray(in_plane))))
    ipbeta = np.asarray(list(product(beta, np.asarray(in_plane))))
    z = np.hstack((ipalpha[:, 0].reshape((-1, 1)), ipbeta))

    raw_grid = Euler(z, axis_convention='szxz')
    grid_rzxz = raw_grid.to_AxAngle().to_Euler(axis_convention='rzxz')  # convert back Bunge convention to return
    rotation_list = grid_rzxz.to_rotation_list(round_to=2)
    return rotation_list


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
    raw_grid = _create_advanced_linearly_spaced_array_in_rzxz(resolution, 360, max_rotation + 10, 360)
    raw_grid_axangle = raw_grid.to_AxAngle()
    raw_grid_axangle.remove_large_rotations(np.deg2rad(max_rotation))
    if not np.all(np.asarray(center) == 0):
        raw_grid_axangle = rotate_axangle(raw_grid_axangle, center)

    return _returnable_eulers_from_axangle(raw_grid_axangle, 'rzxz', round_to=2)


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
    axangle = euler2axangle(beam_rotation[0], beam_rotation[1], beam_rotation[2], 'rzxz')
    euler_szxz = axangle2euler(axangle[0], axangle[1], 'szxz')  # convert to szxz
    rotation_alpha, rotation_beta = np.rad2deg(euler_szxz[0]), np.rad2deg(euler_szxz[1])

    # see _create_advanced_linearly_spaced_array_in_rzxz for details
    steps_gamma = int(np.ceil((angular_range[1] - angular_range[0]) / resolution))
    alpha = np.asarray([rotation_alpha])
    beta = np.asarray([rotation_beta])
    gamma = np.linspace(angular_range[0], angular_range[1], num=steps_gamma, endpoint=False)
    z = np.asarray(list(product(alpha, beta, gamma)))
    raw_grid = Euler(z, axis_convention='szxz')  # we make use of an uncommon euler angle set here for speed
    grid_rzxz = raw_grid.to_AxAngle().to_Euler(axis_convention='rzxz')
    rotation_list = grid_rzxz.to_rotation_list(round_to=2)
    return rotation_list
