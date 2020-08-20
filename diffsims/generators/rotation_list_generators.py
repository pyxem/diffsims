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

from orix.gridding.grid_generators import get_local_grid as oglg
from orix.gridding.grid_generators import get_fundamental_zone_grid as ogfz

from transforms3d.euler import euler2axangle, axangle2euler
from transforms3d.euler import axangle2euler, euler2axangle, euler2mat
from transforms3d.quaternions import quat2axangle, axangle2quat, mat2quat, qmult

from diffsims.utils.rotation_conversion_utils import *
from diffsims.utils.vector_utils import vectorised_spherical_polars_to_cartesians

# Defines the maximum rotation angles [theta_max,psi_max,psi_min] associated with the
# corners of the symmetry reduced region of the inverse pole figure for each crystal system.
crystal_system_dictionary = {
    "cubic": [45, 54.7, 0],
    "hexagonal": [45, 90, 26.565],
    "trigonal": [45, 90, -116.5],
    "tetragonal": [45, 90, 0],
    "orthorhombic": [90, 90, 0],
    "monoclinic": [90, 0, -90],
    "triclinic": [180, 360, 0],
}

def _rotate_axangle(Axangles, new_center):
    """
    Rotates a series of orientation described by axangle to a new center

    Parameters
    ----------
    Axangles : diffsims.Axangles
        Pre-rotation
    new_center : (alpha,beta,gamma)
        The location of the (0,0,0) rotation as an rzxz euler angle

    Returns
    -------
    AxAngles : diffsims.Axangles
        Rotated
    See Also
    --------
    generators.get_local_grid
    """

    quats = Axangles.to_Quat()
    q = mat2quat(rotation_matrix_from_euler_angles((new_center)))
    stored_quats = vectorised_qmult(q, quats)

    return AxAngle.from_Quat(stored_quats)


def _beam_directions_to_euler_angles(points_in_cartesians):
    """
    Converts an array of cartesians (x,y,z unit basis vectors) to the euler angles that would take [0,0,1] to [x,y,z]
    Parameters
    ----------
    points_in_cartesians :
         Generally output from get_beam_directions()
    Returns
    -------
    diffsims.Euler :
         The appropriate euler angles
    """
    axes = np.cross(
        [0, 0, 1], points_in_cartesians
    )  # in unit cartesians so this is fine, [0,0,1] returns [0,0,0]
    norms = np.linalg.norm(axes, axis=1).reshape(-1, 1)
    angle = np.arcsin(norms)

    normalised_axes = np.ones_like(axes)
    np.divide(axes, norms, out=normalised_axes, where=norms != 0)

    np_axangles = np.hstack((normalised_axes, angle.reshape((-1, 1))))
    eulers = AxAngle(np_axangles).to_Euler(axis_convention="rzxz")
    return eulers


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
    z = ogfz(resolution,space_group_number=space_group_number)
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


def get_grid_stereographic(crystal_system, resolution, equal="angle"):
    """
    This functionality is deprecated. The following outline is only given to
    aid dev work
    """
    return get_fundamental_zone_grid(1,resolution)

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
    rotation_list = None
    raise NotImplementedError("This functionality will be (re)added in future")
    return rotation_list


def get_beam_directions_grid(crystal_system, resolution, equal="angle"):
    """
    Produces an array of beam directions, evenly (see equal argument) spaced that lie within the streographic
    triangle of the relevant crystal system.

    Parameters
    ----------
    crystal_system : str
        Allowed are: 'cubic','hexagonal','trigonal','tetragonal','orthorhombic','monoclinic','triclinic'

    resolution : float
        An angle in degrees. If the 'equal' option is set to 'angle' this is the misorientation between a
        beam direction and its nearest neighbour(s). For 'equal'=='area' the density of points is as in
        the equal angle case but each point covers an equal area

    equal : str
        'angle' (default) or 'area'

    Returns
    -------
    points_in_cartesians : np.array (N,3)
        Rows are x,y,z where z is the 001 pole direction.
    """
    theta_max, psi_max, psi_min = crystal_system_dictionary[crystal_system]

    # see docstrings for np.arange, np.linspace has better endpoint handling
    steps_theta = int(np.ceil((theta_max - 0) / resolution))
    steps_psi = int(np.ceil((psi_max - psi_min) / resolution))
    theta = np.linspace(
        0, np.deg2rad(theta_max), num=steps_theta
    )  # radians as we're about to make spherical polar cordinates
    if equal == "area":
        # http://mathworld.wolfram.com/SpherePointPicking.html
        v_1 = (1 + np.cos(np.deg2rad(psi_max))) / 2
        v_2 = (1 + np.cos(np.deg2rad(psi_min))) / 2
        v_array = np.linspace(min(v_1, v_2), max(v_1, v_2), num=steps_psi)
        psi = np.arccos(2 * v_array - 1)  # in radians
    elif equal == "angle":
        # now in radians as we're about to make spherical polar cordinates
        psi = np.linspace(np.deg2rad(psi_min), np.deg2rad(psi_max), num=steps_psi)

    psi_theta = np.asarray(list(product(psi, theta)))
    r = np.ones((psi_theta.shape[0], 1))
    points_in_spherical_polars = np.hstack((r, psi_theta))

    # keep only theta ==0 psi ==0, do this with np.abs(theta) > 0 or psi == 0 - more generally use the smallest psi value
    points_in_spherical_polars = points_in_spherical_polars[
        np.logical_or(
            np.abs(psi_theta[:, 1]) > 0, psi_theta[:, 0] == np.min(psi_theta[:, 0])
        )
    ]
    points_in_cartesians = vectorised_spherical_polars_to_cartesians(
        points_in_spherical_polars
    )

    if crystal_system == "cubic":
        # add in the geodesic that runs [1,1,1] to [1,0,1]
        v1 = np.divide([1, 1, 1], np.sqrt(3))
        v2 = np.divide([1, 0, 1], np.sqrt(2))

        def cubic_corner_geodesic(t):
            # https://math.stackexchange.com/questions/1883904/a-time-parameterization-of-geodesics-on-the-sphere
            w = v2 - np.multiply(np.dot(v1, v2), v1)
            w = np.divide(w, np.linalg.norm(w))
            # return in cartesians with t_end = np.arccos(np.dot(v1,v2))
            return np.add(
                np.multiply(np.cos(t.reshape(-1, 1)), v1),
                np.multiply(np.sin(t.reshape(-1, 1)), w),
            )

        t_list = np.linspace(0, np.arccos(np.dot(v1, v2)), num=steps_theta)
        geodesic = cubic_corner_geodesic(t_list)
        points_in_cartesians = np.vstack((points_in_cartesians, geodesic))
        # the great circle (from [1,1,1] to [1,0,1]) forms a plane (with the
        # origin), points on the same side as (0,0,1) are safe, the others are not
        plane_normal = np.cross(
            v2, v1
        )  # dotting this with (0,0,1) gives a positive number
        points_in_cartesians = points_in_cartesians[
            np.dot(points_in_cartesians, plane_normal) >= 0
        ]  # 0 is the points on the geodesic

    return points_in_cartesians

def get_beam_directions_grid(crystal_system, resolution, equal="angle"):
    """
    Produces an array of beam directions, evenly (see equal argument) spaced that lie within the streographic
    triangle of the relevant crystal system.

    Parameters
    ----------
    crystal_system : str
        Allowed are: 'cubic','hexagonal','trigonal','tetragonal','orthorhombic','monoclinic','triclinic'

    resolution : float
        An angle in degrees. If the 'equal' option is set to 'angle' this is the misorientation between a
        beam direction and its nearest neighbour(s). For 'equal'=='area' the density of points is as in
        the equal angle case but each point covers an equal area

    equal : str
        'angle' (default) or 'area'

    Returns
    -------
    points_in_cartesians : np.array (N,3)
        Rows are x,y,z where z is the 001 pole direction.
    """
    raise ValueError("This function has been renamed to get_beam_directions_grid")
