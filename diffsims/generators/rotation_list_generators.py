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
from itertools import product

from orix.sampling.sample_generators import get_sample_fundamental, get_sample_local
from orix.quaternion.rotation import Rotation
from orix.vector.neo_euler import AxAngle

from diffsims.utils.vector_utils import vectorised_spherical_polars_to_cartesians
from diffsims.utils.sim_utils import uvtw_to_uvw

crystal_system_dictionary = {
    "cubic": [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
    "hexagonal": [(0, 0, 0, 1), (1, 0, -1, 0), (1, 1, -2, 0)],
    "trigonal": [(0, 0, 0, 1), (0, -1, 1, 0), (1, -1, 0, 0)],
    "tetragonal": [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
    "orthorhombic": [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
    "monoclinic": [(0, 0, 1), (0, 1, 0), (0, -1, 0)],
}


def get_list_from_orix(grid, rounding=2):
    """
    Converts an orix sample to a rotation list

    Parameters
    ----------
    grid : orix.quaternion.rotation.Rotation
        A grid of rotations
    rounding : int, optional
        The number of decimal places to retain, defaults to 2

    Returns
    -------
    rotation_list : list of tuples
        A rotation list
    """
    z = grid.to_euler(convention="bunge")
    rotation_list = z.data.tolist()
    i = 0
    while i < len(rotation_list):
        rotation_list[i] = tuple(
            np.round(np.rad2deg(rotation_list[i]), decimals=rounding)
        )
        i += 1

    return rotation_list


def get_fundamental_zone_grid(resolution=2, point_group=None, space_group=None):
    """
    Generates an equispaced grid of rotations within a fundamental zone.

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour (degrees)
    point_group : orix.quaternion.symmetry.Symmetry, optional
        One of the 11 proper point groups, defaults to None
    space_group: int, optional
        Between 1 and 231, defaults to None

    Returns
    -------
    rotation_list : list of tuples
        Grid of rotations lying within the specified fundamental zone
    """

    orix_grid = get_sample_fundamental(resolution=resolution, space_group=space_group)
    rotation_list = get_list_from_orix(orix_grid, rounding=2)
    return rotation_list


def get_local_grid(resolution=2, center=None, grid_width=10):
    """
    Generates a grid of rotations about a given rotation

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour (degrees)
    center : euler angle tuple or orix.quaternion.rotation.Rotation, optional
        The rotation at which the grid is centered. If None (default) uses the identity
    grid_width : float, optional
        The largest angle of rotation away from center that is acceptable (degrees)

    Returns
    -------
    rotation_list : list of tuples
    """
    if isinstance(center, tuple):
        z = np.deg2rad(np.asarray(center))
        center = Rotation.from_euler(z, convention="bunge", direction="crystal2lab")

    orix_grid = get_sample_local(
        resolution=resolution, center=center, grid_width=grid_width
    )
    rotation_list = get_list_from_orix(orix_grid, rounding=2)
    return rotation_list


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
    z = np.deg2rad(np.asarray(beam_rotation))
    beam_rotation = Rotation.from_euler(z, convention="bunge", direction="crystal2lab")

    angles = np.deg2rad(
        np.arange(start=angular_range[0], stop=angular_range[1], step=resolution)
    )
    axes = np.repeat([[0, 0, 1]], angles.shape[0], axis=0)
    in_plane_rotation = Rotation.from_neo_euler(AxAngle.from_axes_angles(axes, angles))

    orix_grid = beam_rotation * in_plane_rotation
    rotation_list = get_list_from_orix(orix_grid, rounding=2)
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
    steps_theta = int(np.ceil(180 / resolution))  # elevation
    steps_psi = int(np.ceil(360 / resolution))  # azimuthal

    psi = np.linspace(0, 2 * np.pi, num=steps_psi, endpoint=False)

    if equal == "angle":
        theta = np.linspace(0, np.pi, num=steps_theta, endpoint=True)

    elif equal == "area":
        # http://mathworld.wolfram.com/SpherePointPicking.html
        v_array = np.linspace(0, 1, num=steps_psi)
        theta = np.arccos(2 * v_array - 1)  # in radians

    psi_theta = np.asarray(list(product(psi, theta)))
    r = np.ones((psi_theta.shape[0], 1))
    points_in_spherical_polars = np.hstack((r, psi_theta))

    # keep only one theta ==0 point, specifically the psi ==0 one
    points_in_spherical_polars = points_in_spherical_polars[
        np.logical_or(
            np.abs(points_in_spherical_polars[:, 2]) > 0,
            points_in_spherical_polars[:, 1] == 0,
        )
    ]

    # keep only one theta ==180 point, specifically the psi ==0 one
    points_in_spherical_polars = points_in_spherical_polars[
        np.logical_or(
            np.abs(points_in_spherical_polars[:, 2]) < np.deg2rad(180),
            points_in_spherical_polars[:, 1] == 0,
        )
    ]

    points_in_cartesians = vectorised_spherical_polars_to_cartesians(
        points_in_spherical_polars
    )

    if crystal_system == "triclinic":
        return points_in_cartesians

    corners = crystal_system_dictionary[crystal_system]
    a, b, c = corners[0], corners[1], corners[2]
    if len(a) == 4:
        a, b, c = uvtw_to_uvw(a), uvtw_to_uvw(b), uvtw_to_uvw(c)

    # eliminates those points that lie outside of the streographic triangle
    points_in_cartesians = points_in_cartesians[
        np.dot(np.cross(a, b), c) * np.dot(np.cross(a, b), points_in_cartesians.T) >= 0
    ]
    points_in_cartesians = points_in_cartesians[
        np.dot(np.cross(b, c), a) * np.dot(np.cross(b, c), points_in_cartesians.T) >= 0
    ]
    points_in_cartesians = points_in_cartesians[
        np.dot(np.cross(c, a), b) * np.dot(np.cross(c, a), points_in_cartesians.T) >= 0
    ]

    return points_in_cartesians
