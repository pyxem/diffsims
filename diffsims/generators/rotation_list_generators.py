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

"""
Provides users with a range of gridding functions
"""

import numpy as np

from orix.sampling.sample_generators import get_sample_fundamental, get_sample_local
from orix.quaternion.rotation import Rotation
from orix.vector.neo_euler import AxAngle

from diffsims.utils.sim_utils import uvtw_to_uvw
from diffsims.generators.sphere_mesh_generators import (
    get_uv_sphere_mesh_vertices,
    get_cube_mesh_vertices,
    get_icosahedral_mesh_vertices,
    get_random_sphere_vertices,
    beam_directions_grid_to_euler,
)


# Corners determined by requiring a complete coverage of the pole figure. The pole
# figures are plotted with MTEX without implying any crystal symmetry. The plotted
# orientations are obtained by converting vectors returned by get_beam_directions_grid()
# into Euler angles using the procedure by GitHub user @din14970 described here:
# https://github.com/pyxem/orix/issues/125#issuecomment-698956290.
crystal_system_dictionary = {
    "cubic": [(0, 0, 1), (1, 1, 1), (1, 0, 1)],
    "hexagonal": [(0, 0, 0, 1), (9, 1, -10, 0), (2, -1, -1, 0)],
    "trigonal": [(0, 0, 0, 1), (-2, 1, 1, 0), (-1, 2, -1, 0)],
    "tetragonal": [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
    "orthorhombic": [(0, 0, 1), (-1, 0, 0), (0, 1, 0)],
    "monoclinic": [(0, -1, 0), (0, 0, 1), (0, 1, 0)],
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
    z = grid.to_euler()
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
        center = Rotation.from_euler(z)

    orix_grid = get_sample_local(
        resolution=resolution, center=center, grid_width=grid_width
    )
    rotation_list = get_list_from_orix(orix_grid, rounding=2)
    return rotation_list


def get_grid_around_beam_direction(beam_rotation, resolution, angular_range=(0, 360)):
    """Creates a rotation list of rotations for which the rotation is
    about given beam direction.

    Parameters
    ----------
    beam_rotation : tuple
        A desired beam direction as a rotation (rzxz eulers), usually
        found via get_rotation_from_z_to_direction.
    resolution : float
        The resolution of the grid (degrees).
    angular_range : tuple
        The minimum (included) and maximum (excluded) rotation around the
        beam direction to be included.

    Returns
    -------
    rotation_list : list of tuples

    Examples
    --------
    >>> from diffsims.generators.zap_map_generator import get_rotation_from_z_to_direction
    >>> beam_rotation = get_rotation_from_z_to_direction(structure, [1, 1, 1])
    >>> grid = get_grid_around_beam_direction(beam_rotation, 1)
    """
    z = np.deg2rad(np.asarray(beam_rotation))
    beam_rotation = Rotation.from_euler(z)

    angles = np.deg2rad(
        np.arange(start=angular_range[0], stop=angular_range[1], step=resolution)
    )
    axes = np.repeat([[0, 0, 1]], angles.shape[0], axis=0)
    in_plane_rotation = Rotation.from_neo_euler(AxAngle.from_axes_angles(axes, angles))

    orix_grid = beam_rotation * in_plane_rotation
    rotation_list = get_list_from_orix(orix_grid, rounding=2)
    return rotation_list


def get_beam_directions_grid(crystal_system, resolution, mesh="spherified_cube_edge"):
    """Produces an array of beam directions, within the stereographic
    triangle of the relevant crystal system. The way the array is
    constructed is based on different methods of meshing the sphere
    [Cajaravelli2015]_ and can be specified through the `mesh` argument.

    Parameters
    ----------
    crystal_system : str
        Allowed are: 'cubic','hexagonal','trigonal','tetragonal',
        'orthorhombic','monoclinic','triclinic'
    resolution : float
        An angle in degrees representing the worst-case angular
        distance to a first nearest neighbor grid point.
    mesh : str
        Type of meshing of the sphere that defines how the grid is
        created. Options are: uv_sphere, normalized_cube,
        spherified_cube_corner (default), spherified_cube_edge,
        icosahedral, random.

    Returns
    -------
    rotation_list : list of tuples
    """
    if mesh == "uv_sphere":
        points_in_cartesians = get_uv_sphere_mesh_vertices(resolution)
    elif mesh == "spherified_cube_corner":
        points_in_cartesians = get_cube_mesh_vertices(
            resolution, grid_type="spherified_corner"
        )
    elif mesh == "icosahedral":
        points_in_cartesians = get_icosahedral_mesh_vertices(resolution)

    elif mesh == "normalized_cube" or mesh == "spherified_cube_edge":
        # special case: hexagon is a very small slice and 001 point can
        # be isolated. Hence we increase resolution to ensure minimum angle.
        if crystal_system == "hexagonal":
            resolution = resolution / np.sqrt(2)

        if mesh == "normalized_cube":
            points_in_cartesians = get_cube_mesh_vertices(
                resolution, grid_type="normalized"
            )
        else:
            points_in_cartesians = get_cube_mesh_vertices(
                resolution, grid_type="spherified_edge"
            )
    elif mesh == "random":
        points_in_cartesians = get_random_sphere_vertices(resolution)
    else:
        raise NotImplementedError(
            f"The mesh {mesh} is not recognized. "
            f"Please use: uv_sphere, normalized_cube, "
            f"spherified_cube_edge, "
            f"spherified_cube_corner, icosahedral, random"
        )

    # crop to stereographic triangle which depends on crystal system
    epsilon = -1e-13
    if crystal_system == "triclinic":
        return beam_directions_grid_to_euler(points_in_cartesians)
    if crystal_system == "monoclinic":
        points_in_cartesian = points_in_cartesians[
            np.dot(np.array([0, 0, 1]), points_in_cartesians.T) >= epsilon
        ]
        points_in_cartesian = points_in_cartesians[
            np.dot(np.array([1, 0, 0]), points_in_cartesians.T) >= epsilon
        ]
        return beam_directions_grid_to_euler(points_in_cartesian)

    # for all other systems, determine it from the triangle vertices
    corners = crystal_system_dictionary[crystal_system]
    a, b, c = corners[0], corners[1], corners[2]
    if len(a) == 4:
        a, b, c = uvtw_to_uvw(a), uvtw_to_uvw(b), uvtw_to_uvw(c)

    # eliminates those points that lie outside of the stereographic triangle
    points_in_cartesians = points_in_cartesians[
        np.dot(np.cross(a, b), c) * np.dot(np.cross(a, b), points_in_cartesians.T)
        >= epsilon
    ]
    points_in_cartesians = points_in_cartesians[
        np.dot(np.cross(b, c), a) * np.dot(np.cross(b, c), points_in_cartesians.T)
        >= epsilon
    ]
    points_in_cartesians = points_in_cartesians[
        np.dot(np.cross(c, a), b) * np.dot(np.cross(c, a), points_in_cartesians.T)
        >= epsilon
    ]

    angle_grid = beam_directions_grid_to_euler(points_in_cartesians)
    return angle_grid
