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

"""Provides users with a range of gridding functions."""

import numpy as np
from typing import Mapping, Optional

from orix.sampling.sample_generators import get_sample_fundamental, get_sample_local
from orix.quaternion.rotation import Rotation
from orix.vector.neo_euler import AxAngle
from orix.quaternion import Symmetry, symmetry
from orix.sampling import sample_S2
from orix.vector import Vector3d

from diffsims.utils.sim_utils import uvtw_to_uvw
from diffsims.utils.orientation_utils import ConstrainedRotation


__all__ = [
    "get_beam_directions_grid",
    "get_fundamental_zone_grid",
    "get_grid_around_beam_direction",
    "get_list_from_orix",
    "get_local_grid",
]

# for all other systems, determine it from the triangle vertices
# Corners determined by requiring a complete coverage of the pole figure. The pole
# figures are plotted with MTEX without implying any crystal symmetry. The plotted
# orientations are obtained by converting vectors returned by get_beam_directions_grid()
# into Euler angles using the procedure by GitHub user @din14970 described here:
# https://github.com/pyxem/orix/issues/125#issuecomment-698956290.
_CRYSTAL_SYSTEM_DICTIONARY = {
    "cubic": ((0, 0, 1), (1, 1, 1), (1, 0, 1)),
    "hexagonal": ((0, 0, 0, 1), (9, 1, -10, 0), (2, -1, -1, 0)),
    "trigonal": ((0, 0, 0, 1), (-2, 1, 1, 0), (-1, 2, -1, 0)),
    "tetragonal": ((0, 0, 1), (1, 0, 0), (1, 1, 0)),
    "orthorhombic": ((0, 0, 1), (-1, 0, 0), (0, 1, 0)),
    "monoclinic": ((0, -1, 0), (0, 0, 1), (0, 1, 0)),
}


def get_list_from_orix(grid, rounding=2):
    """Converts an orix sample to a rotation list.

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


def get_reduced_fundamental_zone_grid(
    resolution: float,
    mesh: str = None,
    point_group: Symmetry = None,
) -> ConstrainedRotation:
    """Produces orientations to align various crystallographic directions with
    the z-axis, with the constraint that the first Euler angle phi_1=0.
    The crystallographic directions sample the fundamental zone, representing
    the smallest region of symmetrically unique directions of the relevant
    crystal system or point group.

    Parameters
    ----------
    resolution
        An angle in degrees representing the maximum angular distance to a
        first nearest neighbor grid point.
    mesh
        Type of meshing of the sphere that defines how the grid is created. See
        orix.sampling.sample_S2 for all the options. A suitable default is
        chosen depending on the crystal system.
    point_group
        Symmetry operations that determines the unique directions. Defaults to
        no symmetry, which means sampling all 3D unit vectors.

    Returns
    -------
    ConstrainedRotation
        (N, 3) array representing Euler angles for the different orientations
    """
    if point_group is None:
        point_group = symmetry.C1

    if mesh is None:
        s2_auto_sampling_map = {
            "triclinic": "icosahedral",
            "monoclinic": "icosahedral",
            "orthorhombic": "spherified_cube_edge",
            "tetragonal": "spherified_cube_edge",
            "cubic": "spherified_cube_edge",
            "trigonal": "hexagonal",
            "hexagonal": "hexagonal",
        }
        mesh = s2_auto_sampling_map[point_group.system]

    s2_sample: Vector3d = sample_S2(resolution, method=mesh)
    fundamental: Vector3d = s2_sample[s2_sample <= point_group.fundamental_sector]
    return ConstrainedRotation.from_vector(fundamental)
