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

import numpy as np
from itertools import product

from diffsims.utils.vector_utils import vectorised_spherical_polars_to_cartesians


__all__ = [
    "beam_directions_grid_to_euler",
    "get_cube_mesh_vertices",
    "get_icosahedral_mesh_vertices",
    "get_random_sphere_vertices",
    "get_uv_sphere_mesh_vertices",
]


def _normalize_vectors(vectors):
    """
    Helper function which returns a list of vectors normalized to length 1 from
    a 2D array representing a list of 3D vectors
    """
    return (vectors.T / np.linalg.norm(vectors, axis=1)).T


def get_uv_sphere_mesh_vertices(resolution):
    """Return the vertices of a UV (spherical coordinate) mesh on a unit
    sphere [Cajaravelli2015]_. The mesh vertices are defined by the
    parametrization:

    .. math::
        x = sin(u)cos(v)\n
        y = sin(u)sin(v)\n
        z = cos(u)

    Parameters
    ----------
    resolution : float
        An angle in degrees. The maximum angle between nearest neighbor
        grid points. In this mesh this occurs on the equator of the
        sphere. All elevation grid lines are separated by at most
        resolution. The step size of u and v are rounded up to get an
        integer number of elevation and azimuthal grid lines with equal
        spacing.

    Returns
    -------
    points_in_cartesian : numpy.ndarray (N,3)
        Rows are x, y, z where z is the 001 pole direction
    """
    pass


def get_cube_mesh_vertices(resolution, grid_type="spherified_corner"):
    """Return the (x, y, z) coordinates of the vertices of a cube mesh
    on a sphere. To generate the mesh, a cube is made to surround the
    sphere. The surfaces of the cube are subdivided into a grid. The
    vectors from the origin to these grid points are normalized to unit
    length. The grid on the cube can be generated in three ways, see
    `grid_type` and reference [Cajaravelli2015]_.

    Parameters
    ----------
    resolution : float
        The maximum angle in degrees between first nearest neighbor grid
        points.
    grid_type : str
        The type of cube grid, can be either `normalized` or `spherified_edge`
        or `spherified_corner` (default). For details see notes.

    Returns
    -------
    points_in_cartesian : numpy.ndarray (N,3)
        Rows are x, y, z where z is the 001 pole direction

    Notes
    -----
    The resolution determines the maximum angle between first nearest
    neighbor grid points, but to get an integer number of points between the
    cube face center and the edges, the number of grid points is rounded up.
    In practice this means that resolution is always an upper limit.
    Additionally, where on the grid this maximum angle will be will depend
    on the type of grid chosen. Resolution says something about the maximum
    angle but nothing about the distribution of nearest neighbor angles or
    the minimum angle - also this is fixed by the chosen grid.

    In the normalized grid, the grid on the surface of the cube is linear.
    The maximum angle between nearest neighbors is found between the <001>
    directions and the first grid point towards the <011> directions. Points
    approaching the edges and corners of the cube will have a smaller angular
    deviation, so orientation space will be oversampled there compared to the
    cube faces <001>.

    In the spherified_edge grid, the grid is constructed so that there
    are still two sets of perpendicular grid lines parallel to the {100}
    directions on each cube face, but the spacing of the grid lines is
    chosen so that the angles between the grid points on the line
    connecting the face centers (<001>) to the edges (<011>) are equal.
    The maximum angle is also between the <001> directions and the first
    grid point towards the <011> edges. This grid slightly oversamples the
    directions between <011> and <111>

    The spherified_corner case is similar to the spherified_edge case, but
    the spacing of the grid lines is chosen so that the angles between
    the grid points on the line connecting the face centers to the cube
    corners (<111>) is equal. The maximum angle in this grid is from the
    corners to the first grid point towards the cube face centers.

    References
    ----------
    .. [Cajaravelli2015] O. S. Cajaravelli, "Four Ways to Create a Mesh for a Sphere,"
        https://medium.com/@oscarsc/four-ways-to-create-a-mesh-for-a-sphere-d7956b825db4.
    """
    pass


def get_icosahedral_mesh_vertices(resolution):
    """
    Return the (x, y, z) coordinates of the vertices of an icosahedral
    mesh of a cube, see [Cajaravelli2015]_. Method was adapted from
    meshzoo [Meshzoo]_.

    Parameters
    ----------
    resolution : float
        The maximum angle in degrees between neighboring grid points.
        Since the mesh is generated iteratively, the actual maximum angle
        in the mesh can be slightly smaller.

    Returns
    -------
    points_in_cartesian : numpy.ndarray (N,3)
        Rows are x, y, z where z is the 001 pole direction

    References
    ----------
    .. [Meshzoo] The `meshzoo.sphere` module.
    """
    pass


def get_random_sphere_vertices(resolution, seed=None):
    """
    Create a mesh that randomly samples the surface of a sphere

    Parameters
    ----------
    resolution : float
        The expected mean angle between nearest neighbor
        grid points in degrees.
    seed : int, optional
        passed to np.random.default_rng(), defaults to None which
        will give a "new" random result each time

    Returns
    -------
    points_in_cartesian : numpy.ndarray (N,3)
        Rows are x, y, z where z is the 001 pole direction

    References
    ----------
    https://mathworld.wolfram.com/SpherePointPicking.html
    """
    pass


def beam_directions_grid_to_euler(vectors):
    """
    Convert list of vectors representing zones to a list of Euler angles
    in the bunge convention with the constraint that phi1=0.

    Parameters
    ----------
    vectors: numpy.ndarray (N, 3)
        N 3-dimensional vectors to convert to Euler angles

    Returns
    -------
    grid: numpy.ndarray (N, 3)
        Euler angles in bunge convention corresponding to each vector in
        degrees.

    Notes
    -----
    The Euler angles represent the orientation of the crystal if that
    particular vector were parallel to the beam direction [001]. The
    additional constraint of phi1=0 means that this orientation is uniquely
    defined for most vectors. phi1 represents the rotation of the crystal
    around the beam direction and can be interpreted as the rotation of
    a particular diffraction pattern.
    """
    norm = np.linalg.norm(vectors, axis=1)
    z_comp = vectors[:, 2]
    # second euler angle: around x' = angle between z and z''
    Phi = np.arccos(z_comp / norm)
    # first euler angle: around z = angle between x and x'
    x_comp = vectors[:, 0]
    y_comp = vectors[:, 1]
    norm_proj = np.linalg.norm(vectors[:, :2], axis=1)
    sign = np.sign(y_comp)
    # correct for where we have y=0
    sign[y_comp == 0] = np.sign(x_comp[y_comp == 0])
    phi2 = sign * np.nan_to_num(np.arccos(x_comp / norm_proj))
    # phi1 is just 0, rotation around z''
    phi1 = np.zeros(phi2.shape[0])
    grid = np.rad2deg(np.vstack([phi1, Phi, np.pi / 2 - phi2]).T)
    return grid
