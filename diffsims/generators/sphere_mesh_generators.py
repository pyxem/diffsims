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
from scipy.spatial import cKDTree
from itertools import product

from diffsims.utils.vector_utils import vectorised_spherical_polars_to_cartesians


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
    steps_theta = int(np.ceil(180 / resolution)) + 1  # elevation
    steps_psi = int(np.ceil(360 / resolution))  # azimuthal

    psi = np.linspace(0, 2 * np.pi, num=steps_psi, endpoint=False)
    theta = np.linspace(0, np.pi, num=steps_theta, endpoint=True)

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
    return points_in_cartesians


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
    # the angle between 001 and 011
    max_angle = np.deg2rad(45)
    # the distance on the cube face from 001 to 011 grid point
    max_dist = 1
    if grid_type == "normalized":
        grid_len = np.tan(np.deg2rad(resolution))
        steps = np.ceil(max_dist / grid_len)
        i = np.arange(-steps, steps) / steps
    elif grid_type == "spherified_edge":
        steps = np.ceil(np.rad2deg(max_angle) / resolution)
        k = np.arange(-steps, steps)
        theta = np.arctan(max_dist) / steps
        i = np.tan(k * theta)
    elif grid_type == "spherified_corner":
        # the angle from 001 to 111
        max_angle_111 = np.arccos(1 / np.sqrt(3))
        res_111 = np.deg2rad(resolution)
        steps = np.ceil(max_angle_111 / res_111)
        k = np.arange(-steps, steps)
        theta = np.arctan(np.sqrt(2)) / steps
        i = np.tan(k * theta) / np.sqrt(2)
    else:
        raise ValueError(
            f"grid type {grid_type} not a valid grid type. "
            f"Valid options: normalized, spherified_edge, "
            f"spherified_corner."
        )
    x, y = np.meshgrid(i, i)
    x, y = x.ravel(), y.ravel()
    z = np.ones(x.shape[0])
    # the grid on all faces of the cube, avoiding overlap of points on edges
    bottom = np.vstack([-x, -y, -z]).T
    top = np.vstack([x, y, z]).T
    east = np.vstack([z, x, -y]).T
    west = np.vstack([-z, -x, y]).T
    south = np.vstack([x, -z, y]).T
    north = np.vstack([-x, z, -y]).T
    # two corners are missing with this procedure
    m_c = np.array([[-1, 1, 1], [1, -1, -1]])
    # combine
    all_vecs = np.vstack([bottom, top, east, west, south, north, m_c])
    return _normalize_vectors(all_vecs)


def _compose_from_faces(corners, faces, n):
    """
    Helper function to refine a grid starting from a platonic solid,
    adapted from meshzoo

    Parameters
    ----------
    corners: numpy.ndarray (N, 3)
        Coordinates of vertices for starting shape
    faces : list of 3-tuples of int elements
        Each tuple in the list corresponds to the vertex indices making
        up the face of the mesh
    n : int
        number of times the mesh is refined

    Returns
    -------
    vertices: numpy.ndarray (N, 3)
        The coordinates of the refined mesh vertices.

    See also
    --------
    :func:`get_icosahedral_mesh_vertices`
    """
    # create corner nodes
    vertices = [corners]
    vertex_count = len(corners)
    corner_nodes = np.arange(len(corners))
    # create edges
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    edges = list(edges)
    # create edge nodes:
    edge_nodes = {}
    t = np.linspace(1 / n, 1.0, n - 1, endpoint=False)
    corners = vertices[0]
    k = corners.shape[0]
    for edge in edges:
        i0, i1 = edge
        new_vertices = np.outer(1 - t, corners[i0]) + np.outer(t, corners[i1])
        vertices.append(new_vertices)
        vertex_count += len(vertices[-1])
        edge_nodes[edge] = np.arange(k, k + len(t))
        k += len(t)
    triangle_cells = []
    k = 0
    for i in range(n):
        j = np.arange(n - i)
        triangle_cells.append(np.column_stack([k + j, k + j + 1, k + n - i + j + 1]))
        j = j[:-1]
        triangle_cells.append(
            np.column_stack([k + j + 1, k + n - i + j + 2, k + n - i + j + 1])
        )
        k += n - i + 1
    triangle_cells = np.vstack(triangle_cells)
    for face in faces:
        corners = face
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        is_edge_reverted = [False, False, False]
        for k, edge in enumerate(edges):
            if edge[0] > edge[1]:
                edges[k] = (edge[1], edge[0])
                is_edge_reverted[k] = True
        # First create the interior points in barycentric coordinates
        if n == 1:
            num_new_vertices = 0
        else:
            bary = (
                np.hstack(
                    [[np.full(n - i - 1, i), np.arange(1, n - i)] for i in range(1, n)]
                )
                / n
            )
            bary = np.array([1.0 - bary[0] - bary[1], bary[1], bary[0]])
            corner_verts = np.array([vertices[0][i] for i in corners])
            vertices_cart = np.dot(corner_verts.T, bary).T

            vertices.append(vertices_cart)
            num_new_vertices = len(vertices[-1])
        # translation table
        num_nodes_per_triangle = (n + 1) * (n + 2) // 2
        tt = np.empty(num_nodes_per_triangle, dtype=int)
        # first the corners
        tt[0] = corner_nodes[corners[0]]
        tt[n] = corner_nodes[corners[1]]
        tt[num_nodes_per_triangle - 1] = corner_nodes[corners[2]]
        # then the edges.
        # edge 0
        tt[1:n] = edge_nodes[edges[0]]
        if is_edge_reverted[0]:
            tt[1:n] = tt[1:n][::-1]
        #
        # edge 1
        idx = 2 * n
        for k in range(n - 1):
            if is_edge_reverted[1]:
                tt[idx] = edge_nodes[edges[1]][n - 2 - k]
            else:
                tt[idx] = edge_nodes[edges[1]][k]
            idx += n - k - 1
        #
        # edge 2
        idx = n + 1
        for k in range(n - 1):
            if is_edge_reverted[2]:
                tt[idx] = edge_nodes[edges[2]][k]
            else:
                tt[idx] = edge_nodes[edges[2]][n - 2 - k]
            idx += n - k
        # now the remaining interior nodes
        idx = n + 2
        j = vertex_count
        for k in range(n - 2):
            for _ in range(n - k - 2):
                tt[idx] = j
                j += 1
                idx += 1
            idx += 2
        vertex_count += num_new_vertices
    vertices = np.concatenate(vertices)
    return vertices


def _get_first_nearest_neighbors(points, leaf_size=50):
    """
    Helper function to get an array of first nearest neighbor points
    for all points in a point cloud

    Parameters
    ----------
    points : numpy.ndarray (N, D)
        Point cloud with N points in D dimensions
    leaf_size : int
        The NN search is performed using a cKDTree object. The way
        this tree is constructed depends on leaf_size, so this parameter
        will influence speed of tree construction and search.

    Returns
    -------
    nn1_vec : numpy.ndarray (N,D)
        Point cloud with N points in D dimensions, representing the nearest
        neighbor point of each point in "points"
    """
    tree = cKDTree(points, leaf_size)
    # get the indexes of the first nearest neighbor of each vertex
    nn1 = tree.query(points, k=2)[1][:, 1]
    nn1_vec = points[nn1]
    return nn1_vec


def _get_angles_between_nn_gridpoints(vertices, leaf_size=50):
    """
    Helper function to get the angles between all nearest neighbor grid
    points on a grid of a sphere.
    """
    # normalize the vertex vectors
    vertices = _normalize_vectors(vertices)
    nn1_vec = _get_first_nearest_neighbors(vertices, leaf_size)
    # the dot product between each point and its nearest neighbor
    nn_dot = np.sum(vertices * nn1_vec, axis=1)
    # angles
    angles = np.rad2deg(np.arccos(nn_dot))
    return angles


def _get_max_grid_angle(vertices, leaf_size=50):
    """
    Helper function to get the maximum angle between nearest neighbor grid
    points on a grid.
    """
    return np.max(_get_angles_between_nn_gridpoints(vertices, leaf_size))


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
    t = (1.0 + np.sqrt(5.0)) / 2.0
    corners = np.array(
        [
            [-1, +t, +0],
            [+1, +t, +0],
            [-1, -t, +0],
            [+1, -t, +0],
            #
            [+0, -1, +t],
            [+0, +1, +t],
            [+0, -1, -t],
            [+0, +1, -t],
            #
            [+t, +0, -1],
            [+t, +0, +1],
            [-t, +0, -1],
            [-t, +0, +1],
        ]
    )
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    n = 1
    angle = _get_max_grid_angle(corners)
    # maybe not the most efficient approach, but the least work
    while angle > resolution:
        vertices = _compose_from_faces(corners, faces, n)
        angle = _get_max_grid_angle(vertices)
        n = n + 1
    # push all nodes to the sphere
    norms = np.sqrt(np.einsum("ij,ij->i", vertices, vertices))
    vertices = (vertices.T / norms.T).T
    return vertices


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
    # convert resolution in degrees to number of points
    number = int(1 / (4 * np.pi) * (360 / resolution) ** 2)
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    xyz = rng.normal(size=(number, 3))
    xyz = (xyz.T / np.linalg.norm(xyz, axis=1)).T
    return xyz


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
