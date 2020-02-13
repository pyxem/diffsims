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
Helper functions for gridding
"""

import numpy as np
from itertools import product
from transforms3d.euler import axangle2euler, euler2axangle, euler2mat
from transforms3d.quaternions import quat2axangle, axangle2quat, mat2quat, qmult
from diffsims.utils.rotation_conversion_utils import *
from diffsims.utils.vector_utils import vectorised_spherical_polars_to_cartesians

# Defines the maximum rotation angles [theta_max,psi_max,psi_min] associated with the
# corners of the symmetry reduced region of the inverse pole figure for each crystal system.
crystal_system_dictionary = {'cubic': [45, 54.7, 0],
                             'hexagonal': [45, 90, 26.565],
                             'trigonal': [45, 90, -116.5],
                             'tetragonal': [45, 90, 0],
                             'orthorhombic': [90, 90, 0],
                             'monoclinic': [90, 0, -90],
                             'triclinic': [180, 360, 0]}


def vectorised_qmult(q1, qdata):
    """ A vectorised implementation that multiplies qdata (array) by q1 (single quaternion) """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = qdata[:, 0], qdata[:, 1], qdata[:, 2], qdata[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z]).T


def rotation_matrix_from_euler_angles(euler_angles):
    """
    Finds the rotation matrix that takes (0,0,0) to (alpha,beta,gamma)

    Parameters
    ----------
    euler_angles : (alpha,beta,gamma)
        in 'rzxz'

    Returns
    -------
    rotation_matrix : np.array(3,3)

    See Also
    --------
    generators.get_local_grid
    """
    M_initial = euler2mat(0, 0, 0, 'rzxz')
    ai, aj, ak = np.deg2rad(euler_angles[0]), np.deg2rad(euler_angles[1]), np.deg2rad(euler_angles[2])
    M_target = euler2mat(ai, aj, ak, 'rzxz')
    rotation_matrix = M_target @ np.linalg.inv(M_initial)
    return rotation_matrix


def rotate_axangle(Axangles, new_center):
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


def create_linearly_spaced_array_in_rzxz(resolution):
    """
    Creates an array of euler angles that covers all rotation space.

    Parameters
    ----------
    resolution : angle in degrees

    Returns
    -------
    grid : diffsims.Euler

    Notes
    -----
    We use angular ranges alpha [0,360], beta [0,180] and gamma [0,360] in
    line with Convention 4 described in Reference [1]

    References
    ----------
    [1]  D Rowenhorst et al 2015 Modelling Simul. Mater. Sci. Eng.23 083501
         https://iopscience.iop.org/article/10.1088/0965-0393/23/8/083501/meta
    """
    return _create_advanced_linearly_spaced_array_in_rzxz(resolution, max_alpha=360, max_beta=180, max_gamma=360)


def _create_advanced_linearly_spaced_array_in_rzxz(resolution, max_alpha, max_beta, max_gamma):
    """
    Creates an array of euler angles that covers a specified range of euler angles.

    Parameters
    ----------
    resolution :
        The final lists will include [0,res,2*res etc] (in degrees)
    max_alpha :
        End point of the range of alpha, not included (in degrees)
    max_beta :
        End point of the range of beta, not included (in degrees)
    max_gamma :
        End point of the range of gamma, not included (in degrees)

    Returns
    -------
    diffsims.Euler

    """
    # We use np.linspace rather than np.arange to get list of evenly spaced Euler
    # angles due to better end point handling. Therefore convert "step_size" to a "num"
    steps_alpha = int(np.ceil((max_alpha - 0) / resolution))
    steps_beta = int(np.ceil((max_beta - 0) / resolution))
    steps_gamma = int(np.ceil((max_gamma - 0) / resolution))

    alpha = np.linspace(0, max_alpha, num=steps_alpha, endpoint=False)
    beta = np.linspace(0, max_beta, num=steps_beta, endpoint=False)
    gamma = np.linspace(0, max_gamma, num=steps_gamma, endpoint=False)
    z = np.asarray(list(product(alpha, beta, gamma)))
    return Euler(z, axis_convention='rzxz')


def get_beam_directions(crystal_system, resolution, equal='angle'):
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
    Notes
    -----
    For all cases: The input 'resolution' may differ slightly from the expected value. This is so that each of the corners
    of the streographic triangle are included. Actual 'resolution' will always be equal to or higher than the input resolution. As
    an example, if resolution is set to 4 to cover a range [0,90] we can't include both endpoints. The code makes 23 steps
    of 3.91 degrees instead.

    For the cubic case: Each edge of the streographic triangle will behave as expected. The region above the (1,0,1), (1,1,1) edge
    will (for implementation reasons) be slightly more densly packed than the wider region.
    """
    theta_max, psi_max, psi_min = crystal_system_dictionary[crystal_system]

    # see docstrings for np.arange, np.linspace has better endpoint handling
    steps_theta = int(np.ceil((theta_max - 0) / resolution))
    steps_psi = int(np.ceil((psi_max - psi_min) / resolution))
    theta = np.linspace(0, np.deg2rad(theta_max), num=steps_theta)  # radians as we're about to make spherical polar cordinates
    if equal == 'area':
        # http://mathworld.wolfram.com/SpherePointPicking.html
        v_1 = (1 + np.cos(np.deg2rad(psi_max))) / 2
        v_2 = (1 + np.cos(np.deg2rad(psi_min))) / 2
        v_array = np.linspace(min(v_1, v_2), max(v_1, v_2), num=steps_psi)
        psi = np.arccos(2 * v_array - 1)  # in radians
    elif equal == 'angle':
        # now in radians as we're about to make spherical polar cordinates
        psi = np.linspace(np.deg2rad(psi_min), np.deg2rad(psi_max), num=steps_psi)

    psi_theta = np.asarray(list(product(psi, theta)))
    r = np.ones((psi_theta.shape[0], 1))
    points_in_spherical_polars = np.hstack((r, psi_theta))

    # keep only theta ==0 psi ==0, do this with np.abs(theta) > 0 or psi == 0 - more generally use the smallest psi value
    points_in_spherical_polars = points_in_spherical_polars[np.logical_or(
        np.abs(psi_theta[:, 1]) > 0, psi_theta[:, 0] == np.min(psi_theta[:, 0]))]
    points_in_cartesians = vectorised_spherical_polars_to_cartesians(points_in_spherical_polars)

    if crystal_system == 'cubic':
        # add in the geodesic that runs [1,1,1] to [1,0,1]
        v1 = np.divide([1, 1, 1], np.sqrt(3))
        v2 = np.divide([1, 0, 1], np.sqrt(2))

        def cubic_corner_geodesic(t):
            # https://math.stackexchange.com/questions/1883904/a-time-parameterization-of-geodesics-on-the-sphere
            w = v2 - np.multiply(np.dot(v1, v2), v1)
            w = np.divide(w, np.linalg.norm(w))
            # return in cartesians with t_end = np.arccos(np.dot(v1,v2))
            return np.add(np.multiply(np.cos(t.reshape(-1, 1)), v1), np.multiply(np.sin(t.reshape(-1, 1)), w))

        t_list = np.linspace(0, np.arccos(np.dot(v1, v2)), num=steps_theta)
        geodesic = cubic_corner_geodesic(t_list)
        points_in_cartesians = np.vstack((points_in_cartesians, geodesic))
        # the great circle (from [1,1,1] to [1,0,1]) forms a plane (with the
        # origin), points on the same side as (0,0,1) are safe, the others are not
        plane_normal = np.cross(v2, v1)  # dotting this with (0,0,1) gives a positive number
        points_in_cartesians = points_in_cartesians[np.dot(
            points_in_cartesians, plane_normal) >= 0]  # 0 is the points on the geodesic

    return points_in_cartesians


def beam_directions_to_euler_angles(points_in_cartesians):
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
    axes = np.cross([0, 0, 1], points_in_cartesians)  # in unit cartesians so this is fine, [0,0,1] returns [0,0,0]
    norms = np.linalg.norm(axes, axis=1).reshape(-1, 1)
    angle = np.arcsin(norms)

    normalised_axes = np.ones_like(axes)
    np.divide(axes, norms, out=normalised_axes, where=norms != 0)

    np_axangles = np.hstack((normalised_axes, angle.reshape((-1, 1))))
    eulers = AxAngle(np_axangles).to_Euler(axis_convention='rzxz')
    return eulers
