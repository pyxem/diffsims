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
Helper functions for gridding
"""

import numpy as np
from itertools import product
from transforms3d.euler import axangle2euler, euler2axangle, euler2mat
from transforms3d.quaternions import quat2axangle, axangle2quat, mat2quat, qmult
from diffsims.utils.rotation_conversion_utils import *
from diffsims.utils.vector_utils import vectorised_spherical_polars_to_cartesians

crystal_system_dictionary = {'cubic':[45,54.7,0],
 'hexagonal':[45,90,26.565],
 'tetragonal':[45,90,0],
 'orthorhombic':[90.90,0],
 'trigonal': [45,90,-116.5],
 'monoclinic':[90,0,-90],
 'none':[360,180,0]}


def vectorised_qmult(q1, qdata):
    """ A vectorised implementation that multiplies qdata (array) by q1 (single quaternion) """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = qdata[:,0],qdata[:,1],qdata[:,2],qdata[:,3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z]).T

def _get_rotation_to_beam_direction(beam_direction):
    """ A helper function for getting rotations around a beam direction, the
    returns the first two angles (szxz) needed to place the viewer looking down the
    given zone axis.

    Parameters
    ----------
    beam_direction : [vx,vy,vz]

    Returns
    -------
    alpha,beta : angles in degrees

    See Also
    --------
    generators.get_grid_around_beam_direction
    """
    from transforms3d.euler import axangle2euler
    beam_direction = np.divide(beam_direction,np.linalg.norm(beam_direction))
    axis = np.cross(beam_direction,[0,0,1]) # [0,0,1] is the starting direction for diffsims
    angle = np.arcsin(np.linalg.norm(axis))
    alpha,beta,gamma = axangle2euler(axis,angle,'szxz')
    return np.rad2deg(alpha),np.rad2deg(beta)



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
    Axangles :
        Axangles in the correct class
    new_center : (alpha,beta,gamma)
        The location of the (0,0,0) rotation as an rzxz euler angle

    Returns
    -------
    AxAngles :

    See Also
    --------
    generators.get_local_grid
    """

    quats = Axangles.to_Quat()
    q = mat2quat(rotation_matrix_from_euler_angles((new_center)))
    stored_quats = vectorised_qmult(q,quats)

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
    return _create_advanced_linearly_spaced_array_in_rzxz(resolution,max_alpha=360,max_beta=180,max_gamma=360)


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
    steps_alpha = int(np.ceil((max_alpha - 0)/resolution)) #see docstrings for np.arange, np.linspace has better endpoint handling
    steps_beta  = int(np.ceil((max_beta  - 0)/resolution))
    steps_gamma = int(np.ceil((max_gamma - 0)/resolution))

    alpha = np.linspace(0, max_alpha, num=steps_alpha, endpoint=False)
    beta = np.linspace(0, max_beta, num=steps_beta, endpoint=False)
    gamma = np.linspace(0, max_gamma, num=steps_gamma, endpoint=False)
    z = np.asarray(list(product(alpha, beta, gamma)))
    return Euler(z, axis_convention='rzxz')

def get_beam_directions(resolution,crystal_system,equal='angle'):
    """
    """
    theta_max,psi_max,psi_min = crystal_system_dictionary[crystal_system]

    if equal == 'area':
        # http://mathworld.wolfram.com/SpherePointPicking.html
        # Need to do sensible point counting for this
        raise NotImplementedError("Use equal='angle' instead")
    else:
        steps_theta = int(np.ceil((theta_max - 0)/resolution)) #see docstrings for np.arange, np.linspace has better endpoint handling
        steps_psi   = int(np.ceil((psi_max - psi_max)/resolution))
        theta = np.linspace(0,theta_max,num=steps_theta)
        psi   = np.linspace(psi_min,psi_max,num=steps_theta)

    psi_theta = np.asarray(list(product(psi,theta)))
    r = np.ones((psi_theta.shape[0],1))
    points_in_spherical_polars = np.hstack((r,psi_theta))

    if crystal_system == 'cubic':
        # reject points below the geodesic
        # add points on the geodesic
        pass

    points_in_cartesians = vectorised_spherical_polars_to_cartesians(points_in_spherical_polars)
    axis = np.cross([0,0,1],points_in_cartesians) #in unit cartesians so this is fine, [0,0,1] returns [0,0,0]
    angle = np.arcsin(np.linalg.norm(axis))
    eulers = AxAngle(np.hstack(axis,angle)).to_Euler(axis_convention='rzxz')
    return eulers
