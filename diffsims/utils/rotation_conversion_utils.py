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

These utils provide vectorised implementations of conversions as described in the package transforms3d.

Currently avaliable are:

euler2quat
quat2axangle
euler2axangle (via chaining of the above)

axangle2mat
mat2euler
axangle2euler (via chaining of the above)

It also provides two implementations (one vectorised) that convert axis-angles pairs to the correct
angular ranges

Finally - two classes, Euler & AxAngle are provided
"""

import numpy as np
import warnings


def vectorised_euler2quat(eulers, axes='rzxz'):
    """ Applies the transformation that takes eulers to quaternions

    Parameters
    ----------
    eulers : (N,3) numpy array
        [First,second,third] euler angles (in radians)
    axes :
        Euler angles conventions, as detailed in transforms3d. Only 'rzxz' and 'szxz' are
        currently supported

    Returns
    -------
    q : (N,4) numpy array
            Contains elements w,x,y,z

    Notes
    -----
    This function is a port from transforms3d
    """

    ai = eulers[:, 0]
    aj = eulers[:, 1]
    ak = eulers[:, 2]

    _NEXT_AXIS = [1, 2, 0, 1]

    if axes != 'rzxz' and axes != 'szxz':
        raise ValueError()
    elif axes == 'rzxz':
        firstaxis, parity, repetition, frame = 2, 0, 1, 1
    elif axes == 'szxz':
        firstaxis, parity, repetition, frame = 2, 0, 1, 0

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai

    """
    if parity:
         aj = -aj #not currently supported, commented out for coverage
    """

    ai = np.divide(ai, 2.0)
    aj = np.divide(aj, 2.0)
    ak = np.divide(ak, 2.0)
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((ai.shape[0], 4))
    if repetition:
        q[:, 0] = cj * (cc - ss)
        q[:, i] = cj * (cs + sc)
        q[:, j] = sj * (cc + ss)
        q[:, k] = sj * (cs - sc)
    """
    #Not currently supported, commented out for coverage
    else:
        q[:,0] = cj*cc + sj*ss
        q[:,i] = cj*sc - sj*cs
        q[:,j] = cj*ss + sj*cc
        q[:,k] = cj*cs - sj*sc

    if parity:
        q[:,j] *= -1.0
    """
    return q


def vectorised_quat2axangle(q):
    """ Applies the transformation that takes quaternions to axis angles

    Parameters
    ----------
    q : (N,4) numpy array
        Contains elements w,x,y,z

    Returns
    -------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the normalised vector and
        theta is the angle in radians

    Notes
    -----
    This function is a port of the associated function in transforms3d. As there
    for the identity rotation [1,0,0,0] is returned

    """

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Nq = w * w + x * x + y * y + z * z
    if not np.all(np.isfinite(Nq)):
        raise ValueError("You have infinte elements, please check your entry")
    if np.any(Nq < 1e-6):
        raise ValueError("Very small numbers are at risk when we normalise, try normalising your quaternions")

    if np.any(Nq != 1):  # normalize
        s = np.sqrt(Nq)
        w, x, y, z = np.divide(w, s), np.divide(x, s), np.divide(y, s), np.divide(z, s)

    len_img = np.sqrt((x * x) + (y * y) + (z * z))
    # case where the vector is nearly [0,0,0], return axangle [1,0,0,0], q = [1,1,0,0] does it
    x = np.where(len_img == 0, 1, x)
    y = np.where(len_img == 0, 0, y)
    z = np.where(len_img == 0, 0, z)
    w = np.where(len_img == 0, 1, w)

    len_img = np.sqrt((x * x) + (y * y) + (z * z))  # recalculated so we avoid a divide by zero
    xr, yr, zr = np.divide(x, len_img), np.divide(y, len_img), np.divide(z, len_img)

    w[w > 1] = 1
    w[w < -1] = -1
    theta = 2 * np.arccos(w)
    axangles = np.asarray((xr, yr, zr, theta)).T
    return axangles


def vectorised_euler2axangle(eulers, axes='rzxz'):
    """ Applies the transformation that takes eulers to axis-angles

    Parameters
    ----------
    eulers : (N,3) numpy array
        [First,second,third] euler angles (in radians)
    axes :
        Euler angles conventions, as detailed in transforms3d. Only 'rzxz' and 'szxz' are
        currently supported

    Returns
    -------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the normalised vector and
        theta is the angle in radians

    Notes
    -----
    This function is a port of the associated function(s) in transforms3d. As there
    for the identity rotation [1,0,0,0] is returned

    """
    return vectorised_quat2axangle(vectorised_euler2quat(eulers, axes))


def vectorised_axangle2mat(axangles):
    """ Applies the transformation that takes eulers to axis-angles

    Parameters
    ----------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the (ideally normalised) vector and
        theta is the angle in radians

    Returns
    -------
    M : (3,3,N) np.array
        For internal use only

    Notes
    -----
    This function is a port of the associated function in transforms3d

    """
    x, y, z, angle = axangles[:, 0], axangles[:, 1], axangles[:, 2], axangles[:, 3]

    # Normalise for safety, skipping divide by zeros
    n = np.sqrt(x * x + y * y + z * z)
    x = np.where(n != 0, x / n, 0)
    y = np.where(n != 0, y / n, 0)
    z = np.where(n != 0, z / n, 0)

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    M = np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])

    return M


def vectorised_mat2euler(M, axes='rzxz'):
    """
    Applies the transformation to take rotation matricies to euler angles

    Parameters
    ----------
    M : (3,3,N) np.array
        From internal use only, returned by vectorised_axangle2mat
    axes: string
        Compliant convention string, only 'rzxz' and 'szxz' are currently accepted

    Returns
    -------
    eulers : (N,4) numpy array
        Contains elements ai,aj,ak where i,j,k are determined by axes

    Notes
    -----
    This function is a port from transforms3d
    """
    _NEXT_AXIS = [1, 2, 0, 1]

    if axes != 'rzxz' and axes != 'szxz':
        raise ValueError()
    elif axes == 'rzxz':
        firstaxis, parity, repetition, frame = 2, 0, 1, 1
    elif axes == 'szxz':
        firstaxis, parity, repetition, frame = 2, 0, 1, 0

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if repetition:
        sy = np.sqrt(M[i, j, :] * M[i, j, :] + M[i, k, :] * M[i, k, :])

        ax = np.where(sy > 0, np.arctan2(M[i, j, :], M[i, k, :]), np.arctan2(-M[j, k, :], M[j, j, :]))
        ay = np.arctan2(sy, M[i, i, :])
        az = np.where(sy > 0, np.arctan2(M[j, i, :], -M[k, i, :]), 0.0)

    """if parity:
        ax, ay, az = -ax, -ay, -az #not currently supported"""
    if frame:
        ax, az = az, ax

    euler = np.vstack((ax, ay, az)).T
    return euler


def vectorised_axangle2euler(axangles, axes='rzxz'):
    """ Applies the transformation that takes eulers to axis-angles

    Parameters
    ----------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the normalised vector and
        theta is the angle in radians
    axes: string
        Compliant convention string, only 'rzxz' and 'szxz' are currently accepted

    Returns
    -------
    eulers : (N,4) numpy array
        Contains elements ai,aj,ak where i,j,k are determined by axes

    Notes
    -----
    This function is a port of the associated function(s) in transforms3d.
    """
    return vectorised_mat2euler(vectorised_axangle2mat(axangles), axes)


def convert_axangle_to_correct_range(vector, angle):
    """
    This repo uses axis-angle pairs between (0,pi) - however often wider
    ranges are used, most common are (0,2pi) and (-pi,pi), this function corrects
    for these

    Parameters
    ----------
    vector : iterable of length 3
        [x,y,z] of the axis
    angle : float
        in radians

    Returns
    -------
    vector, angle :
        correct forms of the inputs

    See Also
    --------
    vectorised_axangle_to_correct_range : for fast processing of bigger inputs

    """
    if (angle >= 0) and (angle < np.pi):  # input in the desired convention
        pass
    elif (angle >= -np.pi) and (angle < 0):
        vector = np.multiply(vector, -1)
        angle = angle * -1
    elif (angle >= np.pi):
        vector = np.multiply(vector, -1)
        angle = 2 * np.pi - angle

    return vector, angle


def vectorised_axangle_to_correct_range(data):
    """
    This repo uses axis-angle pairs between (0,pi) - however often wider
    ranges are used, most common are (0,2pi) and (-pi,pi), this function corrects
    for these

    Parameters
    ----------
    data : (N,4)
        axangles

    Returns
    -------
    data : (N,4)
        Corrected forms of the input

    See Also
    --------
    convert_axangle_to_correct_range : for single data items
    """

    z = data.copy()

    # second clause in unvectorised
    second_case_truth = np.logical_and(z[:, 3] >= -np.pi, z[:, 3] < 0)
    for i in [0, 1, 2, 3]:
        z[:, i] = np.where(second_case_truth, -z[:, i], z[:, i])

    # third clause in unvectorised
    third_case_truth = np.logical_and(z[:, 3] >= np.pi, z[:, 3] <= 2 * np.pi)
    for i in [0, 1, 2]:  # third clause part 1
        z[:, i] = np.where(third_case_truth, -z[:, i], z[:, i])
    z[:, 3] = np.where(third_case_truth, 2 * np.pi - z[:, 3], z[:, 3])  # third clause part 2

    return z


def convert_identity_rotations(data):
    """ Turns 0 angles axangles to [1,0,0,0] """
    data[:, 0][data[:, 3] == 0] = 1
    data[:, 1][data[:, 3] == 0] = 0
    data[:, 2][data[:, 3] == 0] = 0
    return data


class AxAngle():
    """
    Class storing rotations in the axis-angle convention. Each row reads
    as [vx,vy,vz,theta], where [vx,vy,vz] is the rotation axis (normalised)
    and theta is the rotation angle in radians in range (0,pi]
    """

    def __init__(self, data):
        self.data = data.astype('float')
        self.data = vectorised_axangle_to_correct_range(self.data)
        self.data = convert_identity_rotations(self.data)
        self._check_data()
        return None

    def _check_data(self):
        """ Checks the data within AxAngle is acceptable, having the correct dimensions,
        acceptable angles and normalised vectors """
        if self.data.shape[1] != 4:
            raise ValueError("Your data is not in the correct shape")
        if np.any(self.data[:, 3] < 0) or np.any(self.data[:, 3] > np.pi):
            raise ValueError("Some of your angles lie outside of the range (0,pi)")
        if not np.allclose(np.linalg.norm(self.data[:, :3][self.data[:, 3] > 0], axis=1), 1):
            raise ValueError("You no longer have normalised direction vectors")
        return None

    def remove_large_rotations(self, threshold_angle):
        """
        Removes rotations that above a threshold angle

        Parameters
        ----------
        thereshold_angle : float
            angle in radians, rotations larger than this are removed

        Returns
        -------
        None :
            This functions operates in place
        """
        self._check_data()
        self.data = self.data[self.data[:, 3] < threshold_angle]
        return None

    def remove_with_mask(self, mask):
        """
        Removes rotations using a mask

        Parameters
        ----------
        mask : np.array

        Returns
        -------
        None :
            This functions operates in place
        """
        self._check_data()
        self.data = self.data[mask]
        return None

    def to_Euler(self, axis_convention):
        """
        Produces euler angles from the axis-angle pairs.

        Parameters
        ----------
        axis_convention : str
            transforms3d compliant euler string

        Returns
        -------
        out_eulers : diffsims.Euler
        """
        self._check_data()
        eulers = vectorised_axangle2euler(self.data, axis_convention)
        eulers = np.rad2deg(eulers)
        return Euler(eulers, axis_convention)

    def to_Quat(self):
        """ A lightweight port of transforms3d functionality, vectorised"""
        self._check_data()  # means that our vectors need not be checked for normalisation
        vector = self.data[:, :3]
        t2 = self.data[:, 3] / 2.0
        st2 = np.sin(t2)
        w = np.cos(t2).reshape(self.data.shape[0], 1)
        xyz = np.multiply(vector, st2.reshape(self.data.shape[0], 1))
        return np.hstack((w, xyz))

    @classmethod
    def from_Quat(cls, data):
        axangles = vectorised_quat2axangle(data)
        return AxAngle(axangles)


class Euler():
    """
    Class storing rotations as euler angles.
    Each row reads as [alpha,beta,gamma], where alpha, beta and gamma are rotations
    in degrees around the axes specified by Euler.axis_conventions
    as defined in transforms3d. Please always remember that Euler angles are difficult.
    """

    def __init__(self, data, axis_convention='rzxz'):
        self.data = data.astype('float')
        self.axis_convention = axis_convention
        self._check_data()
        return None

    def _check_data(self):
        """ Checks data within Euler is acceptable, to be used at the start
        of methods """
        if self.data.shape[1] != 3:
            raise ValueError("Your data is not in the correct shape")
        if np.any(self.data[:] > 360):
            raise ValueError("Some of your angles are greater 360")
        if np.all(np.abs(self.data[:]) < 2 * np.pi):
            warnings.warn("Your angles all seem quite small, are you sure you're not in radians?")

        return None

    def to_AxAngle(self):
        """ Converts an Euler object to an AxAngle object

        Returns
        -------
        axangle : diffsims.AxAngle object
        """
        self._check_data()
        self.data = np.deg2rad(self.data)  # for the transform operation

        if self.axis_convention == 'rzxz' or self.axis_convention == 'szxz':
            stored_axangle = vectorised_euler2axangle(self.data, axes=self.axis_convention)
            stored_axangle = vectorised_axangle_to_correct_range(stored_axangle)

        else:
            # This is very slow
            from transforms3d.euler import euler2axangle
            stored_axangle = np.ones((self.data.shape[0], 4))
            for i, row in enumerate(self.data):
                temp_vect, temp_angle = euler2axangle(row[0], row[1], row[2], self.axis_convention)
                temp_vect, temp_angle = convert_axangle_to_correct_range(temp_vect, temp_angle)
                for j in [0, 1, 2]:
                    stored_axangle[i, j] = temp_vect[j]
                    stored_axangle[i, 3] = temp_angle  # in radians!

        self.data = np.rad2deg(self.data)  # leaves our eulers safe and sound
        return AxAngle(stored_axangle)

    def to_rotation_list(self, round_to=2):
        """
        Parameters:

        round_to : int or None
            The number of decimal places to keep on the rotation, if None no rounding is performed
        """

        if round_to is not None:
            round = np.round(self.data, 2)
        else:
            round = self.data

        starter_list = round.tolist()
        tuple_list = [tuple(x) for x in starter_list]
        return tuple_list
