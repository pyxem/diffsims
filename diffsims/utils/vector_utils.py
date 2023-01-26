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
import math


def get_angle_cartesian_vec(a, b):
    """Compute the angles between two lists of vectors in a cartesian
    coordinate system.

    Parameters
    ----------
    a, b : np.array()
        The two lists of directions to compute the angle between in Nx3 float
        arrays.

    Returns
    -------
    angles : np.array()
        List of angles between `a` and `b` in radians.
    """
    if a.shape != b.shape:
        raise ValueError(
            "The shape of a {} and b {} must be the same.".format(a.shape, b.shape)
        )

    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denom_nonzero = denom != 0.0
    angles = np.zeros(a.shape[0])
    angles[denom_nonzero] = np.arccos(
        np.clip(
            np.sum(a[denom_nonzero] * b[denom_nonzero], axis=-1) / denom[denom_nonzero],
            -1.0,
            1.0,
        )
    ).ravel()
    return angles


def get_angle_cartesian(a, b):
    """Compute the angle between two vectors in a cartesian coordinate system.

    Parameters
    ----------
    a, b : array-like with 3 floats
        The two directions to compute the angle between.

    Returns
    -------
    angle : float
        Angle between `a` and `b` in radians.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, np.dot(a, b) / denom)))


def vectorised_spherical_polars_to_cartesians(z):
    """
    Converts an array of spherical polars into an array of
    (x,y,z) = r(cos(psi)sin(theta),sin(psi)sin(theta),cos(theta))

    Parameters
    ----------
    z : np.array
        With rows of
        r : the radius value, r = sqrt(x**2+y**2+z**2)
        psi : The azimuthal angle generally (0,2pi])
        theta : The elevation angle generally (0,pi)

    Returns
    -------
    xyz : np.array
        With rows of
        x,y,z
    """
    r, psi, theta = z[:, 0], z[:, 1], z[:, 2]
    x = r * np.cos(psi) * np.sin(theta)
    y = r * np.sin(psi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.asarray([x, y, z]).T
