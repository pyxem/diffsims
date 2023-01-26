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
Back end for computing diffraction patterns with a kinematic model.
"""

from diffsims.utils.discretise_utils import get_discretisation
from numpy import array, pi, sin, cos, empty
from scipy.interpolate import interpn
from diffsims.utils.fourier_transform import (
    get_DFT,
    to_recip,
    fftshift_phase,
    plan_fft,
    fast_abs,
)
from diffsims.utils.generic_utils import to_mesh


def normalise(arr):
    return arr / arr.max()


def get_diffraction_image(
    coordinates,
    species,
    probe,
    x,
    wavelength,
    precession,
    GPU=True,
    pointwise=False,
    **kwargs
):
    """
    Return kinematically simulated diffraction pattern

    Parameters
    ----------
    coordinates : `numpy.ndarray` [`float`],  (n_atoms, 3)
        List of atomic coordinates
    species : `numpy.ndarray` [`int`],  (n_atoms,)
        List of atomic numbers
    probe : `diffsims.ProbeFunction`
        Function representing 3D shape of beam
    x : `list` [`numpy.ndarray` [`float`] ], of shapes [(nx,), (ny,), (nz,)]
        Mesh on which to compute the volume density
    wavelength : `float`
        Wavelength of electron beam
    precession : a pair (`float`, `int`)
        The float dictates the angle of precession and the int how many points are
        used to discretise the integration.
    dtype : (`str`, `str`)
        tuple of floating/complex datatypes to cast outputs to
    ZERO : `float` > 0, optional
        Rounding error permitted in computation of atomic density. This value is
        the smallest value rounded to 0.
    GPU : `bool`, optional
        Flag whether to use GPU or CPU discretisation. Default (if available) is True
    pointwise : `bool`, optional
        Optional parameter whether atomic intensities are computed point-wise at
        the centre of a voxel or an integral over the voxel. default=False

    Returns
    -------
    DP : `numpy.ndarray` [`dtype[0]`], (nx, ny, nz)
        The two-dimensional diffraction pattern evaluated on the reciprocal grid
        corresponding to the first two vectors of `x`.
    """
    FTYPE = kwargs["dtype"][0]
    kwargs["GPU"] = GPU
    kwargs["pointwise"] = pointwise

    x = [X.astype(FTYPE, copy=False) for X in x]
    y = to_recip(x)
    if wavelength == 0:
        p = probe(x).mean(-1)
        vol = get_discretisation(coordinates, species, x[:2], **kwargs)[..., 0]
        ft = get_DFT(x[:-1], y[:-1])[0]
    else:
        p = probe(x)
        vol = get_discretisation(coordinates, species, x, **kwargs)
        ft = get_DFT(x, y)[0]

    if precession[0] == 0:
        arr = ft(vol * p)
        arr = fast_abs(arr, arr).real ** 2
        if wavelength == 0:
            return normalise(arr)
        else:
            return normalise(grid2sphere(arr, y, None, 2 * pi / wavelength))

    R = [
        precess_mat(precession[0], i * 360 / precession[1])
        for i in range(precession[1])
    ]

    if wavelength == 0:
        return normalise(
            sum(
                get_diffraction_image(
                    coordinates.dot(r), species, probe, x, wavelength, (0, 1), **kwargs
                )
                for r in R
            )
        )

    fftshift_phase(vol)  # removes need for fftshift after fft
    buf = empty(vol.shape, dtype=FTYPE)
    ft, buf = plan_fft(buf, overwrite=True, planner=1)
    DP = None
    for r in R:
        probe(to_mesh(x, r.T, dtype=FTYPE), out=buf, scale=vol)  # buf = bess*vol

        # Do convolution
        newFT = ft()
        newFT = fast_abs(newFT, buf).real
        newFT *= newFT  # newFT = abs(newFT) ** 2
        newFT = grid2sphere(newFT.real, y, list(r), 2 * pi / wavelength)

        if DP is None:
            DP = newFT
        else:
            DP += newFT

    return normalise(DP.astype(FTYPE, copy=False))


def precess_mat(alpha, theta):
    """
    Generates rotation matrices for precession curves.

    Parameters
    ----------
    alpha : `float`
        Angle (in degrees) of precession tilt
    theta : `float`
        Angle (in degrees) along precession curve

    Returns
    -------
    R : `numpy.ndarray` [`float`], (3, 3)
        Rotation matrix associated to the tilt of `alpha` away from the vertical
        axis and a rotation of `theta` about the vertical axis.
    """
    if alpha == 0:
        return array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    alpha, theta = alpha * pi / 180, theta * pi / 180
    R_a = array([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])
    R_t = array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    R = R_t.T.dot(R_a.dot(R_t))

    return R


def grid2sphere(arr, x, dx, C):
    """
    Projects 3d array onto a sphere

    Parameters
    ----------
    arr : np.ndarray [`float`], (nx, ny, nz)
        Input function to be projected
    x : list [np.ndarray [float]], of shapes [(nx,), (ny,), (nz,)]
        Vectors defining mesh of <arr>
    dx : list [np.ndarray [float]], of shapes [(3,), (3,), (3,)]
        Basis in which to orient sphere. Centre of sphere will be at `C*dx[2]`
        and mesh of output array will be defined by the first two vectors
    C : float
        Radius of sphere

    Returns
    -------
    out : np.ndarray [float], (nx, ny)
        If y is the point on the line between `i*dx[0]+j*dx[1]` and
        `C*dx[2]` which also lies on the sphere of radius `C` from
        `C*dx[2]` then: `out[i,j] = arr(y)`.
        Interpolation on arr is linear.
    """
    if C in (None, 0) or x[2].size == 1:
        if arr.ndim == 2:
            return arr
        elif arr.shape[2] == 1:
            return arr[:, :, 0]

    y = to_mesh((x[0], x[1], array([0])), dx).reshape(-1, 3)

    if C is not None:  # project on line to centre
        w = 1 / (1 + (y ** 2).sum(-1) / C ** 2)
        y *= w[:, None]
        if dx is None:
            y[:, 2] = C * (1 - w)
        else:
            y += C * (1 - w)[:, None] * dx[2]

    out = interpn(x, arr, y, method="linear", bounds_error=False, fill_value=0)

    return out.reshape(x[0].size, x[1].size)
