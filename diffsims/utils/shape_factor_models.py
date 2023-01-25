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


def binary(excitation_error, max_excitation_error):
    """
    Returns a unit intensity for all reflections

    Parameters
    ----------
    excitation_error : array-like or float
        The distance (reciprocal) from a reflection to the Ewald sphere

    max_excitation_error : float
        The distance at which a reflection becomes extinct

    Returns
    -------
    intensities : array-like or float
    """
    return 1


def linear(excitation_error, max_excitation_error):
    """
    Returns an intensity linearly scaled with by the excitation error

    Parameters
    ----------
    excitation_error : array-like or float
        The distance (reciprocal) from a reflection to the Ewald sphere

    max_excitation_error : float
        The distance at which a reflection becomes extinct

    Returns
    -------
    intensities : array-like or float
    """
    sf = 1 - np.abs(excitation_error) / max_excitation_error
    if isinstance(excitation_error, np.ndarray):
        sf[sf < 0.0] = 0.0
    else:
        sf = max(sf, 0.)
    return sf


def sinc(excitation_error, max_excitation_error, minima_number=5):
    """
    Returns an intensity with a sinc profile

    Parameters
    ----------
    excitation_error : array-like or float
        The distance (reciprocal) from a reflection to the Ewald sphere

    max_excitation_error : float
        The distance at which a reflection becomes extinct

    minima_number : int
        The minima_number'th minima lies at max_excitation_error from 0

    Returns
    -------
    intensity : array-like or float
    """
    fac = np.pi * minima_number / max_excitation_error
    num = np.sin(fac * excitation_error)
    denom = fac * excitation_error
    return np.nan_to_num(
        np.abs(np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)),
        nan=1,
    )


def sin2c(excitation_error, max_excitation_error, minima_number=5):
    """
    Intensity with sin^2(s)/s^2 profile, after Howie-Whelan rel-rod

    Parameters
    ----------
    excitation_error : array-like or float
        The distance (reciprocal) from a reflection to the Ewald sphere

    max_excitation_error : float
        The distance at which a reflection becomes extinct

    minima_number : int
        The minima_number'th minima lies at max_excitation_error from 0

    Returns
    -------
    intensity : array-like or float
    """
    return sinc(excitation_error, max_excitation_error, minima_number) ** 2


def atanc(excitation_error, max_excitation_error, minima_number=5):
    """
    Intensity with arctan(s)/s profile that closely follows sin(s)/s but
    is smooth for s!=0.

    Parameters
    ----------
    excitation_error : array-like or float
        The distance (reciprocal) from a reflection to the Ewald sphere

    max_excitation_error : float
        The distance at which a reflection becomes extinct

    minima_number : int
        The minima_number'th minima in the corresponding sinx/x lies at
        max_excitation_error from 0

    Returns
    -------
    intensity : array-like or float
    """
    fac = np.pi * minima_number / np.abs(max_excitation_error)
    return np.nan_to_num(
        np.arctan(fac * excitation_error) / (fac * excitation_error),
        nan=1,
    )


def lorentzian(excitation_error, max_excitation_error):
    """
    Lorentzian intensity profile that should approximate
    the two-beam rocking curve. This is equation (6) in reference [1].

    Parameters
    ----------
    excitation_error : array-like or float
        The distance (reciprocal) from a reflection to the Ewald sphere

    max_excitation_error : float
        The distance at which a reflection becomes extinct

    Returns
    -------
    intensity_factor : array-like or float
        Vector representing the rel-rod factor for each reflection

    References
    ----------
    [1] L. Palatinus, P. Brázda, M. Jelínek, J. Hrdá, G. Steciuk, M. Klementová, Specifics of the data processing of precession electron diffraction tomography data and their implementation in the program PETS2.0, Acta Crystallogr. Sect. B Struct. Sci. Cryst. Eng. Mater. 75 (2019) 512–522. doi:10.1107/S2052520619007534.
    """
    # in the paper, sigma = pi*thickness.
    # We assume thickness = 1/max_exitation_error
    sigma = np.pi / max_excitation_error
    fac = (
        sigma
        / (np.pi * (sigma ** 2 * excitation_error ** 2 + 1))
        * max_excitation_error
    )
    return fac


def lorentzian_precession(
    excitation_error, max_excitation_error, r_spot, precession_angle
):
    """
    Intensity profile factor for a precessed beam assuming a Lorentzian
    intensity profile for the un-precessed beam. This is equation (10) in
    reference [1].

    Parameters
    ----------
    excitation_error : array-like or float
        The distance (reciprocal) from a reflection to the Ewald sphere

    max_excitation_error : float
        The distance at which a reflection becomes extinct

    r_spot : array-like or float
        The distance (reciprocal) from each reflection to the origin

    precession_angle : float
        The beam precession angle in radians; the angle the beam makes
        with the optical axis.

    Returns
    -------
    intensity_factor : array-like or float
        Vector representing the rel-rod factor for each reflection

    References
    ----------
    [1] L. Palatinus, P. Brázda, M. Jelínek, J. Hrdá, G. Steciuk, M. Klementová, Specifics of the data processing of precession electron diffraction tomography data and their implementation in the program PETS2.0, Acta Crystallogr. Sect. B Struct. Sci. Cryst. Eng. Mater. 75 (2019) 512–522. doi:10.1107/S2052520619007534.
    """
    sigma = np.pi / max_excitation_error
    u = sigma ** 2 * (r_spot ** 2 * precession_angle ** 2 - excitation_error ** 2) + 1
    z = np.sqrt(u ** 2 + 4 * sigma ** 2 * excitation_error ** 2)
    fac = (sigma / np.pi) * np.sqrt(2 * (u + z) / z ** 2)
    return fac
