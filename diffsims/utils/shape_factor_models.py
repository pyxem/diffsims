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

    return 1 - excitation_error / max_excitation_error


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

    num = np.sin(np.pi * minima_number * excitation_error / max_excitation_error)
    denom = excitation_error
    return np.abs(np.divide(num, denom, out=np.zeros_like(num), where=denom != 0))
