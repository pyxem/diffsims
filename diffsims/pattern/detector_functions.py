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

from numpy.random import default_rng
from scipy import ndimage as ndi


def constrain_to_dynamic_range(pattern, detector_max=None):
    """Force the values within pattern to lie between [0,detector_max]

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector after corruption
    detector_max : float
        The maximum allowed value at the detector

    Returns
    -------
    within_range_pattern: numpy.ndarray
        The pattern, with values >=0 and =< detector_max
    """
    within_range = pattern.copy()
    within_range[within_range < 0] = 0

    if detector_max is not None:
        within_range[within_range > detector_max] = detector_max

    return within_range


def add_gaussian_point_spread(pattern, sigma):
    """
    Blurs intensities across space with a gaussian function

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector
    sigma : float
        The standard deviation of the gaussian blur, in pixels

    Returns
    -------
    blurred_pattern : numpy.ndarray
        The blurred pattern (deterministic)
    """
    return ndi.gaussian_filter(pattern, sigma)


def add_shot_noise(pattern, seed=None):
    """
    Applies shot noise to a pattern

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector
    seed : int or None
        seed value for the random number generator

    Returns
    -------
    shotted_pattern : numpy.ndarray
        A single sample of the pattern after accounting for shot noise

    Notes
    -----
    This function will (as it should) behave differently depending on the
    pattern intensity, so be mindful to put your intensities in physical units
    """
    rng = default_rng(seed)

    return rng.poisson(pattern)


def add_shot_and_point_spread(pattern, sigma, shot_noise=True, seed=None):
    """
    Adds shot noise (optional) and gaussian point spread (via a convolution) to a pattern

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector
    sigma : float
        The standard deviation of the gaussian blur, in pixels
    shot_noise : bool
        Whether to include shot noise in the original signal, default True
    seed : int or None
        seed value for the random number generator (effects the shot noise only)

    Returns
    -------
    detector_pattern : numpy.ndarray
        A single sample of the pattern after accounting for detector properties

    See also
    --------
    add_shot_noise : adds only shot noise
    add_gaussian_point_spread : adds only point spread
    """

    # shot noise happens before the detector response (operations won't commute)
    if shot_noise:
        pattern = add_shot_noise(pattern, seed)

    pattern = add_gaussian_point_spread(pattern, sigma)

    return pattern


def add_gaussian_noise(pattern, sigma, seed=None):
    """
    Applies gaussian noise at each pixel within the pattern

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector
    sigma : float
        The (absolute) deviation of the gaussian errors
    seed : int or None
        seed value for the random number generator

    Returns
    -------
    corrupted_pattern :
    """
    rng = default_rng(seed)
    pertubations = rng.normal(loc=0, scale=sigma, size=pattern.shape)
    pattern = pattern + pertubations

    return constrain_to_dynamic_range(pattern)


def add_dead_pixels(pattern, n=None, fraction=None, seed=None):
    """
    Adds randomly placed dead pixels onto a pattern

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector
    n : int
        The number of dead pixels, defaults to None
    fraction : float
        The fraction of dead pixels, defaults to None
    seed : int or None
        seed value for the random number generator

    Returns
    -------
    corrupted_pattern : numpy.ndarray
        The pattern, with dead pixels included
    """
    # sorting the n/fraction kwargs

    both_none = n is None and fraction is None
    neither_none = n is not None and fraction is not None

    if both_none or neither_none:
        raise ValueError("Exactly one of 'n' and 'fraction' must be None")

    # converting fraction to n if needs be
    if fraction is not None:
        pattern_size = pattern.shape[0] * pattern.shape[1]
        n = int(fraction * pattern_size)

    rng = default_rng(seed)
    # .astype rounds down, these generate values from 0 to (pattern.shape - 1)
    xdead = rng.uniform(low=0, high=pattern.shape[0], size=n).astype(int)
    ydead = rng.uniform(low=0, high=pattern.shape[1], size=n).astype(int)

    # otherwise pattern will also have 0 elements
    corrupted = pattern.copy()
    corrupted[ydead, xdead] = 0

    return corrupted


def add_linear_detector_gain(pattern, gain):
    """
    Multiplies the pattern by a gain (which is not a function of the pattern)

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector
    gain : float or numpy.ndarray
        Multiplied through the pattern, broadcasting applies
    Returns
    -------
    corrupted_pattern : numpy.ndarray
        The pattern, with gain applied
    """
    return np.multiply(pattern, gain)


def add_detector_offset(pattern, offset):
    """
    Adds/subtracts a fixed offset value from a pattern

    Parameters
    ----------
    pattern : numpy.ndarray
        The diffraction pattern at the detector
    offset : float or numpy.ndarray
        Added through the pattern, broadcasting applies
    Returns
    -------
    corrupted_pattern : np.ndarray
        The pattern, with offset applied, pixels that would have been negative
        are instead 0.
    """
    pattern = np.add(pattern, offset)
    return constrain_to_dynamic_range(pattern)
