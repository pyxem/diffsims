# -*- coding: utf-8 -*-
# Copyright 2017-2024 The diffsims developers
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

from typing import Tuple

import numpy as np
from numba import jit

from numpy.random import default_rng
from scipy import ndimage as ndi


__all__ = [
    "add_dead_pixels",
    "add_detector_offset",
    "add_gaussian_noise",
    "add_gaussian_point_spread",
    "add_linear_detector_gain",
    "add_shot_noise",
    "add_shot_and_point_spread",
    "constrain_to_dynamic_range",
    "get_pattern_from_pixel_coordinates_and_intensities",
]


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


def get_pattern_from_pixel_coordinates_and_intensities(
    coordinates: np.ndarray,
    intensities: np.ndarray,
    shape: Tuple[int, int],
    sigma: float,
    clip_threshold: float = 1,
) -> np.ndarray:
    """Generate a diffraction pattern from spot pixel-coordinates and intensities,
    using a gaussian blur.
    This is subpixel-precise, meaning the coordinates can be floats.
    Values less than `clip_threshold` are rounded down to 0 to simplify computation.

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinates of reflections, in pixels. Shape (n, 2) or (n, 3). Can be floats
    intensities : np.ndarray
        Intensities of each reflection. Must have same same first dimension as `coordinates`
    shape : tuple[int, int]
        Output shape
    sigma : float
        For Gaussian blur
    intensity_scale : float
        Scale to multiply the final diffraction pattern with

    Returns
    -------
    np.ndarray
        dtype int

    Notes
    -----
    Not all values below the clipping threshold are ignored.
    The threshold is used to estimate a radius (box) around each reflection where the pixel intensity is greater than the threshold.
    As the radius is rounded up and as the box is square rather than circular, some values below the threshold can be included.

    When using float coordinates, the intensity is spread as if the edge was not there.
    This is in line with what should be expected from a beam on the edge of the detector, as part of the beam is simply outside the detector area.
    However, when using integer coordinates, the total intensity is preserved for the pixels in the pattern.
    This means that the intensity contribution from parts of the beam which would hit outside the detector are now kept in the pattern.
    Thus, reflections wich are partially outside the detector will have higher intensities than expected, when using integer coordinates.
    """
    if np.issubdtype(coordinates.dtype, np.integer):
        # Much simpler with integer coordinates
        coordinates = coordinates.astype(int)
        out = np.zeros(shape)
        # coordinates are xy(z), out array indices are yx.
        out[coordinates[:, 1], coordinates[:, 0]] = intensities
        out = add_shot_and_point_spread(out, sigma, shot_noise=False)
        return out

    # coordinates of each pixel in the output, such that the final axis is yx coordinates
    inds = np.transpose(np.indices(shape), (1, 2, 0))
    return _subpixel_gaussian(
        coordinates,
        intensities,
        inds,
        shape,
        sigma,
        clip_threshold,
    )


@jit(
    nopython=True
)  # Not parallel, we might get a race condition with overlapping spots
def _subpixel_gaussian(
    coordinates: np.ndarray,
    intensities: np.ndarray,
    inds: np.ndarray,
    shape: Tuple[int, int],
    sigma: float,
    clip_threshold: float = 1,
) -> np.ndarray:
    out = np.zeros(shape)

    # Pre-calculate the constants
    prefactor = 1 / (2 * np.pi * sigma**2)
    exp_prefactor = -1 / (2 * sigma**2)

    for i in range(intensities.size):
        # Reverse since coords are xy, but indices are yx
        coord = coordinates[i][:2][::-1]
        intens = intensities[i]

        # The gaussian is expensive to evaluate for all pixels and spots.
        # Therefore, we limit the calculations to a box around each reflection where the intensity is above a threshold.
        # Formula found by inverting the gaussian
        radius = np.sqrt(np.log(clip_threshold / (prefactor * intens)) / exp_prefactor)

        if np.isnan(radius):
            continue
        slic = (
            slice(
                max(0, int(np.ceil(coord[0] - radius))),
                min(shape[0], int(np.floor(coord[0] + radius + 1))),
            ),
            slice(
                max(0, int(np.ceil(coord[1] - radius))),
                min(shape[1], int(np.floor(coord[1] + radius + 1))),
            ),
        )
        # Calculate the values of the Gaussian manually
        out[slic] += (
            intens
            * prefactor
            * np.exp(exp_prefactor * np.sum((inds[slic] - coord) ** 2, axis=-1))
        )
    return out
