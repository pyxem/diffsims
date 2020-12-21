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

from numpy.random import default_rng
from scipy import ndimage as ndi

def add_gaussian_blur(pattern,sigma):
    """
    Blurs intensities with a gaussian function

    Parameters
    ----------
    pattern : np.array
        The diffraction pattern at the detector
    sigma : float
        The standard deviation of the gaussian blur, in pixels

    Returns
    -------
    blurred_pattern : np.array
        The blurred pattern (deterministic)
    """
    return ndi.gaussian_filter(pattern, sigma)

def add_shot_noise(pattern,seed=None):
    """
    Applies shot noise to a pattern

    Parameters
    ----------
    pattern : np.array
        The diffraction pattern at the detector
    seed : int or None
        seed value for the random number generator

    Returns
    -------
    shotted_pattern : np.array
        A single sample of the pattern after accounting for shot noise

    Notes
    -----
    This function will (as it should) behave differently depending on the
    pattern intensity, so be mindful to put your intensities in physical units
    """
    if seed is not None:
        rng = default_rng(seed)
    else:
        rng = default_rng()

    return rng.poisson(z)
