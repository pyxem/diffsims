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


from scipy import ndimage as ndi

def add_gaussian_blur(pattern,sigma):
    """

    Parameters
    ----------
    pattern : np.array

    sigma : float
          The standard deviation of the gaussian blur, in pixels
    Returns
    -------
    blurred_pattern : np.array

    Notes
    -----
    This blurring, as well as being inuitive is often a good first model
    of a detector response function [source]

    References
    ----------
    [source]
    """
    return ndi.gaussian_filter(pattern, sigma)

def add_shot_noise():
    pass
