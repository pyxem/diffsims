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

from PIL import ImageDraw
from PIL import Image
import numpy as np


def create_mask(shape, fill=True):
    """
    Initiate an empty mask
    """
    return np.full(shape, fill, dtype=bool)


def invert_mask(mask):
    """
    Turn True into False and False into True
    """
    mask[:] = np.invert(mask)


def add_polygon_to_mask(mask, coords, fill=False):
    """
    Add a polygon defined by sequential vertex coordinates to the mask.

    Parameters
    ----------
    mask: (H, W) array of dtype bool
        boolean mask for an image
    coords: (N, 2) array
        (x, y) coordinates of vertices
    fill: int, optional
        Fill value. 0 is black (negative, False) and 1 is white (True)

    Returns
    -------
    None:
        the mask is adjusted inplace
    """
    coords = np.array(coords)
    coords = np.ravel(coords, order='C').tolist()
    tempmask = Image.fromarray(mask)
    draw = ImageDraw.Draw(tempmask)
    draw.polygon(coords, fill=fill)
    mask[:] = np.array(tempmask).astype(bool)


def add_circles_to_mask(mask, coords, r, fill=False):
    """
    Add a circle on a mask at each (x, y) coordinate with a radius r

    Parameters
    ----------
    mask: (H, W) array of dtype bool
        boolean mask for an image
    coords: (N, 2) array
        (x, y) coordinates of circle centers
    r: float or (N,) array
        radii of the circles
    fill: int, optional
        Fill value. 0 is black (negative, False) and 1 is white (True)

    Returns
    -------
    None:
        the mask is adjusted inplace
    """
    coords = np.array(coords)
    r = r*np.ones(coords.shape[0])
    for i, j in zip(coords, r):
        add_circle_to_mask(mask, i[0], i[1], j, fill=fill)


def add_circle_to_mask(mask, x, y, r, fill=False):
    """
    Add a single circle to the mask

    Parameters
    ----------
    mask: (H, W) array of dtype bool
        boolean mask for an image
    x: float
        x-coordinate of the circle center in pixels
    y: float
        y-coordinate of the circle center in pixels
    r: float
        radius of the circles in pixels
    fill: int, optional
        Fill value. 0 is black (negative, False) and 1 is white (True)

    Returns
    -------
    None:
        the mask is adjusted inplace
    """
    xx = np.arange(mask.shape[1])
    yy = np.arange(mask.shape[0])
    X, Y = np.meshgrid(xx, yy)
    condition = ((X - x)**2 + (Y - y)**2 < r**2)
    mask[condition] = fill


def add_annulus_to_mask(mask, r1, r2, x=None, y=None, fill=False):
    """
    Add an annular feature on the mask

    Parameters
    ----------
    mask: (H, W) array of dtype bool
        boolean mask for an image
    r1: float
        radius of the inner circle in pixels
    r2: float
        radius of the outer circle in pixels
    x: float
        x-coordinate of the circle center in pixels. Defaults to the center of the mask.
    y: float
        y-coordinate of the circle center in pixels. Defaults to the center of the mask.
    fill: int, optional
        Fill value. 0 is black (block, False) and 1 is white (pass, True)

    Returns
    -------
    None:
        the mask is adjusted inplace
    """
    if x is None:
        x = mask.shape[1] / 2
    if y is None:
        y = mask.shape[0] / 2
    xx = np.arange(mask.shape[1])
    yy = np.arange(mask.shape[0])
    X, Y = np.meshgrid(xx, yy)
    condition = ((X - x)**2 + (Y - y)**2 > r1**2) & ((X - x)**2 + (Y - y)**2 < r2**2)
    mask[condition] = fill


def add_band_to_mask(mask, x, y, theta, width, fill=False):
    """
    Add a straight band to a mask

    Parameters
    ----------
    mask: (H, W) array of dtype bool
        boolean mask for an image
    x: float
        x-coordinate of point that the center of the band must pass through
        in pixels
    y: float
        y-coordinate of point that the center of the band must pass through
        in pixels
    theta: float
        angle in degrees of the band relative to the x-axis
    width: float
        width of the band in pixels
    fill: int, optional
        Fill value. 0 is black (block, False) and 1 is white (pass, True)

    Returns
    -------
    None:
        the mask is adjusted inplace
    """
    # see distance from point to line formula https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    theta_r = np.deg2rad(theta)
    a = np.sin(theta_r)
    b = -np.cos(theta_r)
    c = -(a*x+b*y)
    denom = np.sqrt(a**2 + b**2)
    xx = np.arange(mask.shape[1])
    yy = np.arange(mask.shape[0])
    X, Y = np.meshgrid(xx, yy)
    condition = np.abs(a*X+b*Y+c)/denom < width/2
    mask[condition] = fill
