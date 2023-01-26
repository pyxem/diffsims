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

from diffsims.utils import mask_utils as mu
import numpy as np


def test_create_mask():
    mask = mu.create_mask((20, 10))
    assert mask.shape[0] == 20
    assert mask.shape[1] == 10


def test_invert_mask():
    mask = mu.create_mask((20, 10))
    initial = mask[0,0]
    mu.invert_mask(mask)
    assert initial != mask[0,0]


def test_add_polygon():
    mask = mu.create_mask((20, 10))
    coords = np.array([[5, 5],[15, 5],[10,10]])
    mu.add_polygon_to_mask(mask, coords)


def test_add_circles_to_mask():
    mask = mu.create_mask((20, 10))
    coords = np.array([[5, 5],[15, 5],[10,10]])
    mu.add_circles_to_mask(mask, coords, 3)
    

def test_add_circle_to_mask():
    mask = mu.create_mask((20, 10))
    mu.add_circle_to_mask(mask, 5, 5, 5)


def test_add_annulus_to_mask():
    mask = mu.create_mask((20, 10))
    mu.add_annulus_to_mask(mask, 4, 7)


def test_add_band_to_mask():
    mask = mu.create_mask((20, 10))
    mu.add_band_to_mask(mask, 4, 7, 10, 4)
