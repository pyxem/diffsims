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

import pytest
import numpy as np
from diffsims.pattern.detector_functions import (
    constrain_to_dynamic_range,
    add_shot_and_point_spread,
    add_shot_noise,
    add_gaussian_noise,
    add_dead_pixels,
    add_linear_detector_gain,
    add_detector_offset,
)


@pytest.fixture()
def pattern():
    z = np.zeros((128, 128))
    z[40:44, 50:53] = 1e5
    return z


def test_constrain_to_dynamic_range(pattern):
    max = np.max(pattern)
    dmax = int(max / 3)
    z = constrain_to_dynamic_range(pattern, detector_max=dmax)
    assert np.max(z) == dmax
    assert not np.may_share_memory(pattern, z)


class TestReturnsareCopies:
    """ We want pattern to remain untouched by the noise addition """

    def test_copy_shot_and_point_spread(self, pattern):
        """ Also covers shot/point independantly """
        z = add_shot_and_point_spread(pattern, 2, shot_noise=True)
        assert not np.may_share_memory(pattern, z)

    def test_copy_gaussian(self, pattern):
        z = add_gaussian_noise(pattern, 2)
        assert not np.may_share_memory(pattern, z)

    def test_copy_gain_and_offset(self, pattern):
        zgain = add_linear_detector_gain(pattern, 1.1)
        zoff = add_detector_offset(pattern, 0.1)
        assert not np.may_share_memory(pattern, zgain)
        assert not np.may_share_memory(pattern, zoff)

    def test_copy_deadpixel(self, pattern):
        pattern = pattern + 1  # so that we can detect dead pixels
        z = add_dead_pixels(pattern, n=6, seed=7)
        assert not np.may_share_memory(pattern, z)


class TestShotAndPointSpread:
    def test_add_shot_and_point_spread(self, pattern):
        z = add_shot_and_point_spread(pattern, 0, shot_noise=False)
        assert np.allclose(z, pattern)
        # seed testing so we can go through shot_noise = True
        z1a = add_shot_and_point_spread(pattern, 2, shot_noise=True, seed=7)
        z1b = add_shot_and_point_spread(pattern, 2, shot_noise=True, seed=7)
        z2 = add_shot_and_point_spread(pattern, 2, shot_noise=True, seed=8)
        assert np.allclose(z1a, z1b)
        assert not np.allclose(z1a, z2)


class TestShotNoise:
    def test_seed_duplicates(self, pattern):
        """ Same seed should imply same result """
        z1 = add_shot_noise(pattern, seed=7)
        z2 = add_shot_noise(pattern, seed=7)
        assert np.allclose(z1, z2)

    def test_seed_unduplicates(self, pattern):
        """ Different seeds should (almost always) give different results"""
        z1 = add_shot_noise(pattern, seed=7)
        z2 = add_shot_noise(pattern, seed=312)
        z3 = add_shot_noise(pattern, seed=None)
        assert not np.allclose(z1, z2)
        assert not np.allclose(z1, z3)


class TestGaussianNoise:
    def test_seed_duplicates(self, pattern):
        """ Same seed should imply same result """
        z1 = add_gaussian_noise(pattern, sigma=3, seed=7)
        z2 = add_gaussian_noise(pattern, sigma=3, seed=7)
        assert np.allclose(z1, z2)


class TestDeadPixel:
    def test_seed_duplicates(self, pattern):
        """ Same seed should imply same result """
        pattern = pattern + 1  # so that we can detect dead pixels
        z1 = add_dead_pixels(pattern, n=6, seed=7)
        z2 = add_dead_pixels(pattern, n=6, seed=7)
        assert np.allclose(z1, z2)
        assert np.sum(z1 == 0) == 6  # we should have 6 dead pixels!

    def test_frac_kwarg(self, pattern):
        pattern = pattern + 1  # so that we can detect dead pixels
        pattern_size = pattern.shape[0] * pattern.shape[1]
        fraction = 6 / pattern_size
        z1 = add_dead_pixels(pattern, fraction=fraction)
        assert np.sum(z1 == 0) == 6  # we should have 6 dead pixels!

    @pytest.mark.xfail(strict=True)
    def test_bad_kwarg_choices_a(self, pattern):
        _ = add_dead_pixels(pattern, n=None, fraction=None)

    @pytest.mark.xfail(strict=True)
    def test_bad_kwarg_choices_b(self, pattern):
        _ = add_dead_pixels(pattern, n=6, fraction=0.2)


class TestDetectorGainOffset:
    def test_gain_scalar(self, pattern):
        """ Tests scalar gains are invertible """
        g1 = add_linear_detector_gain(pattern, 1.1)
        g2 = add_linear_detector_gain(g1, 1 / 1.1)
        assert np.allclose(pattern, g2)

    def test_gain_array(self, pattern):
        """ Test array gain are invertible """
        pattern = pattern + 1  # avoids problems inverting zeroes
        g1 = add_linear_detector_gain(pattern, pattern)
        g2 = add_linear_detector_gain(g1, np.divide(1, pattern))
        assert np.allclose(pattern, g2)

    def test_offset_scalar(self, pattern):
        """ Test postive scalar offsets are invertible """
        g1 = add_detector_offset(pattern, 3)
        g2 = add_detector_offset(g1, -3)
        assert np.allclose(pattern, g2)

    def test_offset_array(self, pattern):
        """ Test postive array offsets are invertible """
        g1 = add_detector_offset(pattern, pattern)
        g2 = add_detector_offset(g1, -pattern)
        assert np.allclose(pattern, g2)
