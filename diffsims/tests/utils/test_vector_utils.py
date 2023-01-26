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
import pytest

from diffsims.utils.vector_utils import get_angle_cartesian
from diffsims.utils.vector_utils import get_angle_cartesian_vec


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_angle",
    [([0, 0, 1], [0, 1, 0], np.deg2rad(90)), ([0, 0, 0], [0, 0, 1], 0)],
)
def test_get_angle_cartesian(vec_a, vec_b, expected_angle):
    angle = get_angle_cartesian(vec_a, vec_b)
    np.testing.assert_allclose(angle, expected_angle)


@pytest.mark.parametrize(
    "a, b, expected_angles",
    [
        (
            np.array([[0, 0, 1], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 1]]),
            [np.deg2rad(90), 0],
        )
    ],
)
def test_get_angle_cartesian_vec(a, b, expected_angles):
    angles = get_angle_cartesian_vec(a, b)
    np.testing.assert_allclose(angles, expected_angles)


@pytest.mark.xfail(raises=ValueError)
def test_get_angle_cartesian_vec_input_validation():
    get_angle_cartesian_vec(np.empty((2, 3)), np.empty((5, 3)))
