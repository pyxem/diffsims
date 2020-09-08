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

from diffpy.structure.spacegroups import GetSpaceGroup
import numpy as np
import pytest

from diffsims.structure_factor import (
    find_asymmetric_positions,
    get_kinematical_structure_factor,
    get_doyleturner_structure_factor,
    get_refraction_corrected_wavelength,
)


@pytest.mark.parametrize(
    "positions, space_group, desired_asymmetric",
    [
        ([[0, 0, 0], [0, 0.5, 0.5]], 229, [True, False]),
        ([[0.0349, 0.3106, 0.2607], [0.0349, 0.3106, 0.2607]], 229, [True, True]),
    ],
)
def test_find_asymmetric_positions(positions, space_group, desired_asymmetric):
    assert np.allclose(
        find_asymmetric_positions(
            positions=positions, space_group=GetSpaceGroup(space_group)
        ),
        desired_asymmetric,
    )


def test_get_kinematical_structure_factor():
    pass


def test_get_doyleturner_structure_factor():
    pass


def test_get_refraction_corrected_wavelength():
    pass
