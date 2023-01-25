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

from diffsims.structure_factor import (
    get_atomic_scattering_parameters,
    get_element_id_from_string,
)
from diffsims.structure_factor.atomic_scattering_parameters import (
    _get_string_from_element_id,
)


@pytest.mark.parametrize(
    "element, unit, desired_parameters",
    [
        (10, "m", [0.303, 0.720, 0.475, 0.153, 17.640, 5.860, 1.762, 0.266]),
        ("Fe", None, [2.544, 2.343, 1.759, 0.506, 64.424, 14.880, 2.854, 0.350]),
        (
            "Fe",
            "NM",
            [0.02544, 0.02343, 0.01759, 0.00506, 0.64424, 0.14880, 0.02854, 0.00350],
        ),
        ("bk", "Å", [6.502, 5.478, 2.510, 0.000, 28.375, 4.975, 0.561, 0.0]),
    ],
)
def test_get_atomic_scattering_parameters(element, unit, desired_parameters):
    a, b = get_atomic_scattering_parameters(element, unit=unit)
    assert np.allclose(a, desired_parameters[:4])
    assert np.allclose(b, desired_parameters[4:])


def test_get_element_id_from_string():
    # fmt: off
    elements = [
        "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al",
        "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn",
        "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb",
        "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in",
        "sn", "SB", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd", "pm",
        "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta",
        "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at",
        "rn", "FR", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk",
        "cf", "es", "fm", "md", "no", "lr", "rf", "db", "sg", "bh", "hs", "mt",
        "ds", "rg", "cn", "nh", "fl", "mc", "lv", "ts", "OG"
    ]
    # fmt: on
    elements_id = np.arange(98) + 1
    for i, element in zip(elements_id, elements):
        assert get_element_id_from_string(element) == i
        # The reverse
        assert _get_string_from_element_id(i) == element.capitalize()
