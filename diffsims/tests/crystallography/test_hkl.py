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
from orix.quaternion.symmetry import get_point_group
from orix.vector import Vector3d
import pytest

from diffsims.crystallography import get_equivalent_hkl, get_highest_hkl, get_hkl


class TestGetHKL:
    @pytest.mark.parametrize("highest_hkl, n", [([1, 2, 3], 105), ([1, 1, 1], 27)])
    def test_get_hkl(self, highest_hkl, n):
        hkl = get_hkl(highest_hkl)
        assert np.allclose(hkl.max(axis=0), highest_hkl)
        assert np.allclose(hkl.min(axis=0), -np.array(highest_hkl))
        assert hkl.shape[0] == n
        assert np.allclose(abs(hkl).min(axis=0), [0, 0, 0])  # Contains zero-vector

    @pytest.mark.parametrize(
        "d, hkl", [(0.5, [6, 6, 21]), (1, [3, 3, 11]), (1.5, [2, 2, 7])]
    )
    def test_get_highest_hkl(self, silicon_carbide_phase, d, hkl):
        hkl_out = get_highest_hkl(
            silicon_carbide_phase.structure.lattice, min_dspacing=d
        )
        assert np.allclose(hkl_out, hkl)

    def test_get_equivalent_hkl(self):
        pgm3m = get_point_group(225)
        hkl1 = get_equivalent_hkl([1, 1, 1], operations=pgm3m, unique=True)
        # fmt: off
        assert np.allclose(
            hkl1.data,
            [
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
            ]
        )
        # fmt: on

        hkl2 = get_equivalent_hkl([1, 1, 1], operations=pgm3m)
        assert hkl2.shape[0] == 48

        pg6mmm = get_point_group(186)
        hkl3, mult3 = get_equivalent_hkl(
            [2, 2, 0], operations=pg6mmm, return_multiplicity=True, unique=True
        )
        assert mult3 == 12
        assert np.allclose(hkl3.data[:2], [[2, 2, 0], [-2.7321, 0.7321, 0]], atol=1e-4)

        hkl4, mult4 = get_equivalent_hkl(
            Vector3d([[2, 2, 0], [4, 0, 0]]),
            pg6mmm,
            unique=True,
            return_multiplicity=True,
        )
        assert np.allclose(mult4, [12, 6])
        assert hkl4.shape == (18,)
