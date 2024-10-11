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

import numpy as np
from orix.crystal_map import Phase
from orix.vector import Vector3d
import pytest

from diffsims.crystallography import (
    get_equivalent_hkl,
    get_highest_hkl,
    get_hkl,
    ReciprocalLatticeVector,
)
from diffsims.constants import VisibleDeprecationWarning


class TestGetHKL:
    @pytest.mark.parametrize("highest_hkl, n", [([1, 2, 3], 105), ([1, 1, 1], 27)])
    def test_get_hkl(self, highest_hkl, n):
        highest_hkl = np.array(highest_hkl)
        with pytest.warns(VisibleDeprecationWarning):
            hkl = get_hkl(highest_hkl)
        assert np.allclose(hkl.max(axis=0), highest_hkl)
        assert np.allclose(hkl.min(axis=0), -highest_hkl)
        assert hkl.shape[0] == n
        assert np.allclose(abs(hkl).min(axis=0), [0, 0, 0])  # Contains zero-vector

        # Slightly different implementation (from orix)
        phase = Phase(point_group="1")
        g = ReciprocalLatticeVector.from_highest_hkl(phase, highest_hkl)
        assert np.allclose(g.data.max(axis=0), highest_hkl)
        assert np.allclose(g.data.min(axis=0), -highest_hkl)
        assert g.size == n - 1
        assert np.allclose(abs(g.data).min(axis=0), [0, 0, 0])

    @pytest.mark.parametrize(
        "d, hkl", [(0.5, [6, 6, 21]), (1, [3, 3, 11]), (1.5, [2, 2, 7])]
    )
    def test_get_highest_hkl(self, silicon_carbide_phase, d, hkl):
        with pytest.warns(VisibleDeprecationWarning):
            hkl_out = get_highest_hkl(
                silicon_carbide_phase.structure.lattice, min_dspacing=d
            )
        assert np.allclose(hkl_out, hkl)

        # Slightly different implementation (from orix)
        g = ReciprocalLatticeVector.from_min_dspacing(silicon_carbide_phase, d)
        assert np.allclose(g.hkl.max(axis=0) + 1, hkl_out)

    def test_get_equivalent_hkl(self):
        phase_225 = Phase(space_group=225)
        with pytest.warns(VisibleDeprecationWarning):
            hkl1 = get_equivalent_hkl(
                [1, 1, 1], operations=phase_225.point_group, unique=True
            )
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
        g1 = ReciprocalLatticeVector(phase_225, hkl=[1, 1, 1])
        assert np.allclose(g1.symmetrise().hkl, hkl1.data)

        with pytest.warns(VisibleDeprecationWarning):
            hkl2 = get_equivalent_hkl([1, 1, 1], operations=phase_225.point_group)
        assert hkl2.shape[0] == g1.symmetrise().size * 6 == 48

        phase_186 = Phase(space_group=186)
        with pytest.warns(VisibleDeprecationWarning):
            hkl3, mult3 = get_equivalent_hkl(
                [2, 2, 0],
                operations=phase_186.point_group,
                return_multiplicity=True,
                unique=True,
            )
        g3 = ReciprocalLatticeVector(phase_186, hkl=[2, 2, 0])
        assert mult3 == g3.symmetrise().size == 12
        assert np.allclose(hkl3.data[:2], [[2, 2, 0], [-2.7321, 0.7321, 0]], atol=1e-4)

        with pytest.warns(VisibleDeprecationWarning):
            hkl4, mult4 = get_equivalent_hkl(
                Vector3d([[2, 2, 0], [4, 0, 0]]),
                phase_186.point_group,
                unique=True,
                return_multiplicity=True,
            )
        g4 = ReciprocalLatticeVector(phase_186, hkl=[[2, 2, 0], [4, 0, 0]])
        g4_sym, mult4_2 = g4.symmetrise(return_multiplicity=True)
        assert np.allclose(mult4, [12, 6])
        assert np.allclose(mult4_2, [12, 6])
        assert hkl4.shape[0] == g4_sym.size == 18
