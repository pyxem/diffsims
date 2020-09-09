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

from diffsims.crystallography import ReciprocalLatticePoint
import numpy as np
from orix.vector import Vector3d
import pytest


class TestReciprocalLatticePoint:
    @pytest.mark.parametrize(
        "hkl", [[[1, 1, 1], [2, 0, 0]], np.array([[1, 1, 1], [2, 0, 0]])]
    )
    def test_init_rlp(self, nickel_phase, hkl):
        rlp = ReciprocalLatticePoint(phase=nickel_phase, hkl=hkl)
        assert rlp.phase.name == nickel_phase.name
        assert isinstance(rlp.hkl, Vector3d)
        assert rlp.structure_factor[0] is None
        assert rlp.theta[0] is None
        assert rlp.size == 2
        assert rlp.shape == (2, 3)
        assert rlp.hkl[0].shape == (1,)
        assert rlp._hkldata[0].shape == (3,)
        assert np.issubdtype(rlp.hkl.data.dtype, int)

    @pytest.mark.parametrize("min_dspacing, desired_size", [(2, 9), (1, 19), (0.5, 83)])
    def test_init_from_min_dspacing(self, ferrite_phase, min_dspacing, desired_size):
        assert ReciprocalLatticePoint.from_min_dspacing(
            phase=ferrite_phase, min_dspacing=min_dspacing
        ).size == desired_size

    @pytest.mark.parametrize(
        "highest_hkl, desired_highest_hkl, desired_lowest_hkl, desired_size",
        [
            ([3, 3, 3], [3, 3, 3], [1, 0, 0], 19),
            ([3, 4, 0], [3, 4, 0], [0, 4, 0], 13),
            ([4, 3, 0], [4, 3, 0], [1, 0, 0], 13)
        ]
    )
    def test_init_from_highest_hkl(
        self,
        silicon_carbide_phase,
        highest_hkl,
        desired_highest_hkl,
        desired_lowest_hkl,
        desired_size
    ):
        rlp = ReciprocalLatticePoint.from_highest_hkl(
            phase=silicon_carbide_phase, highest_hkl=highest_hkl
        )
        assert np.allclose(rlp[0]._hkldata, desired_highest_hkl)
        assert np.allclose(rlp[-1]._hkldata, desired_lowest_hkl)
        assert rlp.size == desired_size

    def test_repr(self, ferrite_phase):
        rlp = ReciprocalLatticePoint.from_min_dspacing(ferrite_phase, min_dspacing=2)
        assert repr(rlp) == (
            f"ReciprocalLatticePoint (9,)\n"
            f"Phase: ferrite (m-3m)\n"
            "[[2 2 2]\n [2 2 1]\n [2 2 0]\n [2 1 1]\n [2 1 0]\n [2 0 0]\n [1 1 1]\n"
            " [1 1 0]\n [1 0 0]]"
        )

    def test_get_item(self):
        pass

    def test_get_hkl(self, silicon_carbide_phase):
        rlp = ReciprocalLatticePoint.from_min_dspacing(
            silicon_carbide_phase, min_dspacing=3
        )
        assert np.allclose(rlp.h, [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        assert np.allclose(rlp.k, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        assert np.allclose(rlp.l, [4, 3, 2, 1, 0, 4, 3, 2, 0, 4, 3, 2])

    def test_multiplicity(self, nickel_phase, silicon_carbide_phase):
        assert np.allclose(
            ReciprocalLatticePoint.from_min_dspacing(
                phase=nickel_phase, min_dspacing=1
            ).multiplicity,
            # fmt: off
            np.array([
                8, 24, 24, 24, 12, 24, 48, 48, 24, 24, 48, 24, 24, 24, 6, 8, 24,
                24, 12, 24, 48, 24, 24, 24,  6,  8, 24, 12, 24, 24,  6,  8, 12, 6
            ])
            # fmt: on
        )
        assert np.allclose(
            ReciprocalLatticePoint.from_min_dspacing(
                phase=silicon_carbide_phase, min_dspacing=1
            ).multiplicity,
            # fmt: off
            np.array([
                12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 12,
                12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1,
                1, 1, 1, 1, 1, 1
            ])
            # fmt: on
        )

    def test_gspacing_dspacing_scattering_parameter(self, ferrite_phase):
        rlp = ReciprocalLatticePoint.from_min_dspacing(
            phase=ferrite_phase, min_dspacing=2
        )
        # fmt: off
        assert np.allclose(
            rlp.gspacing,
            np.array([
                1.2084778, 1.04657248, 0.98671799, 0.85452285, 0.78006907, 0.69771498,
                0.604238, 0.493359, 0.34885749
            ])
        )
        assert np.allclose(
            rlp.dspacing,
            np.array([
                0.82748727, 0.9555, 1.01346079, 1.17024372, 1.28193777, 1.43325,
                1.65497455, 2.02692159, 2.8665
            ])
        )
        assert np.allclose(
            rlp.scattering_parameter,
            np.array([
                0.6042389, 0.52328624, 0.493359, 0.42726142, 0.39003453, 0.34885749,
                0.30211945, 0.2466795, 0.17442875
            ])
        )
        # fmt: on

    def test_allowed(self):
        pass

    def test_unique(self):
        pass

    def test_symmetrise(self):
        pass

    def test_calculate_structure_factor(self):
        pass

    def test_calculate_theta(self):
        pass
