# -*- coding: utf-8 -*-
# Copyright 2017-2021 The diffsims developers
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
from orix.vector import Miller
import pytest

from diffsims.crystallography import ReciprocalLatticeVector


class TestReciprocalLatticeVector:
    @pytest.mark.parametrize(
        "hkl", [[[1, 1, 1], [2, 0, 0]], np.array([[1, 1, 1], [2, 0, 0]])]
    )
    def test_init_reciprocal_lattice_vector(self, nickel_phase, hkl):
        rlv = ReciprocalLatticeVector(phase=nickel_phase, hkl=hkl)
        assert rlv.phase.name == nickel_phase.name
        assert isinstance(rlv, Miller)
        assert rlv.structure_factor[0] is None
        assert rlv.theta[0] is None
        assert rlv.size == 2
        assert rlv.shape == (2,)
        assert rlv[0].shape == (1,)
        assert rlv.hkl[0].shape == (3,)
        assert np.issubdtype(rlv.data.dtype, float)

    @pytest.mark.parametrize(
        "min_dspacing, desired_size", [(2, 26), (1, 124), (0.5, 1330)]
    )
    def test_init_from_min_dspacing(self, ferrite_phase, min_dspacing, desired_size):
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            phase=ferrite_phase, min_dspacing=min_dspacing
        )
        assert rlv.size == desired_size

    @pytest.mark.parametrize(
        "highest_hkl, desired_highest_hkl, desired_lowest_hkl, desired_size",
        [
            ([3, 3, 3], [3, 3, 3], [-3, -3, -3], 342),
            ([3, 4, 0], [3, 4, 0], [-3, -4, 0], 62),
            ([4, 3, 0], [4, 3, 0], [-4, -3, 0], 62),
        ],
    )
    def test_init_from_highest_indices(
        self,
        silicon_carbide_phase,
        highest_hkl,
        desired_highest_hkl,
        desired_lowest_hkl,
        desired_size,
    ):
        rlv = ReciprocalLatticeVector.from_highest_indices(
            phase=silicon_carbide_phase, hkl=highest_hkl
        )
        assert np.allclose(rlv[0].hkl.data, desired_highest_hkl)
        assert np.allclose(rlv[-1].hkl.data, desired_lowest_hkl)
        assert rlv.size == desired_size

    def test_repr(self, ferrite_phase):
        rlv = ReciprocalLatticeVector.from_min_dspacing(ferrite_phase, 2)
        assert repr(rlv).split("\n")[:2] == [
            "ReciprocalLatticeVector (26,), ferrite (m-3m)",
            "[[ 1.  1.  1.]"
        ]

    def test_get_item(self, ferrite_phase):
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            phase=ferrite_phase, min_dspacing=1.5
        )
        rlv.calculate_structure_factor()
        rlv.calculate_theta(voltage=20e3)

        assert rlv[0].size == 1
        assert rlv[:2].size == 2
        assert np.allclose(rlv[5:7].hkl, rlv.hkl[5:7])

        assert np.allclose(rlv[10:13].structure_factor, rlv.structure_factor[10:13])
        assert np.allclose(rlv[20:23].theta, rlv.theta[20:23])

        assert rlv.phase.space_group.number == rlv[0].phase.space_group.number
        assert rlv.phase.point_group.name == rlv[10:15].phase.point_group.name
        assert np.allclose(
            rlv.phase.structure.lattice.abcABG(),
            rlv[20:23].phase.structure.lattice.abcABG(),
        )

    def test_get_hkl(self, silicon_carbide_phase):
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            silicon_carbide_phase, min_dspacing=3
        )
        assert np.allclose(rlv.h, [0, 0, 0, 0, 0, 0])
        assert np.allclose(rlv.k, [0, 0, 0, 0, 0, 0])
        assert np.allclose(rlv.l, [3, 2, 1, -1, -2, -3])

    def test_gspacing_dspacing_scattering_parameter(self, ferrite_phase):
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            phase=ferrite_phase, min_dspacing=1
        )
        rlv = rlv.unique(use_symmetry=True)
        # fmt: off
        assert np.allclose(
            rlv.gspacing,
            np.array([
                0.34885749, 0.69771498, 0.493359, 0.78006907, 0.98671799, 0.604238,
                0.85452285, 1.04657248, 1.2084778
            ])
        )
        assert np.allclose(
            rlv.dspacing,
            np.array([
                2.8665, 1.43325, 2.02692159, 1.28193777, 1.01346079, 1.65497455,
                1.17024372, 0.9555, 0.8274872
            ])
        )
        assert np.allclose(
            rlv.scattering_parameter,
            np.array([
                0.17442875, 0.34885749, 0.2466795, 0.39003453, 0.493359,
                0.30211945, 0.42726142, 0.52328624, 0.6042389
            ])
        )
        # fmt: on

    @pytest.mark.parametrize(
        "space_group, hkl, centering, desired_allowed",
        [
            (224, [[1, 1, -1], [-2, 2, 2], [3, 1, 0]], "P", [True, True, True]),
            (230, [[1, 1, -1], [-2, 2, 2], [3, 1, 0]], "I", [False, True, True]),
            (225, [[1, 1, 1], [2, 2, 1], [-3, 3, 3]], "F", [True, False, True]),
            (167, [[1, 2, 3], [2, 2, 3], [-3, 2, 4]], "H", [False, True, True]),  # R
            (68, [[1, 2, 3], [2, 2, 3], [-2, 2, 4]], "C", [False, True, True]),
            (41, [[1, 2, 2], [1, 1, -1], [1, 1, 2]], "A", [True, True, False]),
        ],
    )
    def test_allowed(self, space_group, hkl, centering, desired_allowed):
        rlv = ReciprocalLatticeVector(phase=Phase(space_group=space_group), hkl=hkl)
        assert rlv.phase.space_group.short_name[0] == centering
        assert np.allclose(rlv.allowed, desired_allowed)

    def test_allowed_b_centering(self):
        """B centering will never fire since no diffpy.structure space group has
        'B' first in the name.
        """
        phase = Phase(space_group=15)
        phase.space_group.short_name = "B"
        rlv = ReciprocalLatticeVector(
            phase=phase, hkl=[[1, 2, 2], [1, 1, -1], [1, 1, 2]]
        )
        assert np.allclose(rlv.allowed, [False, True, False])

    def test_allowed_raises(self, silicon_carbide_phase):
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            phase=silicon_carbide_phase, min_dspacing=1
        )
        with pytest.raises(NotImplementedError):
            _ = rlv.allowed

    @pytest.mark.parametrize(
        "method, voltage, hkl, desired_factor",
        [
            ("kinematical", None, [1, 1, 0], 35.783295),
            (None, None, [1, 1, 0], 35.783295),
            ("doyleturner", 20e3, [[2, 0, 0], [1, 1, 0]], [5.581302, 8.096651]),
        ],
    )
    def test_calculate_structure_factor(
        self, ferrite_phase, method, voltage, hkl, desired_factor
    ):
        rlv = ReciprocalLatticeVector(phase=ferrite_phase, hkl=hkl)
        if method is None:
            rlv.calculate_structure_factor(voltage=voltage)
        else:
            rlv.calculate_structure_factor(method=method, voltage=voltage)
        assert np.allclose(rlv.structure_factor, desired_factor)

    def test_calculate_structure_factor_raises(self, ferrite_phase):
        rlv = ReciprocalLatticeVector(phase=ferrite_phase, hkl=[1, 0, 0])
        with pytest.raises(ValueError, match="method=man must be among"):
            rlv.calculate_structure_factor(method="man")
        with pytest.raises(ValueError, match="'voltage' parameter must be set when"):
            rlv.calculate_structure_factor(method="doyleturner")

    @pytest.mark.parametrize(
        "voltage, hkl, desired_theta",
        [(20e3, [1, 1, 1], 0.00259284), (200e3, [2, 0, 0], 0.00087484)],
    )
    def test_calculate_theta(self, ferrite_phase, voltage, hkl, desired_theta):
        rlv = ReciprocalLatticeVector(phase=ferrite_phase, hkl=hkl)
        rlv.calculate_theta(voltage=voltage)
        assert np.allclose(rlv.theta, desired_theta)

    def test_one_point(self, ferrite_phase):
        rlv = ReciprocalLatticeVector(phase=ferrite_phase, hkl=[1, 1, 0])
        assert rlv.size == 1
        assert np.allclose(rlv.allowed, True)

    def test_init_without_point_group_raises(self):
        phase = Phase()
        with pytest.raises(ValueError, match=f"The phase {phase} must have a"):
            _ = ReciprocalLatticeVector(phase=phase, hkl=[1, 1, 1])

    def test_get_allowed_without_space_group_raises(self):
        phase = Phase(point_group="432")
        rlv = ReciprocalLatticeVector(phase=phase, hkl=[1, 1, 1])
        with pytest.raises(ValueError, match=f"The phase {phase} must have a"):
            _ = rlv.allowed
