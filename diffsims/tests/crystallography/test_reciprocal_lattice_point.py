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
from orix.crystal_map import Phase
from orix.vector import Vector3d
import pytest

from diffsims.crystallography import ReciprocalLatticePoint


class TestReciprocalLatticePoint:
    @pytest.mark.parametrize(
        "hkl", [[[1, 1, 1], [2, 0, 0]], np.array([[1, 1, 1], [2, 0, 0]])]
    )
    def test_init_rlp(self, nickel_phase, hkl):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=nickel_phase, hkl=hkl)
        assert rlp.phase.name == nickel_phase.name
        assert isinstance(rlp.hkl, Vector3d)
        assert rlp.structure_factor[0] is None
        assert rlp.theta[0] is None
        assert rlp.size == 2
        assert rlp.shape == (2, 3)
        assert rlp.hkl[0].shape == (1,)
        assert rlp.hkl.data[0].shape == (3,)
        assert np.issubdtype(rlp.hkl.data.dtype, int)

    @pytest.mark.parametrize("min_dspacing, desired_size", [(2, 9), (1, 19), (0.5, 83)])
    def test_init_from_min_dspacing(self, ferrite_phase, min_dspacing, desired_size):
        with pytest.warns(np.VisibleDeprecationWarning):
            assert (
                ReciprocalLatticePoint.from_min_dspacing(
                    phase=ferrite_phase, min_dspacing=min_dspacing
                ).size
                == desired_size
            )

    @pytest.mark.parametrize(
        "highest_hkl, desired_highest_hkl, desired_lowest_hkl, desired_size",
        [
            ([3, 3, 3], [3, 3, 3], [1, 0, 0], 19),
            ([3, 4, 0], [3, 4, 0], [0, 4, 0], 13),
            ([4, 3, 0], [4, 3, 0], [1, 0, 0], 13),
        ],
    )
    def test_init_from_highest_hkl(
        self,
        silicon_carbide_phase,
        highest_hkl,
        desired_highest_hkl,
        desired_lowest_hkl,
        desired_size,
    ):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint.from_highest_hkl(
                phase=silicon_carbide_phase, highest_hkl=highest_hkl
            )
        assert np.allclose(rlp[0].hkl.data, desired_highest_hkl)
        assert np.allclose(rlp[-1].hkl.data, desired_lowest_hkl)
        assert rlp.size == desired_size

    def test_repr(self, ferrite_phase):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint.from_min_dspacing(
                ferrite_phase, min_dspacing=2
            )
        assert repr(rlp) == (
            f"ReciprocalLatticePoint (9,)\n"
            f"Phase: ferrite (m-3m)\n"
            "[[2 2 2]\n [2 2 1]\n [2 2 0]\n [2 1 1]\n [2 1 0]\n [2 0 0]\n [1 1 1]\n"
            " [1 1 0]\n [1 0 0]]"
        )

    def test_get_item(self, ferrite_phase):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint.from_min_dspacing(
                phase=ferrite_phase, min_dspacing=1.5
            )
        rlp.calculate_structure_factor()
        rlp.calculate_theta(voltage=20e3)

        assert rlp[0].size == 1
        assert rlp[:2].size == 2
        assert np.allclose(rlp[5:7].hkl.data, rlp.hkl[5:7].data)

        assert np.allclose(rlp[10:13].structure_factor, rlp.structure_factor[10:13])
        assert np.allclose(rlp[20:23].theta, rlp.theta[20:23])

        assert rlp.phase.space_group.number == rlp[0].phase.space_group.number
        assert rlp.phase.point_group.name == rlp[10:15].phase.point_group.name
        assert np.allclose(
            rlp.phase.structure.lattice.abcABG(),
            rlp[20:23].phase.structure.lattice.abcABG(),
        )

    def test_get_hkl(self, silicon_carbide_phase):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint.from_min_dspacing(
                silicon_carbide_phase, min_dspacing=3
            )
        assert np.allclose(rlp.h, [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        assert np.allclose(rlp.k, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        assert np.allclose(rlp.l, [4, 3, 2, 1, 0, 4, 3, 2, 0, 4, 3, 2])

    def test_multiplicity(self, nickel_phase, silicon_carbide_phase):
        with pytest.warns(np.VisibleDeprecationWarning):
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
        with pytest.warns(np.VisibleDeprecationWarning):
            assert np.allclose(
                ReciprocalLatticePoint.from_min_dspacing(
                    phase=silicon_carbide_phase, min_dspacing=1
                ).multiplicity,
                # fmt: off
                np.array([
                    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6,
                    6, 6, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    1, 1, 1, 1, 1, 1, 1, 1
                ])
                # fmt: on
            )

    def test_gspacing_dspacing_scattering_parameter(self, ferrite_phase):
        with pytest.warns(np.VisibleDeprecationWarning):
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
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=Phase(space_group=space_group), hkl=hkl)
        assert rlp.phase.space_group.short_name[0] == centering
        assert np.allclose(rlp.allowed, desired_allowed)

    def test_allowed_b_centering(self):
        """B centering will never fire since no diffpy.structure space group has
        'B' first in the name.
        """
        phase = Phase(space_group=15)
        phase.space_group.short_name = "B"
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(
                phase=phase, hkl=[[1, 2, 2], [1, 1, -1], [1, 1, 2]]
            )
        assert np.allclose(rlp.allowed, [False, True, False])

    def test_allowed_raises(self, silicon_carbide_phase):
        with pytest.raises(NotImplementedError):
            with pytest.warns(np.VisibleDeprecationWarning):
                _ = ReciprocalLatticePoint.from_min_dspacing(
                    phase=silicon_carbide_phase, min_dspacing=1
                ).allowed

    def test_unique(self, ferrite_phase):
        hkl = [[-1, -1, -1], [1, 1, 1], [1, 0, 0], [0, 0, 1]]
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=ferrite_phase, hkl=hkl)
        assert isinstance(rlp.unique(), ReciprocalLatticePoint)
        assert np.allclose(rlp.unique(use_symmetry=False).hkl.data, hkl)
        assert np.allclose(rlp.unique().hkl.data, [[1, 1, 1], [1, 0, 0]])

    def test_symmetrise(self):
        with pytest.warns(np.VisibleDeprecationWarning):
            assert np.allclose(
                ReciprocalLatticePoint(phase=Phase(space_group=225), hkl=[1, 1, 1])
                .symmetrise()
                .hkl.data,
                np.array(
                    [
                        [1, 1, 1],
                        [-1, 1, 1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1],
                    ]
                ),
            )

        with pytest.warns(np.VisibleDeprecationWarning):
            rlp2, multiplicity = ReciprocalLatticePoint(
                phase=Phase(space_group=186), hkl=[2, 2, 0]
            ).symmetrise(return_multiplicity=True)
        assert multiplicity == 12
        assert np.allclose(
            rlp2.hkl.data,
            [
                [2, 2, 0],
                [-2, 0, 0],
                [0, -2, 0],
                [-2, -2, 0],
                [2, 0, 0],
                [0, 2, 0],
                [-2, 2, 0],
                [0, -2, 0],
                [2, 0, 0],
                [2, -2, 0],
                [0, 2, 0],
                [-2, 0, 0],
            ],
        )

        with pytest.warns(np.VisibleDeprecationWarning):
            rlp3 = ReciprocalLatticePoint(
                phase=Phase(space_group=186), hkl=[2, 2, 0]
            ).symmetrise(antipodal=False)
        assert np.allclose(
            rlp3.hkl.data,
            [[2, 2, 0], [-2, 0, 0], [0, -2, 0], [-2, -2, 0], [2, 0, 0], [0, 2, 0]],
        )

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
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=ferrite_phase, hkl=hkl)
        rlp.calculate_structure_factor(method=method, voltage=voltage)
        assert np.allclose(rlp.structure_factor, desired_factor)

    def test_calculate_structure_factor_raises(self, ferrite_phase):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=ferrite_phase, hkl=[1, 0, 0])
        with pytest.raises(ValueError, match="method=man must be among"):
            rlp.calculate_structure_factor(method="man")
        with pytest.raises(ValueError, match="'voltage' parameter must be set when"):
            rlp.calculate_structure_factor(method="doyleturner")

    @pytest.mark.parametrize(
        "voltage, hkl, desired_theta",
        [(20e3, [1, 1, 1], 0.00259284), (200e3, [2, 0, 0], 0.00087484)],
    )
    def test_calculate_theta(self, ferrite_phase, voltage, hkl, desired_theta):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=ferrite_phase, hkl=hkl)
        rlp.calculate_theta(voltage=voltage)
        assert np.allclose(rlp.theta, desired_theta)

    def test_one_point(self, ferrite_phase):
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=ferrite_phase, hkl=[1, 1, 0])

        assert rlp.size == 1
        assert np.allclose(rlp.allowed, True)

    def test_init_without_point_group_raises(self):
        phase = Phase()
        with pytest.raises(ValueError, match=f"The phase {phase} must have a"):
            with pytest.warns(np.VisibleDeprecationWarning):
                _ = ReciprocalLatticePoint(phase=phase, hkl=[1, 1, 1])

    def test_get_allowed_without_space_group_raises(self):
        phase = Phase(point_group="432")
        with pytest.warns(np.VisibleDeprecationWarning):
            rlp = ReciprocalLatticePoint(phase=phase, hkl=[1, 1, 1])

        with pytest.raises(ValueError, match=f"The phase {phase} must have a"):
            _ = rlp.allowed
