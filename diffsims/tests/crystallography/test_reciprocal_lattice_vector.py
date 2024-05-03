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

from diffpy.structure import Atom, Lattice, Structure
import numpy as np
from orix.crystal_map import Phase
from orix.vector import Miller, Vector3d
import pytest

from diffsims.crystallography import ReciprocalLatticeVector
from diffsims.crystallography.reciprocal_lattice_vector import _get_set_per_hkl


class TestReciprocalLatticeVector:
    @pytest.mark.parametrize(
        "hkl", [[[1, 1, 1], [2, 0, 0]], np.array([[1, 1, 1], [2, 0, 0]])]
    )
    def test_init(self, nickel_phase, hkl):
        """`ReciprocalLatticeVector` instance is initialized as
        expected.
        """
        rlv = ReciprocalLatticeVector(nickel_phase, hkl=hkl)
        assert rlv.phase.name == nickel_phase.name
        assert isinstance(rlv, Vector3d)
        assert rlv.size == 2
        assert rlv.shape == (2,)
        assert rlv[0].shape == (1,)
        assert rlv.hkl[0].shape == (3,)
        assert np.issubdtype(rlv.data.dtype, float)
        assert np.allclose(
            rlv.structure_factor,
            np.full(rlv.size, np.nan, dtype="complex128"),
            equal_nan=True,
        )
        assert np.allclose(rlv.theta, np.full(rlv.size, np.nan), equal_nan=True)

    def test_init_hkil(self, nickel_phase):
        """Initialization with hkil sets `coordinate_format` and
        `coordinates` correctly.
        """
        rlv1 = ReciprocalLatticeVector(nickel_phase, hkl=[1, 1, 1])
        assert rlv1.coordinate_format == "hkl"
        assert np.allclose(rlv1.coordinates, [1, 1, 1])

        rlv2 = ReciprocalLatticeVector(nickel_phase, hkil=[1, 1, -2, 1])
        assert rlv2.coordinate_format == "hkil"
        assert np.allclose(rlv2.coordinates, [1, 1, -2, 1])

        assert np.allclose(rlv1.data, rlv2.data)

    def test_init_raises(self, nickel_phase):
        """Error is raised when not exactly one of ``hkl`` and ``hkil``
        is passed.
        """
        with pytest.raises(ValueError, match="Exactly one of "):
            _ = ReciprocalLatticeVector(nickel_phase, hkl=[1, 1, 1], hkil=[1, 1, -2, 1])
        with pytest.raises(ValueError, match="Exactly one of "):
            _ = ReciprocalLatticeVector(nickel_phase)

    @pytest.mark.parametrize("d, desired_size", [(2, 18), (1, 92), (0.5, 750)])
    def test_init_from_min_dspacing(self, ferrite_phase, d, desired_size):
        """Class method gives desired number of vectors."""
        rlv = ReciprocalLatticeVector.from_min_dspacing(ferrite_phase, d)
        assert rlv.size == desired_size

    @pytest.mark.parametrize(
        "hkl, desired_highest_hkl, desired_lowest_hkl, desired_size",
        [
            ([3, 3, 3], [3, 3, 3], [-3, -3, -3], 342),
            ([3, 4, 0], [3, 4, 0], [-3, -4, 0], 62),
            ([4, 3, 0], [4, 3, 0], [-4, -3, 0], 62),
        ],
    )
    def test_init_from_highest_hkl(
        self,
        silicon_carbide_phase,
        hkl,
        desired_highest_hkl,
        desired_lowest_hkl,
        desired_size,
    ):
        """Class method gives desired number of vectors and indices."""
        rlv = ReciprocalLatticeVector.from_highest_hkl(silicon_carbide_phase, hkl)
        assert np.allclose(rlv[0].hkl, desired_highest_hkl)
        assert np.allclose(rlv[-1].hkl, desired_lowest_hkl)
        assert rlv.size == desired_size

    def test_repr(self, ferrite_phase):
        """String representation gives desired (start of) string."""
        rlv = ReciprocalLatticeVector.from_min_dspacing(ferrite_phase, 2)
        assert repr(rlv).split("\n")[:2] == [
            "ReciprocalLatticeVector (18,), ferrite (m-3m)",
            "[[ 1.  1.  0.]",
        ]

    def test_get_item(self, ferrite_phase):
        """Indexing gives desired vectors and properties carry over."""
        rlv = ReciprocalLatticeVector.from_min_dspacing(ferrite_phase, 1.5)
        rlv.sanitise_phase()
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

    def test_get_hkil(self, silicon_carbide_phase):
        """Miller index properties give desired values."""
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            silicon_carbide_phase, min_dspacing=3
        )
        assert np.allclose(rlv.h, [0, 0, 0, 0, 0, 0])
        assert np.allclose(rlv.k, [0, 0, 0, 0, 0, 0])
        assert np.allclose(rlv.i, [0, 0, 0, 0, 0, 0])
        assert np.allclose(rlv.l, [3, 2, 1, -1, -2, -3])

    def test_gspacing_dspacing_scattering_parameter(self, ferrite_phase):
        """Length and scattering parameter properties give desired
        values.
        """
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            phase=ferrite_phase, min_dspacing=1
        )
        rlv = rlv.unique(use_symmetry=True)
        # fmt: off
        assert np.allclose(
            rlv.gspacing,
            np.array([
                0.34885749, 0.69771498, 0.493359, 0.78006907, 0.98671799, 0.604238,
                0.85452285,
            ])
        )
        assert np.allclose(
            rlv.dspacing,
            np.array([
                2.8665, 1.43325, 2.02692159, 1.28193777, 1.01346079, 1.65497455,
                1.17024372,
            ])
        )
        assert np.allclose(
            rlv.scattering_parameter,
            np.array([
                0.17442875, 0.34885749, 0.2466795, 0.39003453, 0.493359, 0.30211945,
                0.42726142,
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
        """Selection rules are correct."""
        rlv = ReciprocalLatticeVector(phase=Phase(space_group=space_group), hkl=hkl)
        assert rlv.phase.space_group.short_name[0] == centering
        assert np.allclose(rlv.allowed, desired_allowed)

    def test_allowed_b_centering(self):
        """B centering will never fire since no diffpy.structure space
        group has "B" first in the name.
        """
        phase = Phase(space_group=15)
        phase.space_group.short_name = "B"
        rlv = ReciprocalLatticeVector(
            phase=phase, hkl=[[1, 2, 2], [1, 1, -1], [1, 1, 2]]
        )
        assert np.allclose(rlv.allowed, [False, True, False])

    def test_allowed_raises(self, silicon_carbide_phase):
        """Error is raised since no selection rules are implemented for
        hexagonal phases.
        """
        rlv = ReciprocalLatticeVector.from_min_dspacing(
            phase=silicon_carbide_phase, min_dspacing=1
        )
        with pytest.raises(NotImplementedError):
            _ = rlv.allowed

        rlv.phase.space_group.short_name = "D"
        with pytest.raises(ValueError):
            _ = rlv.allowed

    def test_allowed_decimals(self, nickel_phase):
        """Rounding of Miller indices gives expected allowed vectors."""
        rlv = ReciprocalLatticeVector.from_min_dspacing(nickel_phase)
        rlv.sanitise_phase()
        rlv.calculate_structure_factor()
        structure_factor = abs(rlv.structure_factor)
        rlv = rlv[structure_factor > 0.4 * structure_factor.max()]
        assert all(rlv.allowed)

    @pytest.mark.parametrize(
        "voltage, hkl, desired_theta",
        [(20e3, [1, 1, 1], 0.0259313), (200e3, [2, 0, 0], 0.008748)],
    )
    def test_calculate_theta(self, ferrite_phase, voltage, hkl, desired_theta):
        """Bragg angle calculation gives desired value."""
        rlv = ReciprocalLatticeVector(phase=ferrite_phase, hkl=hkl)
        rlv.calculate_theta(voltage=voltage)
        assert np.allclose(rlv.theta, desired_theta, atol=1e-6)

    def test_one_vector(self, ferrite_phase):
        """Creating a single vector instance works."""
        rlv = ReciprocalLatticeVector(phase=ferrite_phase, hkl=[1, 1, 0])
        assert rlv.size == 1
        assert rlv.allowed

    def test_init_without_point_group_raises(self):
        """Error is raised when trying to initialize an instance with a
        phase without a point group.
        """
        phase = Phase()
        with pytest.raises(ValueError, match=f"The phase {phase} must have a"):
            _ = ReciprocalLatticeVector(phase, hkl=[1, 1, 1])

    def test_get_allowed_without_space_group_raises(self):
        """Error is raised when trying to evaluate selection rules from
        an initialize with a phase without a space group.
        """
        phase = Phase(point_group="432")
        rlv = ReciprocalLatticeVector(phase=phase, hkl=[1, 1, 1])
        with pytest.raises(ValueError, match=f"The phase {phase} must have a"):
            _ = rlv.allowed

    def test_multiplicity(self, nickel_phase, silicon_carbide_phase):
        """Correct vector multiplicity for cubic and hexagonal phases."""
        rlv_ni = ReciprocalLatticeVector(nickel_phase, hkl=[[1, 1, 1], [2, 0, 0]])
        assert np.allclose(rlv_ni.multiplicity, [8, 6])

        rlv_sic = ReciprocalLatticeVector(
            silicon_carbide_phase, hkil=[[0, 0, 0, -4], [1, -2, 1, 0]]
        )
        assert np.allclose(rlv_sic.multiplicity, [1, 6])

    def test_has_hexagonal_lattice(self, nickel_phase, silicon_carbide_phase):
        """Correct determination of which vector instance has an
        hexagonal/tringonal lattice.
        """
        rlv_ni = ReciprocalLatticeVector(nickel_phase, hkl=[1, 1, 1])
        rlv_sic = ReciprocalLatticeVector(silicon_carbide_phase, hkl=[1, 1, 1])
        assert not rlv_ni.has_hexagonal_lattice
        assert rlv_sic.has_hexagonal_lattice

    def test_symmetrise(self, nickel_phase):
        """Correct symmetrically equivalent vectors are obtained."""
        rlv = ReciprocalLatticeVector(
            nickel_phase, hkl=[[1, 1, 1], [-1, 1, 1], [2, 0, 0]]
        )

        rlv2 = rlv.symmetrise()
        assert rlv2.size == 22

        rlv3, mult3 = rlv.symmetrise(return_multiplicity=True)
        assert np.allclose(rlv2.hkl, rlv3.hkl)
        assert np.allclose(mult3, [8, 8, 6])

        rlv4, idx4 = rlv.symmetrise(return_index=True)
        assert np.allclose(rlv2.hkl, rlv4.hkl)
        assert np.allclose(rlv[idx4].dspacing, rlv4.dspacing)

        rlv5, mult5, idx5 = rlv.symmetrise(return_multiplicity=True, return_index=True)
        assert np.allclose(mult3, mult5)
        assert np.allclose(rlv4.hkl, rlv5.hkl)
        assert np.allclose(idx4, idx5)

    def test_print_table(self, capsys, nickel_phase):
        """Correctly printed table with indices, structure factor values
        and multiplicity, per unique vector (family).
        """
        rlv = ReciprocalLatticeVector(
            nickel_phase, hkl=[[1, 1, 1], [1, 1, -1], [2, 0, 0]]
        )

        rlv.print_table()
        captured = capsys.readouterr()
        assert captured.out == (
            " h k l      d     |F|_hkl   |F|^2   |F|^2_rel   Mult \n"
            " 1 1 1    2.034     nan      nan       nan       8   \n"
            " 2 0 0    1.762     nan      nan       nan       6   \n"
        )

        rlv.sanitise_phase()
        rlv.calculate_structure_factor()
        rlv.print_table()
        captured = capsys.readouterr()
        assert captured.out == (
            " h k l      d     |F|_hkl   |F|^2   |F|^2_rel   Mult \n"
            " 1 1 1    2.034    11.8     140.0     100.0      8   \n"
            " 2 0 0    1.762    10.4     108.2      77.3      6   \n"
        )

    def test_coordinate_format(self, nickel_phase):
        rlv1 = ReciprocalLatticeVector(nickel_phase, hkl=[1, 1, 1])
        assert rlv1.coordinate_format == "hkl"
        assert rlv1.coordinates.shape == (1, 3)

        rlv2 = ReciprocalLatticeVector(nickel_phase, hkil=[1, 1, -2, 1])
        assert rlv2.coordinate_format == "hkil"
        assert rlv2.coordinates.shape == (1, 4)
        rlv2.coordinate_format = "hkl"
        assert rlv2.coordinates.shape == (1, 3)

        rlv3 = ReciprocalLatticeVector(nickel_phase, xyz=rlv1.data)
        assert rlv2.coordinate_format == "hkl"
        assert np.allclose(rlv1.coordinates, rlv3.coordinates)

        with pytest.raises(ValueError):
            rlv1.coordinate_format = "xyz"

    def test_from_to_miller(self, nickel_phase):
        miller1 = Miller(hkl=[1, 1, 1], phase=nickel_phase)
        rlv1 = ReciprocalLatticeVector.from_miller(miller1)
        assert np.allclose(miller1.coordinates, rlv1.coordinates)
        assert miller1.coordinate_format == rlv1.coordinate_format

        rlv1.coordinate_format = "hkil"
        miller2 = rlv1.to_miller()
        assert miller2.coordinate_format == rlv1.coordinate_format
        assert np.allclose(miller2.coordinates, rlv1.coordinates)
        assert rlv1._compatible_with(ReciprocalLatticeVector.from_miller(miller2))

        miller3 = miller1.deepcopy()
        miller3.coordinate_format = "uvw"
        with pytest.raises(ValueError, match="`Miller` instance must have "):
            _ = ReciprocalLatticeVector.from_miller(miller3)

    def test_sanitise_phase(self, nickel_phase):
        rlv = ReciprocalLatticeVector(nickel_phase, hkl=[[1, 1, 1], [2, 0, 0]])

        assert len(rlv.phase.structure) == 1

        rlv2 = rlv.deepcopy()
        rlv2.sanitise_phase()
        assert len(rlv.phase.structure) == 1
        assert len(rlv2.phase.structure) == 4

        rlv3 = rlv.deepcopy()
        rlv3.phase.structure[0].element = 28
        rlv3.sanitise_phase()
        assert len(rlv3.phase.structure) == 4
        assert rlv3.phase.structure[0].element == "Ni"

    def test_get_hkl_sets(self, nickel_phase):
        hkl = np.array(
            [
                [1, 1, 1],
                [2, 0, 0],
                [2, 2, 0],
                [3, 1, 1],
                [2, 2, 2],
                [4, 0, 0],
                [3, 3, 1],
                [4, 2, 0],
                [4, 2, 2],
                [3, 3, 3],
                [5, 1, 1],
                [4, 4, 0],
                [5, 3, 1],
                [6, 0, 0],
                [4, 4, 2],
            ]
        )
        rlv = ReciprocalLatticeVector(nickel_phase, hkl=hkl)
        assert rlv.size == 15
        assert rlv.unique(use_symmetry=True).size == 15

        hkl_sets = rlv.get_hkl_sets()

        # Key and value type
        keys = list(hkl_sets.keys())
        values = list(hkl_sets.values())
        assert isinstance(keys[0], tuple)
        assert isinstance(values[0], tuple)
        assert isinstance(values[0][0], np.ndarray)
        assert len(hkl_sets) == 15

        # Actual values and indexing into the vectors
        order = [1, 5, 13, 2, 7, 11, 0, 3, 10, 6, 12, 4, 8, 14, 9]
        for key, value in zip(hkl[order], order):
            key_tuple = tuple(key)
            assert np.allclose(hkl_sets[key_tuple][0], np.array([value]))
            assert np.allclose(rlv[hkl_sets[key_tuple]].hkl, key)

        # Test 2D navigation shape
        rlv2 = rlv[:14].reshape(2, 7)
        hkl_sets2 = rlv2.get_hkl_sets()

        # Compare to previous sets
        hkl_sets.pop((4, 4, 2))

        # Compare keys
        keys2 = list(hkl_sets2.keys())
        assert np.allclose(list(hkl_sets.keys()), keys2)

        # Value type
        values2 = list(hkl_sets2.values())
        assert len(values2[0]) == 2
        assert isinstance(values2[0], tuple)
        assert isinstance(values2[1][0], np.ndarray)

        # Compare values
        values3 = np.stack(list(hkl_sets.values())).ravel()
        values3 = np.stack(np.unravel_index(values3, rlv2.shape))
        assert np.allclose(values3, np.column_stack(values2))

        # Index into the vectors with 2D navigation shape
        for key, value in hkl_sets2.items():
            assert np.allclose(rlv2[value].hkl, key)

        # Cover Python function of (JIT compiled) Numba function
        rlv3 = rlv[:2]
        rlv_unique = rlv3.unique(use_symmetry=True)
        rlv_symmetrised, mult = rlv_unique.symmetrise(return_multiplicity=True)
        hkl = rlv3.hkl.reshape(-1, 3).astype(np.float64)
        test_hkl = rlv_symmetrised.hkl.reshape(-1, 3).astype(np.float64)
        mult = mult.astype(np.int64)
        hkl_set_idx_nb = _get_set_per_hkl(hkl, test_hkl, mult)
        hkl_set_idx_py = _get_set_per_hkl.py_func(hkl, test_hkl, mult)
        assert np.allclose(hkl_set_idx_nb, hkl_set_idx_py)


class TestOverwrittenObject3d:
    def setup_class(self):
        rlv = ReciprocalLatticeVector(
            phase=Phase(
                name="nickel",
                space_group=225,
                structure=Structure(
                    lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
                    atoms=[Atom(xyz=[0, 0, 0], atype="Ni", Uisoequiv=0.006332)],
                ),
            ),
            hkl=[(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0)],
        )
        rlv.calculate_theta(20e3)
        rlv.calculate_structure_factor()
        self.vector = rlv

    def test_reshape(self):
        rlv0 = self.vector
        rlv = rlv0.reshape(3, 2)
        assert np.allclose(rlv.structure_factor, rlv0.structure_factor.reshape((3, 2)))
        assert np.allclose(rlv.theta, rlv0.theta.reshape((3, 2)))

    def test_unit(self):
        rlv = self.vector.unit
        assert np.allclose(np.linalg.norm(rlv.data, axis=1), 1)
        assert np.isnan(rlv.structure_factor).all()
        assert np.isnan(rlv.theta).all()

    def test_flatten(self):
        rlv0 = self.vector
        rlv = rlv0.flatten()
        assert rlv.shape == (6,)
        assert np.allclose(rlv.structure_factor, rlv0.structure_factor)
        assert np.allclose(rlv.theta, rlv0.theta)

    def test_squeeze(self):
        rlv0 = self.vector

        rlv1 = rlv0.reshape(2, 1, 3)
        assert rlv1.shape == (2, 1, 3)
        assert np.allclose(
            rlv1.structure_factor, rlv0.structure_factor.reshape(2, 1, 3)
        )
        assert np.allclose(rlv1.theta, rlv0.theta.reshape(2, 1, 3))

        rlv2 = rlv1.squeeze()
        assert rlv2.shape == (2, 3)
        assert np.allclose(rlv2.structure_factor, rlv0.structure_factor.reshape(2, 3))
        assert np.allclose(rlv2.theta, rlv0.theta.reshape(2, 3))

    def test_transpose(self):
        rlv1 = self.vector.reshape(2, 3)
        rlv2 = rlv1.transpose()
        assert np.allclose(rlv1.structure_factor, rlv2.structure_factor.reshape(2, 3))

    def test_stack(self):
        rlv1 = ReciprocalLatticeVector.stack(self.vector)
        assert rlv1.shape == (1, 6)
        assert self.vector._compatible_with(rlv1)

        rlv2 = ReciprocalLatticeVector.stack((self.vector, self.vector))
        assert rlv2.shape == (6, 2)

        rlv3 = self.vector.deepcopy()
        rlv3.phase.structure.lattice.setLatPar(a=2)
        assert not self.vector._compatible_with(rlv3)
        with pytest.raises(ValueError):
            _ = ReciprocalLatticeVector.stack((self.vector, rlv3))

    def test_empty(self):
        with pytest.raises(NotImplementedError):
            _ = ReciprocalLatticeVector.empty()

    def test_unique(self, nickel_phase):
        """Correct unique vectors are obtained."""
        rlv = ReciprocalLatticeVector(
            nickel_phase, hkl=[[1, 1, 1], [-1, 1, 1], [2, 0, 0], [3, 1, 1]]
        )
        rlv2 = rlv.unique()
        assert np.allclose(rlv.hkl, rlv2.hkl)
        rlv3, idx3 = rlv.unique(use_symmetry=True, return_index=True)
        assert np.allclose(rlv3.hkl, [[2, 0, 0], [1, 1, 1], [3, 1, 1]])
        assert np.allclose(rlv[idx3].hkl, rlv3.hkl)


class TestOverwrittenVector3d:
    def setup_class(self):
        rlv = ReciprocalLatticeVector(
            phase=Phase(
                name="nickel",
                space_group=225,
                structure=Structure(
                    lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
                    atoms=[Atom(xyz=[0, 0, 0], atype="Ni", Uisoequiv=0.006332)],
                ),
            ),
            hkl=[(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0)],
        )
        rlv.calculate_theta(20e3)
        rlv.sanitise_phase()
        rlv.calculate_structure_factor()
        self.vector = rlv

    def test_angle_with(self, silicon_carbide_phase):
        rlv1 = ReciprocalLatticeVector(silicon_carbide_phase, hkil=[1, 1, -2, 0])
        rlv2 = ReciprocalLatticeVector(silicon_carbide_phase, hkil=[-1, -1, 2, 0])
        assert np.isclose(rlv1.angle_with(rlv2), np.pi)
        assert np.isclose(rlv1.angle_with(rlv2, use_symmetry=True), 0)

    def test_cross(self, silicon_carbide_phase):
        rlv1 = ReciprocalLatticeVector(silicon_carbide_phase, hkil=[1, -1, 0, 0])
        rlv2 = ReciprocalLatticeVector(silicon_carbide_phase, hkil=[1, 0, -1, 0])
        miller1 = rlv1.cross(rlv2)
        assert isinstance(miller1, Miller)
        assert miller1.coordinate_format == "UVTW"
        assert np.allclose(miller1.round().coordinates, [0, 0, 0, 1])

        rlv1.coordinate_format = "hkl"
        rlv2.coordinate_format = "hkl"
        miller2 = rlv1.cross(rlv2)
        assert miller2.coordinate_format == "uvw"
        assert np.allclose(miller2.round().coordinates, [0, 0, 1])

    def test_dot(self, tetragonal_phase):
        rlv1 = ReciprocalLatticeVector(tetragonal_phase, hkl=[1, 2, 0])
        rlv2 = ReciprocalLatticeVector(tetragonal_phase, hkl=[3, 1, 1])
        assert np.isclose(rlv1.dot(rlv2), 20)

    def test_dot_outer(self, tetragonal_phase):
        rlv1 = ReciprocalLatticeVector(tetragonal_phase, hkl=[[1, 2, 0], [1, 2, 0]])
        rlv2 = ReciprocalLatticeVector(tetragonal_phase, hkl=[[3, 1, 1], [3, 1, 1]])
        dp = rlv1.dot_outer(rlv2)
        assert dp.shape == (2, 2)
        assert np.allclose(dp, [[20, 20], [20, 20]])

    def test_get_nearest(self):
        with pytest.raises(NotImplementedError):
            _ = self.vector.get_nearest()

    def test_in_fundamental_sector(self):
        with pytest.raises(NotImplementedError):
            _ = self.vector.in_fundamental_sector()

    def test_mean(self):
        with pytest.raises(NotImplementedError):
            _ = self.vector.mean()

    def test_rotate(self):
        with pytest.raises(NotImplementedError):
            _ = self.vector.rotate()

    def test_from_polar(self):
        with pytest.raises(NotImplementedError):
            _ = ReciprocalLatticeVector.from_polar(np.pi / 2, np.pi / 2)

    def test_get_random_sample(self, tetragonal_phase):
        rlv = ReciprocalLatticeVector(tetragonal_phase, hkl=[[1, 2, 0], [3, 1, 1]])
        with pytest.raises(NotImplementedError):
            _ = rlv.get_random_sample()

    def test_xvector(self):
        with pytest.raises(NotImplementedError):
            _ = ReciprocalLatticeVector.xvector()

    def test_yvector(self):
        with pytest.raises(NotImplementedError):
            _ = ReciprocalLatticeVector.yvector()

    def test_zvector(self):
        with pytest.raises(NotImplementedError):
            _ = ReciprocalLatticeVector.zvector()

    def test_zero(self):
        with pytest.raises(NotImplementedError):
            _ = ReciprocalLatticeVector.zero((10,))
