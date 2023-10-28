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

from diffpy.structure import Structure, Atom, Lattice
from orix.crystal_map import Phase

from diffsims.crystallography import ReciprocalLatticeVector
from diffsims.simulations.simulation import Simulation, ProfileSimulation


@pytest.fixture
def profile_simulation():
    return ProfileSimulation(
        magnitudes=[
            0.31891931643691351,
            0.52079306292509475,
            0.6106839974876449,
            0.73651261277849378,
            0.80259601243613932,
            0.9020400452156796,
            0.95675794931074043,
            1.0415861258501895,
            1.0893168446141808,
            1.1645286909108374,
            1.2074090451670043,
            1.2756772657476541,
        ],
        intensities=np.array(
            [
                100.0,
                99.34619104,
                64.1846346,
                18.57137199,
                28.84307971,
                41.31084268,
                23.42104951,
                13.996264,
                24.87559364,
                20.85636003,
                9.46737774,
                5.43222307,
            ]
        ),
        hkls=[
            (1, 1, 1),
            (2, 2, 0),
            (3, 1, 1),
            (4, 0, 0),
            (3, 3, 1),
            (4, 2, 2),
            (3, 3, 3),
            (4, 4, 0),
            (5, 3, 1),
            (6, 2, 0),
            (5, 3, 3),
            (4, 4, 4),
        ],
    )


def test_plot_profile_simulation(profile_simulation):
    profile_simulation.plot()


class TestDiffractionSimulation:
    @pytest.fixture
    def al_phase(self):
        p = Phase(
            name="al",
            space_group=225,
            structure=Structure(
                atoms=[Atom("al", [0, 0, 0])],
                lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90),
            ),
        )
        return p

    @pytest.fixture
    def diffraction_simulation(self, al_phase):
        vector = ReciprocalLatticeVector(
            phase=al_phase, xyz=np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]])
        )

        return DiffractionSimulation(vector)

    @pytest.fixture
    def diffraction_simulation_calibrated(self, al_phase):
        vector = ReciprocalLatticeVector(
            phase=al_phase, xyz=np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]])
        )

        return DiffractionSimulation(vector, calibration=0.5)

    def test_init(self, diffraction_simulation):
        assert np.allclose(
            diffraction_simulation.coordinates.data, np.array([[1, 2, 3], [3, 4, 5]])
        )
        assert diffraction_simulation.coordinates.hkl.shape == (2, 3)
        assert np.isnan(diffraction_simulation.coordinates.intensity).all()
        assert diffraction_simulation.coordinates.intensity.shape == (2,)
        assert diffraction_simulation.calibration is None
        assert len(diffraction_simulation) == 2

    def test_single_spot(self, al_phase):
        rlv = ReciprocalLatticeVector(phase=al_phase, xyz=np.array([[1, 2, 3]]))
        assert DiffractionSimulation(rlv).coordinates.data.shape == (1, 3)

    def test_get_item(self, diffraction_simulation):
        assert diffraction_simulation[1].coordinates.data.shape == (1, 3)
        assert diffraction_simulation[0:2].coordinates.data.shape == (2, 3)

    def test_append(self, diffraction_simulation):
        sim = diffraction_simulation.append(diffraction_simulation)
        assert np.allclose(
            sim.coordinates.data,
            np.array(
                [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
            ),
        )
        assert sim.size == 4

    def test_indices_setter_getter(self, diffraction_simulation):
        indices = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        diffraction_simulation.indices = indices[-1]
        assert np.allclose(diffraction_simulation.indices, indices[-1])
        diffraction_simulation.with_direct_beam = True
        diffraction_simulation.indices = indices
        assert np.allclose(diffraction_simulation.indices, indices)

    def test_coordinates_setter(self, diffraction_simulation):
        starting_coords = diffraction_simulation.coordinates[-1]
        diffraction_simulation.coordinates = diffraction_simulation.coordinates[-1]
        assert np.allclose(
            diffraction_simulation.coordinates.data, starting_coords.data
        )
        diffraction_simulation.with_direct_beam = True

    def test_intensities_setter(self, diffraction_simulation):
        ints = np.array([1, 2, 3])
        diffraction_simulation.intensities = ints[-1]
        assert np.allclose(diffraction_simulation.intensities, ints[-1])
        diffraction_simulation.with_direct_beam = True
        diffraction_simulation.intensities = ints
        assert np.allclose(diffraction_simulation.intensities, ints)

    @pytest.mark.parametrize(
        "calibration, expected",
        [
            (5.0, np.array((5.0, 5.0))),
            pytest.param(0, (0, 0), marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param((0, 0), (0, 0), marks=pytest.mark.xfail(raises=ValueError)),
            ((1.5, 1.5), np.array((1.5, 1.5))),
            ((1.3, 1.5), np.array((1.3, 1.5))),
        ],
    )
    def test_calibration(self, diffraction_simulation, calibration, expected):
        diffraction_simulation.calibration = calibration
        assert np.allclose(diffraction_simulation.calibration, expected)

    @pytest.mark.parametrize(
        "coordinates, with_direct_beam, expected",
        [
            (
                np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
                False,
                np.array([True, False, True]),
            ),
            (
                np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
                True,
                np.array([True, True, True]),
            ),
            (np.array([[-1, 0, 0], [1, 0, 0]]), False, np.array([True, True])),
        ],
    )
    def test_direct_beam_mask(self, al_phase, coordinates, with_direct_beam, expected):
        vect = ReciprocalLatticeVector(phase=al_phase, xyz=coordinates)
        diffraction_simulation = DiffractionSimulation(
            vect, with_direct_beam=with_direct_beam
        )
        diffraction_simulation.with_direct_beam = with_direct_beam
        mask = diffraction_simulation.direct_beam_mask
        assert np.all(mask == expected)

    @pytest.mark.parametrize(
        "coordinates, calibration, offset, expected",
        [
            (
                np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
                1.0,
                (0.0, 0.0),
                np.array([[1.0, 0.0], [1.0, 2.0]]),
            ),
            (
                np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
                2.0,
                (3.0, 1.0),
                np.array([[2.0, 0.5], [2.0, 1.5]]),
            ),
            pytest.param(
                np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
                None,
                (0.0, 0.0),
                None,
                marks=pytest.mark.xfail(raises=Exception),
            ),
        ],
    )
    def test_calibrated_coordinates(
        self,
        al_phase,
        coordinates,
        calibration,
        offset,
        expected,
    ):
        vect = ReciprocalLatticeVector(phase=al_phase, xyz=coordinates)
        diffraction_simulation = DiffractionSimulation(vect)
        diffraction_simulation.calibration = calibration
        diffraction_simulation.offset = offset
        assert np.allclose(diffraction_simulation.calibrated_coordinates, expected)

    @pytest.mark.parametrize(
        "units, expected",
        [
            ("real", np.array([[-2, 1, 3], [-4, 3, 5]])),
            ("pixel", np.array([[-2, 1, 3], [-4, 3, 5]])),
        ],
    )
    def test_transform_coordinates(
        self, diffraction_simulation_calibrated, units, expected
    ):
        tc = diffraction_simulation_calibrated._get_transformed_coordinates(
            90, units=units
        )
        assert np.allclose(tc.coordinates.data, expected)

    def test_rotate_shift_coordinates(self, diffraction_simulation):
        rot = diffraction_simulation.rotate_shift_coordinates(90)
        assert np.allclose(rot.coordinates.data, np.array([[-2, 1, 3], [-4, 3, 5]]))

    def test_assertion_free_get_diffraction_pattern(self, al_phase):
        vect = ReciprocalLatticeVector(phase=al_phase, xyz=np.array([[0.3, 1.2, 0]]))
        vect.intensity = np.ones(1)
        short_sim = DiffractionSimulation(vect, calibration=[1, 2])

        z = short_sim.get_diffraction_pattern()

        vect = ReciprocalLatticeVector(phase=al_phase, xyz=np.asarray([[0.3, 1000, 0]]))
        vect.intensity = np.ones(1)
        empty_sim = DiffractionSimulation(vect, calibration=[1, 2])

        z = empty_sim.get_diffraction_pattern(shape=(10, 20))

    def test_get_as_mask(self, al_phase):
        vect = ReciprocalLatticeVector(phase=al_phase, xyz=np.asarray([[0.3, 1.2, 0]]))
        vect.intensity = np.ones(1)
        short_sim = DiffractionSimulation(vect, calibration=[1, 2])
        mask = short_sim.get_as_mask(
            (20, 10),
            radius_function=np.sqrt,
        )
        assert mask.shape[0] == 20
        assert mask.shape[1] == 10

    def test_polar_coordinates(self, al_phase):
        vect = ReciprocalLatticeVector(phase=al_phase, xyz=np.asarray([[1, 1, 0]]))
        vect.intensity = np.ones(1)
        short_sim = DiffractionSimulation(vect, calibration=[0.5, 0.5])
        r, t = short_sim.get_polar_coordinates(real=True)
        assert r == [
            1.4142135623730951,
        ]
        assert t == [
            0.7853981633974483,
        ]
        r, t = short_sim.get_polar_coordinates(real=False)
        assert r == [
            np.sqrt(8),
        ]
        assert t == [
            0.7853981633974483,
        ]

    @pytest.mark.parametrize("units_in", ["pixel", "real"])
    def test_plot_method(self, al_phase, units_in):
        vect = ReciprocalLatticeVector(
            phase=al_phase,
            xyz=np.asarray(
                [
                    [0.3, 1.2, 0],
                    [-2, 3, 0],
                    [2.1, 3.4, 0],
                ]
            ),
        )
        vect.intensity = np.array([3.0, 5.0, 2.0])
        short_sim = DiffractionSimulation(coordinates=vect, calibration=[1, 2])
        ax, sp = short_sim.plot(units=units_in, show_labels=True)
