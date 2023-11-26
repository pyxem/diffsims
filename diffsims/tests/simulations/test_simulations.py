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
from orix.quaternion import Rotation

from diffsims.crystallography import ReciprocalLatticeVector
from diffsims.simulations.simulation import Simulation, ProfileSimulation
from diffsims.generators.simulation_generator import SimulationGenerator


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


@pytest.fixture(scope="module")
def al_phase():
    p = Phase(
        name="al",
        space_group=225,
        structure=Structure(
            atoms=[Atom("al", [0, 0, 0])],
            lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90),
        ),
    )
    return p


class TestDiffractionSimulation:
    @pytest.fixture
    def diffraction_simulation(self, al_phase):
        vector = ReciprocalLatticeVector(
            phase=al_phase,
            xyz=np.array(
                [
                    [0, 0, 0],
                ]
            ),
        )

        return Simulation(
            phases=al_phase,
            rotations=Rotation.from_axes_angles([0, 0, 1], angles=0),
            coordinates=vector,
            simulation_generator=SimulationGenerator(300),
        )

    def test_init(self, diffraction_simulation):
        assert np.allclose(
            diffraction_simulation.coordinates.data,
            np.array(
                [
                    [0, 0, 0],
                ]
            ),
        )
        assert diffraction_simulation.coordinates.hkl.shape == (1, 3)
        assert np.isnan(diffraction_simulation.coordinates.intensity).all()
        assert diffraction_simulation.coordinates.intensity.shape == (1,)
        assert np.allclose(diffraction_simulation.calibration, np.array([0.1, 0.1]))
        assert diffraction_simulation.coordinates.size == 1

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
        diffraction_simulation = Simulation(
            phases=al_phase,
            coordinates=vect,
            rotations=[0, 0, 1],
            simulation_generator=SimulationGenerator(300),
        )
        diffraction_simulation.calibration = calibration
        diffraction_simulation.offset = offset
        assert np.allclose(diffraction_simulation.calibrated_coordinates, expected)

    def test_irot(self, diffraction_simulation):
        with pytest.raises(ValueError):
            diffraction_simulation.irot[0]

    def test_iphase(self, diffraction_simulation):
        with pytest.raises(ValueError):
            diffraction_simulation.iphase[0]

    def test_iter(self, diffraction_simulation):
        count = 0
        for sim in diffraction_simulation:
            count += 1
            assert isinstance(sim, ReciprocalLatticeVector)
        assert count == 1


class TestMultiRotationSimulation:
    @pytest.fixture(scope="class")
    def diffraction_simulation(self, al_phase):
        vector = ReciprocalLatticeVector(
            phase=al_phase, xyz=np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]])
        )

        return Simulation(
            phases=al_phase,
            rotations=Rotation.from_axes_angles([0, 0, 1], angles=[0, 45, 60]),
            coordinates=vector,
            simulation_generator=SimulationGenerator(300),
        )

    def test_irot(self, diffraction_simulation):
        assert isinstance(diffraction_simulation.irot[0], Simulation)
        assert diffraction_simulation.irot[0].rotations == Rotation.from_axes_angles(
            [0, 0, 1], angles=0
        )
        assert isinstance(diffraction_simulation.irot[1], Simulation)
        assert diffraction_simulation.irot[1].rotations == Rotation.from_axes_angles(
            [0, 0, 1], angles=45
        )

    def test_iter(self, diffraction_simulation):
        diffraction_simulation.phase_index = 0
        diffraction_simulation.rotation_index = 0
        count = 0
        for sim in diffraction_simulation:
            count += 1
            assert isinstance(sim, ReciprocalLatticeVector)
        assert count == 3


class TestMultiPhaseMultiRotationSimulation:
    @pytest.fixture(scope="class")
    def diffraction_simulation(self, al_phase):
        vector = ReciprocalLatticeVector(
            phase=al_phase, xyz=np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]])
        )
        al_phase2 = al_phase.deepcopy()
        al_phase2.name = "al2"
        al_phase.name = "al1"

        return Simulation(
            phases=[al_phase, al_phase2],
            rotations=[
                Rotation.from_axes_angles([0, 0, 1], angles=[0, 45, 60]),
                Rotation.from_axes_angles([0, 0, 1], angles=[0, 45, 60]),
            ],
            coordinates=[vector, vector],
            simulation_generator=SimulationGenerator(300),
        )

    def test_iphase(self, diffraction_simulation):
        assert isinstance(diffraction_simulation.iphase[0], Simulation)
        assert diffraction_simulation.iphase[0].current_phase.name == "al1"
        assert diffraction_simulation.iphase["al1"].current_phase.name == "al1"
        assert isinstance(diffraction_simulation.iphase[0].phases, Phase)

        assert isinstance(diffraction_simulation.iphase[1], Simulation)
        assert diffraction_simulation.iphase[1].current_phase.name == "al2"
        assert diffraction_simulation.iphase["al2"].current_phase.name == "al2"

    def test_irot(self, diffraction_simulation):
        assert isinstance(diffraction_simulation.irot[0], Simulation)
        assert diffraction_simulation.iphase[0].irot[
            0
        ].rotations == Rotation.from_axes_angles([0, 0, 1], angles=0)
        assert isinstance(diffraction_simulation.irot[1], Simulation)
        assert diffraction_simulation.iphase[0].irot[
            1
        ].rotations == Rotation.from_axes_angles([0, 0, 1], angles=45)

    def test_iter(self, diffraction_simulation):
        diffraction_simulation.phase_index = 0
        diffraction_simulation.rotation_index = 0
        count = 0
        for sim in diffraction_simulation:
            count += 1
            assert isinstance(sim, ReciprocalLatticeVector)
        assert count == 6
