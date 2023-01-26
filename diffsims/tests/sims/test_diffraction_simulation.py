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

from diffsims.sims.diffraction_simulation import (
    DiffractionSimulation, ProfileSimulation
)


def test_wrong_calibration_setting():
    with pytest.raises(ValueError, match="must be a float"):
        DiffractionSimulation(
            coordinates=np.asarray([[0.3, 1.2, 0]]),
            intensities=np.ones(1),
            calibration=[1, 2, 5],
        )


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
    profile_simulation.get_plot()


class TestDiffractionSimulation:
    @pytest.fixture
    def diffraction_simulation(self):
        return DiffractionSimulation(
            np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]], dtype=float)
        )

    @pytest.fixture
    def diffraction_simulation_calibrated(self):
        return DiffractionSimulation(
            np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]], dtype=float), calibration=0.5
        )

    def test_failed_initialization(self):
        with pytest.raises(ValueError, match="Coordinate"):
            DiffractionSimulation(
                np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]], dtype=float),
                indices=np.array([1, 2, 3])
            )

    def test_init(self, diffraction_simulation):
        assert np.allclose(
            diffraction_simulation.coordinates, np.array([[1, 2, 3], [3, 4, 5]])
        )
        assert np.isnan(diffraction_simulation.indices).all()
        assert diffraction_simulation.indices.shape == (2, 3)
        assert np.isnan(diffraction_simulation.intensities).all()
        assert diffraction_simulation.intensities.shape == (2,)
        assert diffraction_simulation.calibration is None
        assert len(diffraction_simulation) == 2

    def test_single_spot(self):
        assert DiffractionSimulation(np.array([1, 2, 3])).coordinates.shape == (1, 3)

    def test_get_item(self, diffraction_simulation):
        assert diffraction_simulation[1].coordinates.shape == (1, 3)
        assert diffraction_simulation[0:2].coordinates.shape == (2, 3)

    def test_fail_get_item(self, diffraction_simulation):
        with pytest.raises(ValueError, match="Invalid"):
            _ = diffraction_simulation[None]

    def test_addition(self, diffraction_simulation):
        sim = diffraction_simulation + diffraction_simulation
        assert np.allclose(
            sim.coordinates,
            np.array(
                [
                    [1, 2, 3],
                    [3, 4, 5],
                    [1, 2, 3],
                    [3, 4, 5],
                ]
            ),
        )
        assert sim.size == 4

    def test_extend(self, diffraction_simulation):
        diffraction_simulation.extend(diffraction_simulation)
        assert np.allclose(
            diffraction_simulation.coordinates,
            np.array(
                [
                    [1, 2, 3],
                    [3, 4, 5],
                    [1, 2, 3],
                    [3, 4, 5],
                ]
            ),
        )

    def test_indices_setter_getter(self, diffraction_simulation):
        indices = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        diffraction_simulation.indices = indices[-1]
        assert np.allclose(diffraction_simulation.indices, indices[-1])
        diffraction_simulation.with_direct_beam = True
        diffraction_simulation.indices = indices
        assert np.allclose(diffraction_simulation.indices, indices)

    def test_coordinates_setter(self, diffraction_simulation):
        coords = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        diffraction_simulation.coordinates = coords[-1]
        assert np.allclose(diffraction_simulation.coordinates, coords[-1])
        diffraction_simulation.with_direct_beam = True
        diffraction_simulation.coordinates = coords
        assert np.allclose(diffraction_simulation.coordinates, coords)

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
    def test_direct_beam_mask(self, coordinates, with_direct_beam, expected):
        diffraction_simulation = DiffractionSimulation(
            coordinates, with_direct_beam=with_direct_beam
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
        coordinates,
        calibration,
        offset,
        expected,
    ):
        diffraction_simulation = DiffractionSimulation(coordinates)
        diffraction_simulation.calibration = calibration
        diffraction_simulation.offset = offset
        assert np.allclose(diffraction_simulation.calibrated_coordinates, expected)

    @pytest.mark.parametrize(
        "units, expected",
        [
            ("real", np.array([[-2, 1, 3], [-4, 3, 5]])),
            ("pixel", np.array([[-4, 2], [-8, 6]])),
        ],
    )
    def test_transform_coordinates(
        self, diffraction_simulation_calibrated, units, expected
    ):
        tc = diffraction_simulation_calibrated._get_transformed_coordinates(
            90, units=units
        )
        assert np.allclose(tc, expected)

    def test_rotate_shift_coordinates(self, diffraction_simulation):
        diffraction_simulation.rotate_shift_coordinates(90)
        assert np.allclose(
            diffraction_simulation.coordinates, np.array([[-2, 1, 3], [-4, 3, 5]])
        )

    def test_assertion_free_get_diffraction_pattern(self):
        short_sim = DiffractionSimulation(
            coordinates=np.asarray([[0.3, 1.2, 0]]),
            intensities=np.ones(1),
            calibration=[1, 2],
        )
        z = short_sim.get_diffraction_pattern()

        empty_sim = DiffractionSimulation(
            coordinates=np.asarray([[0.3, 1000, 0]]),
            intensities=np.ones(1),
            calibration=[2, 2],
        )
        z = empty_sim.get_diffraction_pattern(shape=(10, 20))

    def test_get_as_mask(self):
        short_sim = DiffractionSimulation(
            coordinates=np.asarray([[0.3, 1.2, 0]]),
            intensities=np.ones(1),
            calibration=[1, 2],
        )
        mask = short_sim.get_as_mask(
            (20, 10),
            radius_function=np.sqrt,
        )
        assert mask.shape[0] == 20
        assert mask.shape[1] == 10

    @pytest.mark.parametrize("units_in", ["pixel", "real"])
    def test_plot_method(self, units_in):
        short_sim = DiffractionSimulation(
            coordinates=np.asarray(
                [
                    [0.3, 1.2, 0],
                    [-2, 3, 0],
                    [2.1, 3.4, 0],
                ]
            ),
            indices=np.array([[-2, 3, 4], [2, -6, 1], [0, 0, 0]]),
            intensities=np.array([3.0, 5.0, 2.0]),
            calibration=[1, 2],
        )
        ax, sp = short_sim.plot(units=units_in, show_labels=True)
