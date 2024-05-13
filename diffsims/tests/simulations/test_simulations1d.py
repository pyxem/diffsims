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

import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import Phase
import pytest

from diffsims.tests.generators.test_simulation_generator import make_phase
from diffsims.simulations import Simulation1D


class TestSingleSimulation:
    @pytest.fixture
    def simulation1d(self):
        al_phase = make_phase()
        al_phase.name = "Al"
        hkls = np.array(["100", "110", "111"])
        magnitudes = np.array([1, 2, 3])
        inten = np.array([1, 2, 3])
        recip = 4.0

        return Simulation1D(
            phase=al_phase,
            hkl=hkls,
            reciprocal_spacing=magnitudes,
            intensities=inten,
            reciprocal_radius=recip,
            wavelength=0.025,
        )

    def test_init(self, simulation1d):
        assert isinstance(simulation1d, Simulation1D)
        assert isinstance(simulation1d.phase, Phase)
        assert isinstance(simulation1d.hkl, np.ndarray)
        assert isinstance(simulation1d.reciprocal_spacing, np.ndarray)
        assert isinstance(simulation1d.intensities, np.ndarray)
        assert isinstance(simulation1d.reciprocal_radius, float)

    @pytest.mark.parametrize("annotate", [True, False])
    @pytest.mark.parametrize("ax", [None, "new"])
    @pytest.mark.parametrize("with_labels", [True, False])
    def test_plot(self, simulation1d, annotate, ax, with_labels):
        if ax == "new":
            _, ax = plt.subplots()
        _ = simulation1d.plot(annotate_peaks=annotate, ax=ax, with_labels=with_labels)

        plt.close()

    def test_repr(self, simulation1d):
        assert simulation1d.__repr__() == "Simulation1D(name: Al, wavelength: 0.025)"

    def test_theta(self, simulation1d):
        np.testing.assert_almost_equal(
            simulation1d.theta, np.array([0.02499479, 0.0499584, 0.07485985])
        )
