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

import matplotlib.pyplot as plt

from diffsims.rotations import ConstrainedRotation
from diffsims.utils.sampling_utils import get_reduced_fundamental_zone_grid
from orix.vector.vector3d import Vector3d
import numpy as np

class TestConstrainedRotation:
    def setup(self):
        rot = get_reduced_fundamental_zone_grid(resolution=1,)
        self.rot = rot

    def test_setup(self):
        assert isinstance(self.rot, ConstrainedRotation)

    def test_corresponding_beam_direction(self):
        assert isinstance(self.rot.corresponding_beam_direction, Vector3d)

    def test_to_euler(self):
        assert isinstance(self.rot.to_euler(), np.ndarray)

    def test_to_stereographic(self):
        assert isinstance(self.rot.to_stereographic(), tuple)
        assert isinstance(self.rot.to_stereographic()[0], np.ndarray)
        assert isinstance(self.rot.to_stereographic()[1], np.ndarray)

    def test_plot(self):
        ax = self.rot.plot()
        assert isinstance(ax, plt.Axes)
        fig, ax = plt.subplots(1)
        self.rot.plot(ax=ax)
