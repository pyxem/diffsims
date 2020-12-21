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

from diffsims.pattern.detector_functions import add_shot_noise

class TestShotNoise():
    @pytest.fixture()
    def pattern(self):
        z = np.zeros((128,128))
        z[40:44,50:53] = 1e5
        return z

    def test_seed_duplicates(self,pattern):
        """ Same seed should imply same result """
        z1 = add_shot_noise(pattern,seed=7)
        z2 = add_shot_noise(pattern,seed=7)
        assert np.allclose(z1,z2)

    def test_seed_unduplicates():
        """ Different seeds should (almost always) give different results"""
        z1 = add_shot_noise(pattern,seed=7)
        z2 = add_shot_noise(pattern,seed=312)
        assert not np.allclose(z1,z2)
