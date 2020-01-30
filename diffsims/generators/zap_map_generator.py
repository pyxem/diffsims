# -*- coding: utf-8 -*-
# Copyright 2017-2019 The diffsims developers
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


def generate_zap_map(structure,simulator,density):
    """
    Produces a number of zone axis patterns for a structure

    Parameters
    ----------
    structure : diffpy.structure

    simulator : diffsims.diffraction_generator

    density : str
        '3' for the corners or '7' (corners + 3 midpoints + 1 centroid)

    Returns
    -------
    zap_dictionary : dict
        Keys are zone axes, values are simulations
    """
    pass
