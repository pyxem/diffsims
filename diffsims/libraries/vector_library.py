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

import numpy as np

class DiffractionVectorLibrary(dict):
    """Maps crystal structure (phase) to diffraction vectors.

    The library is a dictionary mapping from a phase name to phase information.
    The phase information is stored as a dictionary with the following entries:

    'indices' : np.array
        List of peak indices [hkl1, hkl2] as a 2D array.
    'measurements' : np.array
        List of vector measurements [len1, len2, angle] in the same order as
        the indices. Lengths in reciprocal Angstrom and angles in radians.

    Attributes
    ----------
    identifiers : list of strings/ints
        A list of phase identifiers referring to different atomic structures.
    structures : list of diffpy.structure.Structure objects.
        A list of diffpy.structure.Structure objects describing the atomic
        structure associated with each phase in the library.
    reciprocal_radius : float
        Maximum reciprocal radius used when generating the library.
    """

    def __init__(self, *args, **kwargs):
        self.identifiers = None
        self.structures = None
        self.reciprocal_radius = None
