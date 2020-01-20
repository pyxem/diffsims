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

from diffsims.libraries.structure_library import StructureLibrary

class StructureLibraryGenerator:
    """Generates a structure library for the given phases

    Parameters
    ----------
    phases : list
        Array of three-component phase descriptions, where the phase
        description is [<phase name> : string, <structure> :
        diffpy.structure.Structure, <rotation_list> : list]

    Attributes
    ----------
    phase_names : list of string
        List of phase names.
    structures : list of diffpy.structure.Structure
        List of structures.
    orientation : list of lists

    """

    def __init__(self, phases):
        self.phase_names = [phase[0] for phase in phases]
        self.structures = [phase[1] for phase in phases]
        self.orientations = [phase[2] for phase in phases]

    def get_library(self):
        """Create a structure library

        Returns
        -------
        structure_library : StructureLibrary
            Structure library for the given phase names, structures and orientations.
        """
        return StructureLibrary(self.phase_names, self.structures, self.orientations)
