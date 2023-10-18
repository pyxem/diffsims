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

"""Diffraction pattern library generator and associated tools.
"""

import numpy as np
from typing import Tuple, Optional, Mapping
from tqdm import tqdm
from dataclasses import dataclass, field

from diffsims.libraries.diffraction_library import DiffractionLibrary
from diffsims.libraries.vector_library import DiffractionVectorLibrary
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator

from diffsims.utils.vector_utils import get_angle_cartesian_vec


__all__ = [
    "DiffractionLibraryGenerator",
    "VectorLibraryGenerator",
]


@dataclass
class DiffractionLibraryGenerator:
    """Computes a library of electron diffraction patterns for specified atomic
    structures and orientations.

    Parameters
    ----------
    electron_diffraction_calculator
        The calculator used to simulate diffraction patterns.
    structure_library
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    reciprocal_radius
        The maximum g-vector magnitude to be included in the simulations.
    with_direct_beam
        Include the direct beam in the library.
    max_excitation_error
        The extinction distance for reflections, in reciprocal Angstroms.
    shape_factor_width
        Determines the width of the shape functions of the reflections in
        Angstroms. If not set is equal to max_excitation_error.
    debye_waller_factors : dict of str:value pairs
        Maps element names to their temperature-dependent Debye-Waller factors.
    calibration
        The calibration of experimental data to be correlated with the
        library, in reciprocal Angstroms per pixel.
    half_shape
        The half shape of the target patterns in pixels, for 144x144 use
        (72,72) etc.
    """

    electron_diffraction_calculator: DiffractionGenerator
    structure_library: StructureLibrary
    reciprocal_radius: float
    with_direct_beam: bool = True
    max_excitation_error: float = 1e-2
    shape_factor_width: Optional[float] = None
    debye_waller_factors: Optional[Mapping[str, float]] = None
    calibration: Optional[float] = None
    half_shape: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.shape_factor_width is None:
            self.shape_factor_width = self.max_excitation_error
        if self.debye_waller_factors is None:
            self.debye_waller_factors = {}

    def calculate_library(self) -> DiffractionLibrary:
        """Calculates a dictionary of diffraction data for a library of crystal
        structures and orientations.

        Each structure in the structure library is rotated to each associated
        orientation and the diffraction pattern is calculated each time.

        Returns
        -------
        DiffractionLibrary
            Mapping of crystal structure and orientation to diffraction data
            objects.

        """
        # Define DiffractionLibrary object to contain results
        diffraction_library = DiffractionLibrary()
        # The electron diffraction calculator to do simulations
        diffractor = self.electron_diffraction_calculator
        # Iterate through phases in library.
        for phase_name in self.structure_library.keys():
            phase = self.structure_library[phase_name].phase
            orientations = self.structure_library[phase_name].orientations
            num_orientations = orientations.size
            simulations = np.empty(num_orientations, dtype="object")
            pixel_coords = np.empty(num_orientations, dtype="object")
            intensities = np.empty(num_orientations, dtype="object")
            # Iterate through orientations of each phase.
            for i, orientation in enumerate(tqdm(orientations, leave=False)):
                simulation = diffractor.calculate_ed_data(
                    phase=phase,
                    reciprocal_radius=self.reciprocal_radius,
                    rotation=orientation,
                    with_direct_beam=self.with_direct_beam,
                    max_excitation_error=self.max_excitation_error,
                    shape_factor_width=self.shape_factor_width,
                    debye_waller_factors=self.debye_waller_factors,
                )

                # Calibrate simulation
                simulation.calibration = self.calibration
                pixel_coordinates = np.rint(
                    simulation.calibrated_coordinates[:, :2] + self.half_shape
                ).astype(int)

                # Construct diffraction simulation library
                simulations[i] = simulation
                pixel_coords[i] = pixel_coordinates
                intensities[i] = simulation.intensities

            # Add phase to diffraction library
            diffraction_library.add_phase(name=phase_name,
                                          phase=phase,
                                          orientations=orientations,
                                          simulations=simulations,
                                          pixel_coords=pixel_coords,
                                          intensities=intensities)

        # Pass attributes to diffraction library from structure library.
        diffraction_library.diffraction_generator = diffractor
        return diffraction_library