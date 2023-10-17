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

from diffsims.utils.sim_utils import ReciprocalSpaceSample
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
        for phase_name in structure_library.struct_lib.keys():
            structure = structure_library.struct_lib[phase_name][0]
            orientations = structure_library.struct_lib[phase_name][1]

        # Pass attributes to diffraction library from structure library.
        diffraction_library.identifiers = structure_library.identifiers
        diffraction_library.structures = structure_library.structures
        diffraction_library.diffraction_generator = diffractor
        diffraction_library.reciprocal_radius = reciprocal_radius
        diffraction_library.with_direct_beam = with_direct_beam

        return diffraction_library


def _generate_lookup_table(recip_latt, reciprocal_radius: float, unique: bool = True):
    """Generate a look-up table with all combinations of indices,
    including their reciprocal distances and the angle between
    them.

    Parameters
    ----------
    recip_latt : :class:`diffpy.structure.lattice.Lattice`
        Reciprocal lattice
    reciprocal_radius : float
        The maximum g-vector magnitude to be included in the library.
    unique : bool
        Return a unique list of phase measurements

    Returns
    -------
    indices : np.array
        Nx2x3 numpy array containing the miller indices for
        reflection1, reflection2
    measurements : np.array
        Nx3 numpy array containing len1, len2, angle

    """
    miller_indices, coordinates, distances = get_points_in_sphere(
        recip_latt, reciprocal_radius
    )

    # Create pair_indices for selecting all point pair combinations
    num_indices = len(miller_indices)
    pair_a_indices, pair_b_indices = np.mgrid[:num_indices, :num_indices]

    # Only select one of the permutations and don't pair an index with
    # itself (select above diagonal)
    upper_indices = np.triu_indices(num_indices, 1)
    pair_a_indices = pair_a_indices[upper_indices].ravel()
    pair_b_indices = pair_b_indices[upper_indices].ravel()

    # Mask off origin (0, 0, 0)
    origin_index = num_indices // 2
    pair_a_indices = pair_a_indices[pair_a_indices != origin_index]
    pair_b_indices = pair_b_indices[pair_b_indices != origin_index]

    pair_indices = np.vstack([pair_a_indices, pair_b_indices])

    # Create library entries
    angles = get_angle_cartesian_vec(
        coordinates[pair_a_indices], coordinates[pair_b_indices]
    )
    pair_distances = distances[pair_indices.T]
    # Ensure longest vector is first
    len_sort = np.fliplr(pair_distances.argsort(axis=1))
    # phase_index_pairs is a list of [hkl1, hkl2]
    phase_index_pairs = np.take_along_axis(
        miller_indices[pair_indices.T], len_sort[:, :, np.newaxis], axis=1
    )
    # phase_measurements is a list of [len1, len2, angle]
    phase_measurements = np.column_stack(
        (np.take_along_axis(pair_distances, len_sort, axis=1), angles)
    )

    if unique:
        # Only keep unique triplets
        measurements, measurement_indices = np.unique(
            phase_measurements, axis=0, return_index=True
        )
        indices = phase_index_pairs[measurement_indices]
    else:
        measurements = phase_measurements
        indices = phase_index_pairs

    return measurements, indices


class VectorLibraryGenerator:
    """Computes a library of diffraction vectors and pairwise inter-vector
    angles for a specified StructureLibrary.
    """

    def __init__(self, structure_library):
        """Initialises the library with a diffraction calculator.

        Parameters
        ----------
        structure_library : :class:`StructureLibrary`
            The StructureLibrary defining structures to be
        """
        self.structures = structure_library

    def get_vector_library(self, reciprocal_radius):
        """Calculates a library of diffraction vectors and pairwise inter-vector
        angles for a library of crystal structures.

        Parameters
        ----------
        reciprocal_radius : float
            The maximum g-vector magnitude to be included in the library.

        Returns
        -------
        vector_library : :class:`DiffractionVectorLibrary`
            Mapping of phase identifier to phase information in dictionary
            format.
        """
        # Define DiffractionVectorLibrary object to contain results
        vector_library = DiffractionVectorLibrary()
        # Get structures from structure library
        structure_library = self.structures.struct_lib
        # Iterate through phases in library.
        for phase_name in tqdm(structure_library.keys()):
            # Get diffpy.structure object associated with phase
            structure = structure_library[phase_name][0]
            # Get reciprocal lattice points within reciprocal_radius
            recip_latt = structure.lattice.reciprocal()

            measurements, indices = _generate_lookup_table(
                recip_latt=recip_latt, reciprocal_radius=reciprocal_radius, unique=True
            )

            vector_library[phase_name] = {
                "indices": indices,
                "measurements": measurements,
            }

        # Pass attributes to diffraction library from structure library.
        vector_library.identifiers = self.structures.identifiers
        vector_library.structures = self.structures.structures
        vector_library.reciprocal_radius = reciprocal_radius

        return vector_library
