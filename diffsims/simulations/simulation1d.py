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

from typing import Union, Sequence, TYPE_CHECKING, Any
import copy

import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.quaternion import Rotation
from orix.vector import Vector3d

from diffsims.crystallography.reciprocal_lattice_vector import ReciprocalLatticeVector
from diffsims.pattern.detector_functions import add_shot_and_point_spread

# to avoid circular imports
if TYPE_CHECKING:  # pragma: no cover
    from diffsims.generators.simulation_generator import SimulationGenerator


class Simulation1D:
    """Holds the result of a 1D simulation for some phase"""

    def __init__(
        self,
        phase: Phase,
        reciprocal_spacing: np.ndarray,
        intensities: np.ndarray,
        hkl: np.ndarray,
        reciprocal_radius: float,
    ):
        """Initializes the DiffractionSimulation object with data values for
        the coordinates, indices, intensities, calibration and offset.

        Parameters
        ----------
        phase
            The phase of the simulation
        reciprocal_spacing
            The spacing of the reciprocal lattice vectors
        intensities
            The intensities of the diffraction spots
        hkl
            The hkl indices of the diffraction spots
        reciprocal_radius
            The radius which the reciprocal lattice spacings are plotted out to
        """
        self.phase = phase
        self.reciprocal_spacing = reciprocal_spacing
        self.intensities = intensities
        self.hkl = hkl
        self.reciprocal_radius = reciprocal_radius

    def plot(self, ax=None, annotate_peaks=False, fontsize=12, with_labels=True):
        """Plots the 1D diffraction pattern,"""
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for g, i, hkls in zip(self.reciprocal_spacing, self.intensities, self.hkl):
            label = hkls
            ax.plot([g, g], [0, i], color="k", linewidth=3, label=label)
            if annotate_peaks:
                ax.annotate(label, xy=[g, i], xytext=[g, i], fontsize=fontsize)

            if with_labels:
                ax.set_xlabel("A ($^{-1}$)")
                ax.set_ylabel("Intensities (scaled)")
        return ax
