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

from __future__ import annotations
from typing import Sequence, Optional
from dataclasses import dataclass
import pickle

import numpy as np
from diffpy.structure import Structure

from diffsims.generators.diffraction_generator import DiffractionGenerator


@dataclass
class DiffractionLibrary:
    """Maps crystal structure (phase) and orientation to simulated diffraction
    data.
    """

    identifiers: Optional[Sequence[str]] = None
    structures: Optional[Sequence[Structure]] = None
    diffraction_generator: Optional[DiffractionGenerator] = None
    reciprocal_radius: float = 0.0
    with_direct_beam: bool = False

    def get_library_entry(self, phase=None, angle=None):
        """Extracts a single DiffractionLibrary entry.

        Parameters
        ----------
        phase : str
            Key for the phase of interest. If unspecified the choice is random.
        angle : tuple
            The orientation of interest as a tuple of Euler angles following the
            Bunge convention [z, x, z] in degrees. If unspecified the choice is
            random (the first hit).

        Returns
        -------
        library_entries : dict
            Dictionary containing the simulation associated with the specified
            phase and orientation with associated properties.

        """
        if phase is not None:
            phase_entry = self[phase]
            if angle is not None:
                orientation_index = self._get_library_entry_from_angles(self, phase, angle)
            else:
                orientation_index = 0
        elif angle is not None:
            raise ValueError("To select a certain angle you must first specify a phase")
        else:
            phase_entry = next(iter(self.values()))
            orientation_index = 0

        return {
            "Sim": phase_entry["simulations"][orientation_index],
            "intensities": phase_entry["intensities"][orientation_index],
            "pixel_coords": phase_entry["pixel_coords"][orientation_index],
            "pattern_norm": np.linalg.norm(
                phase_entry["intensities"][orientation_index]
            ),
        }

    def save(self, filename: str) -> None:
        """Saves a diffraction library in the pickle format.

        Parameters
        ----------
        filename : str
            The location in which to save the file

        See Also
        --------
        DiffractionLibrary.load

        """
        with open(filename, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        filename: str,
        safety: bool = False
    ) -> DiffractionLibrary:
        """Loads a previously saved diffraction library.

        Parameters
        ----------
        filename
            The location of the file to be loaded.
        safety
            Unpickling is risky, this variable requires you to acknowledge
            this. Default is False.

        Returns
        -------
        DiffractionLibrary
            Previously saved Library.

        See Also
        --------
        DiffractionLibrary.save

        """
        if safety:
            with open(filename, "rb") as handle:
                return pickle.load(handle)
        else:
            raise RuntimeError(
                "Unpickling is risky, turn safety to True if you trust the author of this "
                "content"
            )

    @classmethod
    def _get_library_entry_from_angles(cls, phase, angles):
        """Finds an element that is orientation within 1e-2 of that
        specified.

        This is necessary because of floating point round off / hashability.
        If multiple entries satisfy the above criterion a random (the first
        hit) selection is made.

        Parameters
        ----------
        library : DiffractionLibrary
            The library to be searched.
        phase : str
            The phase of interest.
        angles : tuple
            The orientation of interest as a tuple of Euler angles in
            degrees, following the Bunge convention [z, x, z].

        Returns
        -------
        orientation_index : int
            Index of the given orientation.

        """

        phase_entry = library[phase]
        for orientation_index, orientation in enumerate(phase_entry["orientations"]):
            if np.sum(np.abs(np.subtract(orientation, angles))) < 1e-2:
                return orientation_index

        # We haven't found a suitable key
        raise ValueError(
            "It appears that no library entry lies with 1e-2 of the target angle"
        )
