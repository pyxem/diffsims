# -*- coding: utf-8 -*-
# Copyright 2017-2021 The diffsims developers
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
from orix.vector import Miller

from diffsims.structure_factor.structure_factor import (
    get_kinematical_structure_factor,
    get_doyleturner_structure_factor,
    get_refraction_corrected_wavelength,
)


_FLOAT_EPS = np.finfo(float).eps


class ReciprocalLatticeVector(Miller):
    """Reciprocal lattice vector (or crystal plane normal, reflector, g,
    etc.) with Miller indices, length of the reciprocal lattice vectors
    and other relevant diffraction parameters.

    This class extends :class:`orix.vector.Miller` to reciprocal
    lattice vectors specifically for diffraction experiments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinate_format = "hkl"
        self._structure_factor = [None] * self.size
        self._theta = [None] * self.size
        if self.phase is not None:
            self._raise_if_no_point_group()

    def __getitem__(self, key):
        v_new = super().__getitem__(key)
        if self.structure_factor[0] is None:
            v_new._structure_factor = [None] * v_new.size
        else:
            v_new._structure_factor = self.structure_factor[key]
        if self.theta[0] is None:
            v_new._theta = [None] * v_new.size
        else:
            v_new._theta = self.theta[key]
        return v_new

    def __repr__(self):
        """String representation."""
        name = self.__class__.__name__
        shape = self.shape
        symmetry = None if self.phase is None else self.phase.point_group.name
        data = np.array_str(self.coordinates, precision=4, suppress_small=True)
        phase_name = None if self.phase is None else self.phase.name
        return (
            f"{name} {shape}, {phase_name} ({symmetry})\n" f"{data}"
        )

    @property
    def gspacing(self):
        """Reciprocal lattice vector spacing *g* as
        :class:`numpy.ndarray`.
        """
        return self.phase.structure.lattice.rnorm(self.hkl.data)

    @property
    def dspacing(self):
        r"""Direct lattice interplanar spacing :math:`d = 1 / g` as
        :class:`numpy.ndarray`.
        """
        return 1 / self.gspacing

    @property
    def scattering_parameter(self):
        r"""Scattering parameter :math:`0.5 \cdot g` as
        :class:`numpy.ndarray`.
        """
        return 0.5 * self.gspacing

    @property
    def structure_factor(self):
        """Structure factors *F* as :class:`numpy.ndarray`, or None if
        :meth:`calculate_structure_factor` hasn't been called yet.
        """
        return self._structure_factor

    @property
    def theta(self):
        """Twice the Bragg angle as :class:`numpy.ndarray`, or None if
        :meth:`calculate_theta` hasn't been called yet.
        """
        return self._theta

    @property
    def allowed(self):
        """Return whether vectors diffract according to diffraction
        selection rules assuming kinematic scattering theory.
        """
        self._raise_if_no_space_group()

        # Translational symmetry
        centering = self.phase.space_group.short_name[0]

        if centering == "P":  # Primitive
            if self.phase.space_group.crystal_system == "HEXAGONAL":
                # TODO: See rules in e.g.
                #  https://mcl1.ncifcrf.gov/dauter_pubs/284.pdf, Table 4
                #  http://xrayweb.chem.ou.edu/notes/symmetry.html, Systematic Absences
                raise NotImplementedError
            else:  # Any hkl
                return np.ones(self.size, dtype=bool)
        elif centering == "F":  # Face-centred, hkl all odd/even
            selection = np.sum(np.mod(self.hkl, 2), axis=1)
            return np.array([i not in [1, 2] for i in selection], dtype=bool)
        elif centering == "I":  # Body-centred, h + k + l = 2n (even)
            return np.mod(np.sum(self.hkl, axis=1), 2) == 0
        elif centering == "A":  # Centred on A faces only
            return np.mod(self.k + self.l, 2) == 0
        elif centering == "B":  # Centred on B faces only
            return np.mod(self.h + self.l, 2) == 0
        elif centering == "C":  # Centred on C faces only
            return np.mod(self.h + self.k, 2) == 0
        elif centering in ["R", "H"]:  # Rhombohedral
            return np.mod(-self.h + self.k + self.l, 3) == 0

    def calculate_structure_factor(self, method="kinematical", voltage=None):
        """Populate `self.structure_factor` with the structure factor
        *F* for each vector.

        Parameters
        ----------
        method : str, optional
            Either "kinematical" (default) for kinematical X-ray
            structure factors or "doyleturner" for structure factors
            using Doyle-Turner atomic scattering factors.
        voltage : float, optional
            Beam energy in V used when `method=doyleturner`.
        """
        methods = ["kinematical", "doyleturner"]
        if method not in methods:
            raise ValueError(f"method={method} must be among {methods}")
        elif method == "doyleturner" and voltage is None:
            raise ValueError(
                "'voltage' parameter must be set when method='doyleturner'"
            )

        # TODO: Find a better way to call different methods in the loop
        factors = np.zeros(self.size)
        for i, (hkl, s) in enumerate(zip(self.hkl, self.scattering_parameter)):
            if method == "kinematical":
                factors[i] = get_kinematical_structure_factor(
                    phase=self.phase, hkl=hkl, scattering_parameter=s
                )
            else:
                factors[i] = get_doyleturner_structure_factor(
                    phase=self.phase, hkl=hkl, scattering_parameter=s, voltage=voltage
                )
        self._structure_factor = np.where(factors < _FLOAT_EPS, 0, factors)

    def calculate_theta(self, voltage):
        """Populate `self.theta` with the Bragg angle :math:`theta_B`
        for each vector.

        Parameters
        ----------
        voltage : float
            Beam energy in V.
        """
        wavelength = get_refraction_corrected_wavelength(self.phase, voltage)
        self._theta = np.arcsin(0.5 * wavelength * self.gspacing)

    def _raise_if_no_point_group(self):
        """Raise ValueError if the phase attribute has no point group
        set.
        """
        if self.phase.point_group is None:
            raise ValueError(f"The phase {self.phase} must have a point group set")

    def _raise_if_no_space_group(self):
        """Raise ValueError if the phase attribute has no space group
        set.
        """
        if self.phase.space_group is None:
            raise ValueError(f"The phase {self.phase} must have a space group set")
