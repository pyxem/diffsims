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
    get_refraction_corrected_wavelength,
)
from diffsims.utils.sim_utils import _get_kinematical_structure_factor


class ReciprocalLatticeVector(Miller):
    """Reciprocal lattice vector (or crystal plane normal, reflector, g,
    etc.) with Miller indices, length of the reciprocal lattice vectors
    and other relevant diffraction parameters.

    This class extends :class:`orix.vector.Miller` to reciprocal
    lattice vectors specifically for diffraction experiments.

    All lengths are assumed to be given in Ångströms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinate_format = "hkl"
        self._theta = np.full(self.size, np.nan)
        if self.phase is not None:
            self._raise_if_no_point_group()
        self._structure_factor = np.full(self.size, np.nan, dtype="complex128")

    def __getitem__(self, key):
        v_new = super().__getitem__(key)
        if np.isnan(self.structure_factor).all():
            v_new._structure_factor = np.full(v_new.size, np.nan, dtype="complex128")
        else:
            v_new._structure_factor = self.structure_factor[key]
        if np.isnan(self.theta).all():
            v_new._theta = np.full(v_new.size, np.nan)
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
        return self.phase.structure.lattice.rnorm(self.coordinates)

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

        hkl = np.atleast_2d(np.round(self.hkl).astype(int))

        if centering == "P":  # Primitive
            if self.phase.space_group.crystal_system == "HEXAGONAL":
                # TODO: See rules in e.g.
                #  https://mcl1.ncifcrf.gov/dauter_pubs/284.pdf, Table 4
                #  http://xrayweb.chem.ou.edu/notes/symmetry.html, Systematic Absences
                raise NotImplementedError
            else:  # Any hkl
                return np.ones(self.size, dtype=bool)
        elif centering == "F":  # Face-centred, hkl all odd/even
            selection = np.sum(np.mod(hkl, 2), axis=1)
            return np.array([i not in [1, 2] for i in selection], dtype=bool)
        elif centering == "I":  # Body-centred, h + k + l = 2n (even)
            return np.mod(np.sum(hkl, axis=1), 2) == 0
        elif centering == "A":  # Centred on A faces only
            return np.mod(hkl[:, 1] + hkl[:, 2], 2) == 0
        elif centering == "B":  # Centred on B faces only
            return np.mod(hkl[:, 0] + hkl[:, 2], 2) == 0
        elif centering == "C":  # Centred on C faces only
            return np.mod(hkl[:, 0] + hkl[:, 1], 2) == 0
        elif centering in ["R", "H"]:  # Rhombohedral
            return np.mod(-hkl[:, 0] + hkl[:, 1] + hkl[:, 2], 3) == 0

    def calculate_structure_factor(self, scattering_params="xtables"):
        """Populate `self.structure_factor` with the complex structure
        factor :math:`F_{hkl}` for each vector.

        Parameters
        ----------
        scattering_params : str
            Either "lobato" or "xtables".
        """
        # Reduce number of vectors to calculate the structure factor for
        v, inv = np.unique(self.coordinates, axis=0, return_inverse=True)

        structure_factor = _get_kinematical_structure_factor(
            structure=self.phase.structure,
            g_indices=v,
            g_hkls_array=self.phase.structure.lattice.rnorm(v),
            scattering_params=scattering_params,
        )

        self._structure_factor = structure_factor[inv]

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

    def symmetrise(self, **kwargs):
        return_index = kwargs.get("return_index", False)
        kwargs.setdefault("return_index", True)
        out = super().symmetrise(**kwargs)
        idx = out[-1]
        out[0]._structure_factor = self.structure_factor[idx]
        out[0]._theta = self.theta[idx]
        if return_index:
            return out
        else:
            out = out[:-1]
            if len(out) == 1:
                out = out[0]
            return out

    def unique(self, **kwargs):
        return_index = kwargs.get("return_index", False)
        kwargs.setdefault("return_index", True)
        out = super().unique(**kwargs)
        idx = out[-1][::-1]
        out[0]._structure_factor = self.structure_factor[idx]
        out[0]._theta = self.theta[idx]
        if return_index:
            return out
        else:
            out = out[:-1]
            if len(out) == 1:
                out = out[0]
            return out

    def print_table(self):
        # Column alignment
        align = "^"  # right ">", left "<", or centered "^"

        # Column widths
        width = 6
        no_width = width
        hkl_width = width
        d_width = width
        i_width = width
        f_hkl_width = width
        i_rel_width = width
        mult_width = width

        # Header (note the two-space spacing)
        data = (
            "{:{align}{width}}  ".format("No", width=no_width, align=align)
            + "{:{align}{width}}  ".format("H K L", width=hkl_width, align=align)
            + "{:{align}{width}}  ".format("d", width=d_width, align=align)
            + "{:{align}{width}}  ".format("I", width=i_width, align=align)
            + "{:{align}{width}}  ".format("|F|_HKL", width=f_hkl_width, align=align)
            + "{:{align}{width}}  ".format("I_Rel.", width=i_rel_width, align=align)
            + "{:{align}{width}}\n".format("Mult", width=mult_width, align=align)
        )

        v = self.unique(use_symmetry=True)
        structure_factor = v.structure_factor
        f_hkl = structure_factor.real
        intensity = (structure_factor * structure_factor.conjugate()).real
        order = np.argsort(intensity)
        v = v[order][::-1]
        f_hkl = f_hkl[order][::-1]
        intensity = intensity[order][::-1]

        size = v.size
        no = np.arange(1, size + 2)
        hkl = v.coordinates.round(2).astype(int)
        hkl_string = np.array_str(hkl).replace("[", "").replace("]", "").split("\n")
        d = v.dspacing
        intensity_rel = (intensity / intensity[0]) * 100
        mult = v.multiplicity

        for i in range(size):
            hkl_string_i = hkl_string[i].lstrip(" ")
            data += (
                f"{no[i]:{align}{no_width}}  "
                + f"{hkl_string_i:{align}{hkl_width}}  "
                + f"{d[i]:{align}{d_width}.3f}  "
                + f"{intensity[i]:{align}{i_width}.1f}  "
                + f"{f_hkl[i]:{align}{f_hkl_width}.1f}  "
                + f"{intensity_rel[i]:{align}{i_rel_width}.1f}  "
                + f"{mult[i]:{align}{mult_width}}"
            )
            if i != size - 1:
                data += "\n"

        print(data)

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
