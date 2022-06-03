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
from orix.vector import Miller, Vector3d
from orix.vector.miller import (
    _check_hkil,
    _get_highest_hkl,
    _get_indices_from_highest,
    _hkil2hkl,
    _hkl2hkil,
    _transform_space
)

from diffsims.structure_factor.structure_factor import (
    get_refraction_corrected_wavelength,
)
from diffsims.utils.sim_utils import _get_kinematical_structure_factor


class ReciprocalLatticeVector(Vector3d):
    r"""Reciprocal lattice vector (or crystal plane normal, reflector,
    :math:`g`, etc.) with Miller indices, length of the reciprocal
    lattice vectors and other relevant diffraction parameters.

    All lengths are assumed to be given in Ångströms.

    This class extends :class:`orix.vector.Vector3d` to reciprocal
    lattice vectors :math:`(hkl)` specifically for diffraction
    experiments and simulations. It is thus different from
    :class:`orix.vector.Miller`, which is a general class for Miller
    indices both in reciprocal *and* direct space. It supports relevant
    methods also supported in `Miller`, like obtaining a set of vectors
    from a minimal interplanar spacing.
    """

    def __init__(self, phase, hkl=None, hkil=None):
        r"""Create a set of reciprocal lattice vectors, :math:`(hkl)` or
        :math:`(hkil)`.

        Exactly one of ``hkl`` or ``hkil`` must be passed.

        The vectors are assumed to be integers and are stored internally
        as cartesian coordinates in the ``data`` attribute.

        Parameters
        ----------
        phase : orix.crystal_map.Phase
            A phase with a crystal lattice and symmetry.
        hkl : numpy.ndarray, list, or tuple, optional
            Indices of reciprocal lattice vector(s). Default is
            ``None``.
        hkil : numpy.ndarray, list, or tuple, optional
            Indices of reciprocal lattice vector(s), often preferred
            over ``hkl`` in trigonal and hexagonal lattices. Default is
            ``None``.
        """
        self.phase = phase
        self._raise_if_no_point_group()

        if np.sum([i is not None for i in [hkl, hkil]]) != 1:
            raise ValueError("Exactly one of `hkl`, `hkil` must be passed")
        elif hkil is not None:
            hkil = np.asarray(hkil)
            _check_hkil(hkil)
            hkl = _hkil2hkl(hkil)
            self._coordinate_format = "hkil"
        else:
            hkl = np.asarray(hkl)
            self._coordinate_format = "hkl"
        hkl = np.round(hkl).astype(float)
        xyz = _transform_space(hkl, "r", "c", phase.structure.lattice)
        super().__init__(xyz)

        self._theta = np.full(self.size, np.nan)
        self._structure_factor = np.full(self.size, np.nan, dtype="complex128")

    def __getitem__(self, key):
        miller_new = self._as_miller().__getitem__(key)
        rlv_new = self._from_miller(miller_new, self.coordinate_format)

        if np.isnan(self.structure_factor).all():
            rlv_new._structure_factor = np.full(
                rlv_new.size, np.nan, dtype="complex128"
            )
        else:
            rlv_new._structure_factor = self.structure_factor[key]

        if np.isnan(self.theta).all():
            rlv_new._theta = np.full(rlv_new.size, np.nan)
        else:
            rlv_new._theta = self.theta[key]

        return rlv_new

    def __repr__(self):
        """String representation."""
        name = self.__class__.__name__
        shape = self.shape
        symmetry = self.phase.point_group.name
        data = np.array_str(self.coordinates, precision=0, suppress_small=True)
        phase_name = self.phase.name
        return (
            f"{name} {shape}, {phase_name} ({symmetry})\n" f"{data}"
        )

    @property
    def hkl(self):
        """Miller indices."""
        hkl = _transform_space(self.data, "c", "r", self.phase.structure.lattice)
        return np.round(hkl).astype(float)

    @property
    def hkil(self):
        """Miller-Bravais indices."""
        return _hkl2hkil(self.hkl)

    @property
    def h(self):
        """First reciprocal lattice vector index."""
        return self.hkl[..., 0]

    @property
    def k(self):
        """Second reciprocal lattice vector index."""
        return self.hkl[..., 1]

    @property
    def i(self):
        r"""Third reciprocal lattice vector index in 4-index
        Miller-Bravais indices, equal to :math:`-(h + k)`.
        """
        return self.hkil[..., 2]

    @property
    def l(self):
        """Third reciprocal lattice vector index, or fourth index in
        4-index Miller Bravais indices.
        """
        return self.hkl[..., 2]

    @property
    def multiplicity(self):
        """Number of symmetrically equivalent directions per vector."""
        mult = self.symmetrise(return_multiplicity=True)[1]
        return mult.reshape(self.shape)

    @property
    def is_hexagonal(self):
        """Whether the crystal reference frame is hexagonal/trigonal or
        not.
        """
        return self.phase.is_hexagonal

    @property
    def coordinate_format(self):
        """Vector coordinate format, either ``"hkl"`` or ``"hkil"``."""
        return self._coordinate_format

    @property
    def coordinates(self):
        """Miller or Miller-Bravais indices."""
        return self.__getattribute__(self.coordinate_format)

    @property
    def gspacing(self):
        r"""Reciprocal lattice vector spacing :math:`g`."""
        return self.phase.structure.lattice.rnorm(self.hkl)

    @property
    def dspacing(self):
        r"""Direct lattice interplanar spacing :math:`d = 1 / g`."""
        return 1 / self.gspacing

    @property
    def scattering_parameter(self):
        r"""Scattering parameter :math:`0.5 \cdot g`."""
        return 0.5 * self.gspacing

    @property
    def structure_factor(self):
        r"""Structure factors :math:`F`, ``None`` if
        :meth:`calculate_structure_factor` hasn't been called yet.
        """
        return self._structure_factor

    @property
    def theta(self):
        """Twice the Bragg angle, ``None`` if :meth:`calculate_theta`
        hasn't been called yet.
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
            return np.mod(self.hkl[:, 1] + self.hkl[:, 2], 2) == 0
        elif centering == "B":  # Centred on B faces only
            return np.mod(self.hkl[:, 0] + self.hkl[:, 2], 2) == 0
        elif centering == "C":  # Centred on C faces only
            return np.mod(self.hkl[:, 0] + self.hkl[:, 1], 2) == 0
        elif centering in ["R", "H"]:  # Rhombohedral
            return np.mod(-self.hkl[:, 0] + self.hkl[:, 1] + self.hkl[:, 2], 3) == 0

    def calculate_structure_factor(self, scattering_params="xtables"):
        """Populate `self.structure_factor` with the complex structure
        factor :math:`F_{hkl}` for each vector.

        Parameters
        ----------
        scattering_params : str
            Either "lobato" or "xtables".
        """
        # Reduce number of vectors to calculate the structure factor for
        # TODO: Use symmetry
        hkl, inv = np.unique(self.hkl, axis=0, return_inverse=True)

        structure_factor = _get_kinematical_structure_factor(
            structure=self.phase.structure,
            g_indices=hkl,
            g_hkls_array=self.phase.structure.lattice.rnorm(hkl),
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

    def symmetrise(self, return_multiplicity=False, return_index=False):
        """Unique vectors symmetrically equivalent to the ones in
        ``self``.

        Parameters
        ----------
        return_multiplicity : bool, optional
            Whether to return the multiplicity of each vector. Default
            is ``False``.
        return_index : bool, optional
            Whether to return the index into ``self`` for the returned
            symmetrically equivalent vectors. Default is ``False``.

        Returns
        -------
        ReciprocalLatticeVector
            Flattened symmetrically equivalent vectors.
        multiplicity : numpy.ndarray
            Multiplicity of each vector. Returned if
            ``return_multiplicity=True``.
        idx : numpy.ndarray
            Index into ``self`` for the symmetrically equivalent
            vectors. Returned if ``return_index=True``.
        """
        out = self._as_miller().symmetrise(
            unique=True,
            return_multiplicity=return_multiplicity,
            return_index=True
        )

        if return_multiplicity:
            miller, mult, idx = out
        else:
            miller, idx = out

        new_rlv = self._from_miller(miller, self.coordinate_format)
        new_rlv._structure_factor = self.structure_factor[idx]
        new_rlv._theta = self.theta[idx]

        new_out = (new_rlv,)
        if return_multiplicity:
            new_out += (mult,)
        if return_index:
            new_out += (idx,)
        if len(new_out) == 1:
            return new_out[0]
        else:
            return new_out

    def unique(self, use_symmetry=False, return_index=False):
        """Unique vectors in ``self``.

        Parameters
        ----------
        use_symmetry : bool, optional
            Whether to consider equivalent vectors to compute the unique
            vectors. Default is ``False``.
        return_index : bool, optional
            Whether to return the indices of the (flattened) data where
            the unique entries were found. Default is ``False``.

        Returns
        -------
        ReciprocalLatticeVector
            Flattened unique vectors.
        idx : numpy.ndarray
            Indices of the unique data in the (flattened) array.
        """
        kwargs = dict(use_symmetry=use_symmetry, return_index=True)
        miller, idx = self._as_miller().unique(**kwargs)
        idx = idx[::-1]

        new_rlv = self._from_miller(miller, self.coordinate_format)
        new_rlv._structure_factor = self.structure_factor[idx]
        new_rlv._theta = self.theta[idx]

        if return_index:
            return new_rlv, idx
        else:
            return new_rlv

    def print_table(self):
        """Table with indices, structure factor values and multiplicity.
        """
        # Column alignment
        align = "<"  # right ">", left "<", or centered "^"

        # Column widths
        width = 6
        hkl_width = width
        d_width = width
        i_width = width
        f_hkl_width = width + 1
        i_rel_width = width
        mult_width = width

        # Header (note the two-space spacing)
        data = (
            "{:{align}{width}}  ".format("h k l", width=hkl_width, align=align)
            + "{:{align}{width}}  ".format("d", width=d_width, align=align)
            + "{:{align}{width}}  ".format("I", width=i_width, align=align)
            + "{:{align}{width}}  ".format("|F|_hkl", width=f_hkl_width, align=align)
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
        hkl = v.coordinates.round(2).astype(int)
        hkl_string = np.array_str(hkl).replace("[", "").replace("]", "").split("\n")
        d = v.dspacing
        intensity_rel = (intensity / intensity[0]) * 100
        mult = v.multiplicity

        for i in range(size):
            hkl_string_i = hkl_string[i].lstrip(" ")
            data += (
                f"{hkl_string_i:{align}{hkl_width}}  "
                + f"{d[i]:{align}{d_width}.3f}  "
                + f"{intensity[i]:{align}{i_width}.1f}  "
                + f"{f_hkl[i]:{align}{f_hkl_width}.1f}  "
                + f"{intensity_rel[i]:{align}{i_rel_width}.1f}  "
                + f"{mult[i]:{align}{mult_width}}"
            )
            if i != size - 1:
                data += "\n"

        print(data)

    @classmethod
    def from_highest_hkl(cls, phase, hkl):
        """Create a set of unique reciprocal lattice vectors from three
        highest indices and a phase (crystal lattice and symmetry).

        Parameters
        ----------
        phase : orix.crystal_map.Phase
            A phase with a crystal lattice and symmetry.
        hkl : numpy.ndarray, list, or tuple
            Three highest reciprocal lattice vector indices.
        """
        idx = _get_indices_from_highest(highest_indices=hkl)
        return cls(phase, hkl=idx).unique()

    @classmethod
    def from_min_dspacing(cls, phase, min_dspacing=0.7):
        """Create a set of unique reciprocal lattice vectors with a
        a direct space interplanar spacing greater than a lower
        threshold.

        Parameters
        ----------
        phase : orix.crystal_map.Phase
            A phase with a crystal lattice and symmetry.
        min_dspacing : float, optional
            Smallest interplanar spacing to consider. Default is 0.7,
            in the unit used to define the lattice parameters in
            ``phase``, which is assumed to be Ångström.
        """
        highest_hkl = _get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = _get_indices_from_highest(highest_indices=highest_hkl)
        hkl = np.round(hkl).astype(float)
        dspacing = 1 / phase.structure.lattice.rnorm(hkl)
        idx = dspacing >= min_dspacing
        hkl = hkl[idx]
        return cls(phase, hkl=hkl).unique()

    def _as_miller(self):
        """Return ``self`` as a ``Miller`` instance.

        Returns
        -------
        orix.vector.Miller
        """
        if self.coordinate_format == "hkl":
            return Miller(hkl=self.hkl, phase=self.phase)
        else:
            return Miller(hkil=self.hkil, phase=self.phase)

    @classmethod
    def _from_miller(cls, miller, coordinate_format):
        """Create a new instance from a ``Miller`` instance.

        Parameters
        ----------
        miller : orix.vector.Miller
        coordinate_format : str
            Either ``"hkl"`` or ``"hkil"``.

        Returns
        -------
        ReciprocalLatticeVector
        """
        if coordinate_format == "hkl":
            return cls(miller.phase, hkl=miller.hkl)
        else:
            return cls(miller.phase, hkil=miller.hkil)

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
