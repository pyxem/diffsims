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

from copy import deepcopy
from functools import cached_property

import numpy as np
from orix.vector import Miller, Vector3d
from orix.vector.miller import (
    _check_hkil,
    _get_highest_hkl,
    _get_indices_from_highest,
    _hkil2hkl,
    _hkl2hkil,
    _transform_space,
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

    def __init__(self, phase, xyz=None, hkl=None, hkil=None):
        r"""Create a set of reciprocal lattice vectors from
        :math:`(hkl)` or :math:`(hkil)`.

        The vectors are internally as cartesian coordinates in the
        ``data`` attribute.

        Parameters
        ----------
        phase : orix.crystal_map.Phase
            A phase with a crystal lattice and symmetry.
        xyz : numpy.ndarray, list, or tuple, optional
            Cartesian coordinates of indices of reciprocal lattice
            vector(s) ``hkl``. Default is ``None``. This, ``hkl``, or
            ``hkil`` is required.
        hkl : numpy.ndarray, list, or tuple, optional
            Indices of reciprocal lattice vector(s). Default is
            ``None``. This, ``xyz``, or ``hkil`` is required.
        hkil : numpy.ndarray, list, or tuple, optional
            Indices of reciprocal lattice vector(s), often preferred
            over ``hkl`` in trigonal and hexagonal lattices. Default is
            ``None``. This, ``xyz``, or ``hkl`` is required.
        """
        self.phase = phase
        self._raise_if_no_point_group()

        if np.sum([i is not None for i in [xyz, hkl, hkil]]) != 1:
            raise ValueError("Exactly one of `xyz`, `hkl`, or `hkil` must be passed")
        elif xyz is not None:
            xyz = np.asarray(xyz)
            self._coordinate_format = "hkl"
        elif hkil is not None:
            hkil = np.asarray(hkil)
            _check_hkil(hkil)
            hkl = _hkil2hkl(hkil)
            self._coordinate_format = "hkil"
            xyz = _transform_space(hkl, "r", "c", phase.structure.lattice)
        else:
            hkl = np.asarray(hkl)
            self._coordinate_format = "hkl"
            xyz = _transform_space(hkl, "r", "c", phase.structure.lattice)
        super().__init__(xyz)

        self._theta = np.full(self.shape, np.nan)
        self._structure_factor = np.full(self.shape, np.nan, dtype="complex128")

    def __getitem__(self, key):
        miller_new = self.to_miller().__getitem__(key)
        rlv_new = self.from_miller(miller_new)

        if np.isnan(self.structure_factor).all():
            rlv_new._structure_factor = np.full(
                rlv_new.shape, np.nan, dtype="complex128"
            )
        else:
            rlv_new._structure_factor = self.structure_factor[key]

        if np.isnan(self.theta).all():
            rlv_new._theta = np.full(rlv_new.shape, np.nan)
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
        return f"{name} {shape}, {phase_name} ({symmetry})\n" f"{data}"

    @cached_property
    def hkl(self):
        """Miller indices."""
        return _transform_space(self.data, "c", "r", self.phase.structure.lattice)

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

    @cached_property
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

    @coordinate_format.setter
    def coordinate_format(self, value):
        """Set the vector coordinate format, either ``"hkl"``, or
        ``"hkil"``.
        """
        formats = ["hkl", "hkil"]
        if value not in formats:
            raise ValueError(f"Available coordinate formats are {formats}")
        self._coordinate_format = value

    @property
    def coordinates(self):
        """Miller or Miller-Bravais indices."""
        return self.__getattribute__(self.coordinate_format)

    @cached_property
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

    @cached_property
    def allowed(self):
        """Return whether vectors diffract according to diffraction
        selection rules assuming kinematic scattering theory.
        """
        self._raise_if_no_space_group()

        # Translational symmetry
        centering = self.phase.space_group.short_name[0]

        if centering not in ["P", "I", "A", "B", "C", "R", "H", "F"]:
            raise ValueError(f"Unknown unit cell centering {centering}")
        else:
            hkl = self.hkl.reshape((-1, 3))

        if centering == "P":  # Primitive
            if self.is_hexagonal:
                # TODO: See rules in e.g.
                #  https://mcl1.ncifcrf.gov/dauter_pubs/284.pdf, Table 4
                #  http://xrayweb.chem.ou.edu/notes/symmetry.html, Systematic Absences
                raise NotImplementedError
            else:  # Any hkl
                is_allowed = np.ones(self.size, dtype=bool)
        elif centering == "I":  # Body-centred, h + k + l = 2n (even)
            is_allowed = np.mod(np.sum(hkl, axis=1), 2) == 0
        elif centering == "A":  # Centred on A faces only
            is_allowed = np.mod(hkl[:, 1] + hkl[:, 2], 2) == 0
        elif centering == "B":  # Centred on B faces only
            is_allowed = np.mod(hkl[:, 0] + hkl[:, 2], 2) == 0
        elif centering == "C":  # Centred on C faces only
            is_allowed = np.mod(hkl[:, 0] + hkl[:, 1], 2) == 0
        elif centering in ["R", "H"]:  # Rhombohedral
            is_allowed = np.mod(-hkl[:, 0] + hkl[:, 1] + hkl[:, 2], 3) == 0
        else:  # "F", face-centred, hkl all odd/even
            selection = np.sum(np.mod(hkl, 2), axis=1)
            is_allowed = np.array([i not in [1, 2] for i in selection], dtype=bool)

        return is_allowed.reshape(self.shape)

    # ------------------------- Custom methods ----------------------- #

    def calculate_structure_factor(self, scattering_params="xtables"):
        """Populate `self.structure_factor` with the complex structure
        factor :math:`F_{hkl}` for each vector.

        Parameters
        ----------
        scattering_params : str
            Either "lobato" or "xtables".
        """
        # Reduce number of vectors to calculate the structure factor for
        # TODO: Use symmetry to contract vectors and then expand factors
        hkl = self.hkl.reshape((-1, 3))
        hkl_unique, inv = np.unique(hkl, axis=0, return_inverse=True)

        structure_factor = _get_kinematical_structure_factor(
            structure=self.phase.structure,
            g_indices=hkl_unique,
            g_hkls_array=self.phase.structure.lattice.rnorm(hkl_unique),
            scattering_params=scattering_params,
        )

        self._structure_factor = structure_factor[inv].reshape(self.shape)

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

    def deepcopy(self):
        """Return a deepcopy of the vectors."""
        return deepcopy(self)

    def print_table(self):
        """Table with indices, structure factor values and multiplicity."""
        # Column alignment
        align = "^"  # right ">", left "<", or centered "^"

        # Column widths
        width = 6
        hkl_width = width + 2
        d_width = width
        i_width = width
        f_hkl_width = width + 1
        i_rel_width = width
        mult_width = width

        # Header (note the two-space spacing)
        data = (
            "{:{align}{width}}  ".format(" h k l ", width=hkl_width, align=align)
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
        hkl = np.round(v.coordinates).astype(int)
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
        out = self.to_miller().symmetrise(
            unique=True, return_multiplicity=return_multiplicity, return_index=True
        )

        if return_multiplicity:
            miller, mult, idx = out
        else:
            miller, idx = out

        new_rlv = self.from_miller(miller)
        new_rlv._structure_factor = self.structure_factor.ravel()[idx]
        new_rlv._theta = self.theta.ravel()[idx]

        new_out = (new_rlv,)
        if return_multiplicity:
            new_out += (mult,)
        if return_index:
            new_out += (idx,)
        if len(new_out) == 1:
            return new_out[0]
        else:
            return new_out

    @classmethod
    def from_highest_hkl(cls, phase, hkl):
        """Create a set of unique reciprocal lattice vectors from three
        highest indices.

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
        dspacing = 1 / phase.structure.lattice.rnorm(hkl)
        idx = dspacing >= min_dspacing
        hkl = hkl[idx]
        return cls(phase, hkl=hkl).unique()

    @classmethod
    def from_miller(cls, miller):
        r"""Create a new instance from a ``Miller`` instance.

        Parameters
        ----------
        miller : orix.vector.Miller
            Reciprocal lattice vectors :math:`(hk(i)l)`.

        Returns
        -------
        ReciprocalLatticeVector
        """
        if miller.coordinate_format not in ["hkl", "hkil"]:
            raise ValueError(
                "`Miller` instance must have `coordinate_format` 'hkl' or 'hkil'"
            )
        return cls(miller.phase, **{miller.coordinate_format: miller.coordinates})

    def to_miller(self):
        """Return ``self`` as a ``Miller`` instance.

        Returns
        -------
        orix.vector.Miller
        """
        return Miller(phase=self.phase, **{self.coordinate_format: self.coordinates})

    def _compatible_with(self, other, raise_error=False):
        """Whether ``self`` and ``other`` are the same (the same crystal
        lattice and symmetry) with vectors in the same space.

        Parameters
        ----------
        other : ReciprocalLatticeVector
        raise_error : bool, optional
            Whether to raise a ``ValueError`` if the instances are
            incompatible (default is False).

        Returns
        -------
        bool
        """
        miller1 = self.to_miller()
        miller2 = other.to_miller()
        return miller1._compatible_with(miller2, raise_error=raise_error)

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

    def _update_shapes(self):
        """Update shapes of properties."""
        self._theta = self._theta.reshape(self.shape)
        self._structure_factor = self._structure_factor.reshape(self.shape)

    # ---------- Overwritten Vector3d properties and methods --------- #

    def angle_with(self, other, use_symmetry=False):
        """Calculate angles between reciprocal lattice vectors, possibly
        using symmetrically equivalent vectors to find the smallest
        angle under symmetry.

        Parameters
        ----------
        other : ReciprocalLatticeVector
            Vectors of compatible shape to ``self``.
        use_symmetry : bool, optional
            Whether to consider equivalent vectors to find the smallest
            angle under symmetry. Default is ``False``.

        Returns
        -------
        numpy.ndarray
            The angle between the vectors, in radians. If
            ``use_symmetry=True``, the angles are the smallest under
            symmetry.
        """
        self._compatible_with(other, raise_error=True)
        miller1 = self.to_miller()
        miller2 = other.to_miller()
        return miller1.angle_with(miller2, use_symmetry=use_symmetry)

    def cross(self, other):
        r"""Cross product between reciprocal lattice vectors producing
        zone axes :math:`[uvw]` or :math:`[UVTW]` in the direct lattice.

        Parameters
        ----------
        other : ReciprocalLatticeVector
            Vectors of compatible shape to ``self``.

        Returns
        -------
        orix.vector.Miller
            Direct lattice vector(s) :math:`[uvw]` or :math:`UVTW`,
            depending on whether ``self.coordinate_format`` is ``hkl``
            or ``hkil``, respectively.
        """
        miller = self.to_miller().cross(other.to_miller())
        new_format = {"hkl": "uvw", "hkil": "UVTW"}
        miller.coordinate_format = new_format[self.coordinate_format]
        return miller

    def dot(self, other):
        """Dot product of all reciprocal lattice vectors in ``self``
        with other reciprocal lattice vectors.

        Parameters
        ----------
        other : ReciprocalLatticeVector
            Vectors of compatible shape to ``self``.

        Returns
        -------
        numpy.ndarray
        """
        self._compatible_with(other, raise_error=True)
        return super().dot(other)

    def dot_outer(self, other):
        """Outer dot product of all reciprocal lattice vectors in
        ``self`` with other reciprocal lattice vectors.

        The dot product for every combination of vectors in ``self`` and
        ``other`` is computed.

        Parameters
        ----------
        other : ReciprocalLatticeVector
            Vectors of compatible shape to ``self``.

        Returns
        -------
        numpy.ndarray
        """
        self._compatible_with(other, raise_error=True)
        return super().dot_outer(other)

    def get_nearest(self, *args, **kwargs):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    def in_fundamental_sector(self, symmetry=None):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    def mean(self):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    def rotate(self, *args, **kwargs):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    @classmethod
    def from_polar(cls, azimuth, polar, radial=1):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    @classmethod
    def xvector(cls):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    @classmethod
    def yvector(cls):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    @classmethod
    def zvector(cls):
        raise NotImplementedError("Use `orix.vector.Miller` instead.")

    # ---------- Overwritten Object3d properties and methods --------- #

    @property
    def unit(self):
        """Unit reciprocal lattice vectors.

        Returns
        -------
        ReciprocalLatticeVector
        """
        miller = self.to_miller()
        return self.from_miller(miller.unit)

    def flatten(self):
        """A new instance with these reciprocal lattice vectors in a
        single column.

        Returns
        -------
        ReciprocalLatticeVector
        """
        miller = self.to_miller()
        new = self.from_miller(miller.flatten())
        new._structure_factor = self._structure_factor.reshape(new.shape)
        new._theta = self._theta.copy()
        new._update_shapes()
        return new

    def reshape(self, *shape):
        """A new instance with these reciprocal lattice vectors
        reshaped.

        Parameters
        ----------
        *shape : int
            Multiple integers designating the new shape.

        Returns
        -------
        ReciprocalLatticeVector
        """
        miller = self.to_miller()
        new = self.from_miller(miller.reshape(*shape))
        new._structure_factor = self._structure_factor.copy()
        new._theta = self._theta.copy()
        new._update_shapes()
        return new

    def squeeze(self):
        """A new instance with these reciprocal lattice vectors where
        singleton dimensions are removed.

        Returns
        -------
        ReciprocalLatticeVector
        """
        v = Vector3d(self.data).squeeze()
        new = self.__class__(phase=self.phase, xyz=v.data)
        new._coordinate_format = self.coordinate_format
        new._structure_factor = self._structure_factor.copy()
        new._theta = self._theta.copy()
        new._update_shapes()
        return new

    def transpose(self, *axes):
        """A new instance with the navigation shape of these reciprocal
        lattice vectors transposed.

        If ``self.ndim`` is originally 2, then order may be undefined.
        In this case the first two dimensions will be transposed.

        Parameters
        ----------
        *axes : int, optional
            Transposed axes order. Only navigation axes need to be
            defined. May be undefined if ``self`` only contains two
            navigation dimensions.

        Returns
        -------
        ReciprocalLatticeVector
        """
        miller = self.to_miller()
        new = self.from_miller(miller.transpose(*axes))
        new._structure_factor = self._structure_factor.copy()
        new._theta = self._theta.copy()
        new._update_shapes()
        return new

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
        miller, idx = self.to_miller().unique(**kwargs)
        idx = idx[::-1]

        new_rlv = self.from_miller(miller)
        new_rlv._structure_factor = self.structure_factor.ravel()[idx]
        new_rlv._theta = self.theta.ravel()[idx]

        if return_index:
            return new_rlv, idx
        else:
            return new_rlv

    @classmethod
    def empty(cls):
        raise NotImplementedError

    @classmethod
    def stack(cls, sequence):
        """A new instance from a sequence of reciprocal lattice vectors.

        Parameters
        ----------
        sequence : iterable of ReciprocalLatticeVector
            One or more sets of compatible reciprocal lattice vectors.

        Returns
        -------
        ReciprocalLatticeVector
        """
        # Check instance compatibility. A ValueError is raised in the
        # loop if instances are incompatible.
        sequence = tuple(sequence)  # Make iterable
        if len(sequence) > 1:
            s0 = sequence[0]
            for s in sequence[1:]:
                s0._compatible_with(s, raise_error=True)

        v = Vector3d.stack(sequence)
        new = cls(xyz=v.data, phase=sequence[0].phase)
        new.coordinate_format = sequence[0].coordinate_format

        return new
