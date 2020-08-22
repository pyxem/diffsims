# -*- coding: utf-8 -*-
# Copyright 2017-2020 The diffsims developers
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

from collections import defaultdict
from itertools import product

import numpy as np
from orix.vector import Vector3d

from diffsims.reciprocal_lattice_point.structure_factor import (
    get_xray_structure_factor
)


class ReciprocalLatticePoint:
    """Reciprocal lattice points (reflectors) g with Miller indices,
    length of the reciprocal lattice vectors and other relevant
    diffraction parameters.
    """

    def __init__(self, phase, hkl):
        """A container for Miller indices, structure factors and related
        parameters for reciprocal lattice points (reflectors) g.

        Parameters
        ----------
        phase : orix.crystal_map.phase_list.Phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        hkl : orix.vector.Vector3d, np.ndarray, list, or tuple
            Miller indices.
        """
        self._hkl = Vector3d(hkl)
        self.phase = phase
        self._structure_factor = [None] * self.size

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self.hkl.shape}\n"
            f"Phase: {self.phase.name} ({self.phase.point_group.name})\n"
            f"{np.array_str(self.hkl.data, precision=4, suppress_small=True)}"
        )

    def __getitem__(self, key):
        new_rlp = self.__class__(self.phase, self.hkl[key])
        new_rlp._structure_factor = self.structure_factor[key]
        return new_rlp

    @property
    def hkl(self):
        """Return :class:`~orix.vector.Vector3d` of Miller indices."""
        return Vector3d(self._hkl.data.astype(int))

    @property
    def _hkldata(self):
        """Return :class:`np.ndarray` without 1-dimensions."""
        return np.squeeze(self.hkl.data)

    @property
    def size(self):
        """Return `int`."""
        return self.hkl.size

    @property
    def shape(self):
        """Return `tuple`."""
        return self._hkldata.shape

    @property
    def multiplicity(self):
        """Return either `int` or :class:`np.ndarray` of `int`."""
        return self.symmetrise(antipodal=True, return_multiplicity=True)[1]

    @property
    def gspacing(self):
        """Return :class:`np.ndarray` of reciprocal lattice point
        spacings.
        """
        return self.phase.structure.lattice.rnorm(self._hkldata)

    @property
    def dspacing(self):
        """Return :class:`np.ndarray` of direct lattice interplanar
        spacings.
        """
        return 1 / self.gspacing

    @property
    def scattering_parameter(self):
        """Return :class:`np.ndarray` of scattering parameters s."""
        return 0.5 * self.gspacing

    @property
    def structure_factor(self):
        """Return :class:`np.ndarray` of structure factors F or None."""
        return self._structure_factor

    @classmethod
    def from_min_dspacing(cls, phase, min_dspacing=0.5):
        """Create a ReciprocalLatticePoint object populated by unique
        Miller indices with a direct space interplanar spacing greater
        than a lower threshold.

        Parameters
        ----------
        phase : orix.crystal_map.phase_list.Phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        min_dspacing : float, optional
            Smallest interplanar spacing to consider. Default is 0.5 Å.
        """
        highest_hkl = get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = get_hkl(highest_hkl=highest_hkl)
        return cls(phase=phase, hkl=hkl).unique()

    @classmethod
    def from_highest_hkl(cls, phase, highest_hkl):
        """Create a ReciprocalLatticePoint object populated by unique
        Miller indices below, but including, a set of higher indices.

        Parameters
        ----------
        phase : orix.crystal_map.phase_list.Phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        highest_hkl : np.ndarray, list, or tuple of int
            Highest Miller indices to consider (including).
        """
        hkl = get_hkl(highest_hkl=highest_hkl)
        return cls(phase=phase, hkl=hkl).unique()

    @classmethod
    def from_nfamilies(cls, phase, nfamilies=5):
        raise NotImplementedError

    def calculate_structure_factor(self, method=None):
        """Populate `self.structure_factor` with the structure factor F
        for each point.

        Parameters
        ----------
        method
            Either "xray" for the X-ray structure factor or
            "doyleturner" for the structure factor using Doyle-Turner
            atomic scattering factors.
        """
        structure_factors = np.zeros(self.size)
        hkls = self._hkldata
        scattering_parameters = self.scattering_parameter
        for i, (hkl, s) in enumerate(zip(hkls, scattering_parameters)):
            structure_factors[i] = get_xray_structure_factor(
                phase=self.phase, hkl=hkl, scattering_parameter=s
            )
        self._structure_factor = structure_factors

    def unique(self, use_symmetry=True):
        """Return reciprocal lattice points with unique Miller indices.

        Parameters
        ----------
        use_symmetry : bool, optional
            Whether to use symmetry to remove the indices symmetrically
            equivalent to another set of indices.

        Returns
        -------
        ReciprocalLatticePoint
        """
        if use_symmetry:
            all_hkl = self._hkldata
            all_hkl = all_hkl[~np.all(np.isclose(all_hkl, 0), axis=1)]
            families = defaultdict(list)
            for this_hkl in all_hkl.tolist():
                for that_hkl in families.keys():
                    if is_equivalent(this_hkl, that_hkl):
                        families[tuple(that_hkl)].append(this_hkl)
                        break
                else:
                    families[tuple(this_hkl)].append(this_hkl)

            n_families = len(families)
            unique_hkl = np.zeros((n_families, 3))
            for i, all_hkl_in_family in enumerate(families.values()):
                unique_hkl[i] = sorted(all_hkl_in_family)[-1]
        else:
            unique_hkl = self.hkl.unique()
        return self.__class__(phase=self.phase, hkl=unique_hkl)

    def symmetrise(
        self, antipodal=True, unique=True, return_multiplicity=False,
    ):
        """Return reciprocal lattice points with symmetrically equivalent
        Miller indices.

        Parameters
        ----------
        antipodal : bool, optional
            Whether to include antipodal symmetry operations. Default is
            True.
        unique : bool, optional
            Whether to return only distinct indices. Default is True.
            If true, zero entries which are assumed to be degenerate are
            removed.
        return_multiplicity : bool, optional
            Whether to return the multiplicity of the indices. This
            option is only available if `unique` is True. Default is
            False.

        Returns
        -------
        ReciprocalLatticePoint
            Reciprocal lattice points with Miller indices symmetrically
            equivalent to the original lattice points.
        multiplicity : np.ndarray
            Multiplicity of the original Miller indices. Only returned if
            `return_multiplicity` is True.

        Notes
        -----
        Should be the same as EMsoft's CalcFamily in their symmetry.f90
        module.
        """
        # Get symmetry operations
        pg = self.phase.point_group
        operations = pg[~pg.improper] if not antipodal else pg

        out = get_equivalent_hkl(
            hkl=self.hkl,
            operations=operations,
            unique=unique,
            return_multiplicity=return_multiplicity,
        )

        # Format output and return
        if unique and return_multiplicity:
            multiplicity = out[1]
            if multiplicity.size == 1:
                multiplicity = multiplicity[0]
            return self.__class__(phase=self.phase, hkl=out[0]), multiplicity
        else:
            return self.__class__(phase=self.phase, hkl=out)


def get_highest_hkl(lattice, min_dspacing=0.5):
    """Return the highest Miller indices hkl of the reciprocal
    lattice point with a direct space interplanar spacing greater
    than but closest to a lower threshold.

    Parameters
    ----------
    lattice : diffpy.structure.Lattice
        Crystal structure lattice.
    min_dspacing : float, optional
        Smallest interplanar spacing to consider. Default is 0.5 Å.

    Returns
    -------
    highest_hkl : np.ndarray
        Highest Miller indices.
    """
    highest_hkl = np.ones(3, dtype=int)
    for i in range(3):
        hkl = np.zeros(3)
        d = min_dspacing + 1
        while d > min_dspacing:
            hkl[i] += 1
            d = 1 / lattice.rnorm(hkl)
        highest_hkl[i] = hkl[i]
    return highest_hkl


def get_hkl(highest_hkl):
    """Return a list of reciprocal lattice points from a set of highest
    Miller indices.

    Parameters
    ----------
    highest_hkl : orix.vector.Vector3d, np.ndarray, list, or tuple of int
        Highest Miller indices to consider.

    Returns
    -------
    hkl
        An array of reciprocal lattice points.
    """
    index_ranges = [np.arange(-i, i + 1) for i in highest_hkl]
    return np.asarray(list(product(*index_ranges)))


def get_equivalent_hkl(
    hkl, operations, unique=False, return_multiplicity=False
):
    """Return symmetrically equivalent Miller indices.

    Parameters
    ----------
    hkl : orix.vector.Vector3d, np.ndarray, list or tuple of int
        Miller indices.
    operations : orix.quaternion.symmetry.Symmetry
        Point group describing allowed symmetry operations.
    unique : bool, optional
        Whether to return only unique Miller indices. Default is False.
    return_multiplicity : bool, optional
        Whether to return the multiplicity of the input hkl. Default is
        False.

    Returns
    -------
    new_hkl : orix.vector.Vector3d
        The symmetrically equivalent Miller indices.
    multiplicity : np.ndarray
        Number of symmetrically equivalent indices. Only returned if
        `return_multiplicity` is True.
    """
    new_hkl = operations.outer(Vector3d(hkl))
    new_hkl = new_hkl.flatten().reshape(*new_hkl.shape[::-1])

    multiplicity = None
    if unique:
        n_families = new_hkl.shape[0]
        multiplicity = np.zeros(n_families, dtype=int)
        temp_hkl = new_hkl[0].unique().data
        multiplicity[0] = temp_hkl.shape[0]
        if n_families > 1:
            for i, hkl in enumerate(new_hkl[1:]):
                temp_hkl2 = hkl.unique()
                multiplicity[i + 1] = temp_hkl2.size
                temp_hkl = np.append(temp_hkl, temp_hkl2.data, axis=0)
        new_hkl = Vector3d(temp_hkl[: multiplicity.sum()])

    # Remove 1-dimensions
    new_hkl = new_hkl.squeeze()

    if unique and return_multiplicity:
        return new_hkl, multiplicity
    else:
        return new_hkl


def is_equivalent(this_hkl: list, that_hkl: list) -> bool:
    return sorted(np.abs(this_hkl)) == sorted(np.abs(that_hkl))
