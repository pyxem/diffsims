# -*- coding: utf-8 -*-
# Copyright 2017-2024 The diffsims developers
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

from diffsims.crystallography import ReciprocalLatticeVector
import numpy as np


class DiffractingVector(ReciprocalLatticeVector):
    r"""Reciprocal lattice vectors :math:`(hkl)` for use in electron
    diffraction analysis and simulation.

    All lengths are assumed to be given in Å or inverse Å.

    This extends the :class:`ReciprocalLatticeVector` class.  Diffracting Vectors
    focus on the subset of reciporical lattice vectors that are relevant for
    electron diffraction based on the intersection of the Ewald sphere with the
    reciprocal lattice.

    Parameters
    ----------
    phase : orix.crystal_map.Phase
        A phase with a crystal lattice and symmetry.
    xyz : numpy.ndarray, list, or tuple, optional
        Cartesian coordinates of indices of reciprocal lattice vector(s)
        ``hkl``. Default is ``None``. This, ``hkl``, or ``hkil`` is
        required.
    hkl : numpy.ndarray, list, or tuple, optional
        Indices of reciprocal lattice vector(s). Default is ``None``.
        This, ``xyz``, or ``hkil`` is required.
    hkil : numpy.ndarray, list, or tuple, optional
        Indices of reciprocal lattice vector(s), often preferred over
        ``hkl`` in trigonal and hexagonal lattices. Default is ``None``.
        This, ``xyz``, or ``hkl`` is required.

    Examples
    --------
    >>> from diffpy.structure import Atom, Lattice, Structure
    >>> from orix.crystal_map import Phase
    >>> from diffsims.crystallography import DiffractingVector
    >>> phase = Phase(
    ...     "al",
    ...     space_group=225,
    ...     structure=Structure(
    ...         lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
    ...         atoms=[Atom("Al", [0, 0, 1])],
    ...     ),
    ... )
    >>> rlv = DiffractingVector(phase, hkl=[[1, 1, 1], [2, 0, 0]])
    >>> rlv
    ReciprocalLatticeVector (2,), al (m-3m)
    [[1. 1. 1.]
     [2. 0. 0.]]

    """

    def __init__(self, phase, xyz=None, hkl=None, hkil=None, intensity=None):
        super().__init__(phase, xyz=xyz, hkl=hkl, hkil=hkil)
        if intensity is None:
            self._intensity = np.full(self.shape, np.nan)
        elif len(intensity) != self.size:
            raise ValueError("Length of intensity array must match number of vectors")
        else:
            self._intensity = np.array(intensity)

    def __getitem__(self, key):
        dv_new = super().__getitem__(key)
        if np.isnan(self.intensity).all():
            dv_new._intensity = np.full(dv_new.shape, np.nan)
        else:
            slic = self.intensity[key]
            if not hasattr(slic, "__len__"):
                slic = np.array(
                    [
                        slic,
                    ]
                )
            dv_new._intensity = slic

        return dv_new

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        if not hasattr(value, "__len__"):
            value = np.array(
                [
                    value,
                ]
                * self.size
            )
        if len(value) != self.size:
            raise ValueError("Length of intensity array must match number of vectors")
        self._intensity = np.array(value)

    def calculate_structure_factor(self):
        raise NotImplementedError(
            "Structure factor calculation not implemented for DiffractingVector"
        )
