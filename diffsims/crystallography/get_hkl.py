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

from itertools import product

import numpy as np
from orix.vector import Vector3d

from diffsims.utils._deprecated import deprecated


@deprecated(
    since="0.6",
    alternative="diffsims.crystallography.ReciprocalLatticeVector.from_min_dspacing",
    removal="0.7",
)
def get_highest_hkl(lattice, min_dspacing=0.5):
    """Return the highest Miller indices hkl of the plane with a direct
    space interplanar spacing (d-spacing) greater than but closest to
    *min_dspacing*.

    Parameters
    ----------
    lattice : diffpy.structure.Lattice
        Crystal lattice.
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


@deprecated(
    since="0.6",
    alternative="diffsims.crystallography.ReciprocalLatticeVector.from_highest_hkl",
    removal="0.7",
)
def get_hkl(highest_hkl):
    """Return a list of planes from a set of highest Miller indices.

    Parameters
    ----------
    highest_hkl : orix.vector.Vector3d, np.ndarray, list, or tuple of int
        Highest Miller indices to consider.

    Returns
    -------
    hkl : np.ndarray
        An array of Miller indices.
    """
    index_ranges = [np.arange(-i, i + 1) for i in highest_hkl]
    return np.asarray(list(product(*index_ranges)))


@deprecated(
    since="0.6",
    alternative="diffsims.crystallography.ReciprocalLatticeVector.symmetrise",
    removal="0.7",
)
def get_equivalent_hkl(hkl, operations, unique=False, return_multiplicity=False):
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
        Whether to return the multiplicity of the input indices. Default
        is False.

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
