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
from orix.vector.miller import _transform_space
from orix.quaternion import Rotation


class DiffractingVector(ReciprocalLatticeVector):
    r"""Reciprocal lattice vectors :math:`(hkl)` for use in electron
    diffraction analysis and simulation.

    All lengths are assumed to be given in Å or inverse Å.

    This extends the :class:`ReciprocalLatticeVector` class.  `DiffractingVector`
    focus on the subset of reciprocal lattice vectors that are relevant for
    electron diffraction based on the intersection of the Ewald sphere with the
    reciprocal lattice.

    This class is only used internally to store the DiffractionVectors generated from the
    :class:`~diffsims.simulations.DiffractionSimulation` class. It is not (currently)
    intended to be used directly by the user.

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
    intensity : numpy.ndarray, list, or tuple, optional
        Intensity of the diffraction vector(s). Default is ``None``.
    rotation : orix.quaternion.Rotation, optional
        Rotation matrix previously applied to the reciprocal lattice vector(s) and the
        lattice of the phase. Default is ``None`` which corresponds to the
        identity matrix.


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
        new_data = self.data[key]
        dv_new = self.__class__(self.phase, xyz=new_data)

        if np.isnan(self.structure_factor).all():
            dv_new._structure_factor = np.full(dv_new.shape, np.nan, dtype="complex128")

        else:
            dv_new._structure_factor = self.structure_factor[key]
        if np.isnan(self.theta).all():
            dv_new._theta = np.full(dv_new.shape, np.nan)
        else:
            dv_new._theta = self.theta[key]
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
    def basis_rotation(self):
        """
        Returns the lattice basis rotation.
        """
        return Rotation.from_matrix(self.phase.structure.lattice.baserot)

    def rotate_with_basis(self, rotation):
        """Rotate both vectors and the basis with a given `Rotation`.
        This differs from simply multiplying with a `Rotation`,
        as that would NOT update the basis.

        Parameters
        ----------
        rot : orix.quaternion.Rotation
            A rotation to apply to vectors and the basis.

        Returns
        -------
        DiffractingVector
            A new DiffractingVector with the rotated vectors and basis. This maintains
            the hkl indices of the vectors, but the underlying vector xyz coordinates
            are rotated by the given rotation.

        Notes
        -----
        Rotating the lattice basis may lead to undefined behavior in orix as it violates
        the assumption that the basis is aligned with the crystal axes. Particularly,
        applying symmetry operations to the phase may lead to unexpected results.
        """

        if rotation.size != 1:
            raise ValueError("Rotation must be a single rotation")
        # rotate basis
        new_phase = self.phase.deepcopy()
        br = new_phase.structure.lattice.baserot
        # In case the base rotation is set already
        new_br = br @ rotation.to_matrix().squeeze()
        new_phase.structure.lattice.setLatPar(baserot=new_br)
        # rotate vectors
        vecs = ~rotation * self.to_miller()
        return ReciprocalLatticeVector(new_phase, xyz=vecs.data)

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
            "Structure factor calculation not implemented for DiffractionVector. "
            "Use ReciprocalLatticeVector instead."
        )

    def to_flat_polar(self):
        """Return the vectors in polar coordinates as projected onto the x,y plane"""
        flat_self = self.flatten()
        r = np.linalg.norm(flat_self.data[:, :2], axis=1)
        theta = np.arctan2(
            flat_self.data[:, 1],
            flat_self.data[:, 0],
        )
        return r, theta
