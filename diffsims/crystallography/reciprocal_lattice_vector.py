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

from collections import defaultdict
from copy import deepcopy

from diffpy.structure.symmetryutilities import expandPosition
from diffpy.structure import Structure
import numba as nb
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

from diffsims.structure_factor.atomic_scattering_parameters import (
    _get_string_from_element_id,
)
from diffsims.structure_factor.structure_factor import (
    get_refraction_corrected_wavelength,
)
from diffsims.utils.sim_utils import _get_kinematical_structure_factor


class ReciprocalLatticeVector(Vector3d):
    r"""Reciprocal lattice vectors :math:`(hkl)` for use in electron
    diffraction analysis and simulation.

    All lengths are assumed to be given in Å or inverse Å.

    This class extends :class:`orix.vector.Vector3d` to reciprocal
    lattice vectors :math:`(hkl)` specifically for diffraction
    experiments and simulations. It is thus different from
    :class:`orix.vector.Miller`, which is a general class for Miller
    indices both in reciprocal *and* direct space. It supports relevant
    methods also supported in `Miller`, like obtaining a set of vectors
    from a minimal interplanar spacing.

    Create a set of reciprocal lattice vectors from :math:`(hkl)` or
    :math:`(hkil)`.

    The vectors are stored internally as cartesian coordinates in
    :attr:`data`.

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
    >>> from diffsims.crystallography import ReciprocalLatticeVector
    >>> phase = Phase(
    ...     "al",
    ...     space_group=225,
    ...     structure=Structure(
    ...         lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
    ...         atoms=[Atom("Al", [0, 0, 1])],
    ...     ),
    ... )
    >>> rlv = ReciprocalLatticeVector(phase, hkl=[[1, 1, 1], [2, 0, 0]])
    >>> rlv
    ReciprocalLatticeVector (2,), al (m-3m)
    [[1. 1. 1.]
     [2. 0. 0.]]

    """

    def __init__(self, phase, xyz=None, hkl=None, hkil=None):
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

    @property
    def hkl(self):
        """Miller indices.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.hkl
        array([[1., 1., 1.],
               [2., 0., 0.]])

        """

        return _transform_space(self.data, "c", "r", self.phase.structure.lattice)

    @property
    def hkil(self):
        """Miller-Bravais indices.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.hkil
        array([[ 1.,  1., -2.,  1.],
               [ 2.,  0., -2.,  0.]])

        """

        return _hkl2hkil(self.hkl)

    @property
    def h(self):
        """First reciprocal lattice vector index.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.h
        array([1., 2.])

        """

        return self.hkl[..., 0]

    @property
    def k(self):
        """Second reciprocal lattice vector index.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.k
        array([1., 0.])

        """

        return self.hkl[..., 1]

    @property
    def i(self):
        r"""Third reciprocal lattice vector index in 4-index
        Miller-Bravais indices, equal to :math:`-(h + k)`.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.i
        array([-2., -2.])

        """

        return self.hkil[..., 2]

    @property
    def l(self):
        """Third reciprocal lattice vector index, or fourth index in
        4-index Miller Bravais indices.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.l
        array([1., 0.])

        """

        return self.hkl[..., 2]

    @property
    def multiplicity(self):
        """Number of symmetrically equivalent directions per vector.

        Returns
        -------
        mult : numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.multiplicity
        array([8, 6])

        """

        mult = self.symmetrise(return_multiplicity=True)[1]
        return mult.reshape(self.shape)

    @property
    def has_hexagonal_lattice(self):
        """Whether the crystal lattice is hexagonal/trigonal.

        Returns
        -------
        bool

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.has_hexagonal_lattice
        False

        """

        return self.phase.is_hexagonal

    @property
    def coordinate_format(self):
        """Vector coordinate format, either ``"hkl"`` or ``"hkil"``.

        Returns
        -------
        str

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.coordinate_format
        'hkl'
        >>> rlv.coordinate_format = "hkil"
        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[ 1.  1. -2.  1.]
         [ 2.  0. -2.  0.]]

        """

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
        """Miller or Miller-Bravais indices.

        Returns
        -------
        coordinates : numpy.ndarray
            Miller indices if :attr:`coordiante_format` is ``"hkl"`` or
            Miller-Bravais indices if it is ``"hkil"``.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.coordinates
        array([[1., 1., 1.],
               [2., 0., 0.]])
        >>> rlv.coordinate_format = "hkil"
        >>> rlv.coordinates
        array([[ 1.,  1., -2.,  1.],
               [ 2.,  0., -2.,  0.]])

        """

        return self.__getattribute__(self.coordinate_format)

    @property
    def gspacing(self):
        r"""Reciprocal lattice vector spacing :math:`g`.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]

        Lattice parameters are given in :math:`Å`

        >>> rlv.phase.structure.lattice
        Lattice(a=4.04, b=4.04, c=4.04, alpha=90, beta=90, gamma=90)

        so :math:`g` is given in :math:`Å^-1`

        >>> rlv.gspacing
        array([0.42872545, 0.4950495 ])

        """

        return self.phase.structure.lattice.rnorm(self.hkl)

    @property
    def dspacing(self):
        r"""Direct lattice interplanar spacing :math:`d = 1 / g`.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]

        Lattice parameters are given in :math:`Å`

        >>> rlv.phase.structure.lattice
        Lattice(a=4.04, b=4.04, c=4.04, alpha=90, beta=90, gamma=90)

        so :math:`d` is given in :math:`Å`

        >>> rlv.dspacing
        array([2.33249509, 2.02      ])

        """

        return 1 / self.gspacing

    @property
    def scattering_parameter(self):
        r"""Scattering parameter :math:`0.5 \cdot g`.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]

        Lattice parameters are given in :math:`Å`

        >>> rlv.phase.structure.lattice
        Lattice(a=4.04, b=4.04, c=4.04, alpha=90, beta=90, gamma=90)

        so the scattering parameters are given in :math:`Å^-1`

        >>> rlv.scattering_parameter
        array([0.21436272, 0.24752475])

        """

        return 0.5 * self.gspacing

    @property
    def structure_factor(self):
        r"""Kinematical structure factors :math:`F`.

        Returns
        -------
        structure_factor : numpy.ndarray
            Complex array. Filled with ``None`` if
            :meth:`calculate_structure_factor` hasn't been called yet.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]

        Kinematical structure factors are by default not calculated

        >>> rlv.structure_factor
        array([nan+0.j, nan+0.j])

        A unit cell with all asymmetric atom positions is required to
        calculate structure factors

        >>> rlv.phase.structure
        [Al   0.000000 0.000000 1.000000 1.0000]
        >>> rlv.sanitise_phase()
        >>> rlv.phase.structure
        [Al   0.000000 0.000000 0.000000 1.0000,
         Al   0.000000 0.500000 0.500000 1.0000,
         Al   0.500000 0.000000 0.500000 1.0000,
         Al   0.500000 0.500000 0.000000 1.0000]

        >>> rlv.calculate_structure_factor()
        >>> rlv.structure_factor  # doctest: +SKIP
        array([8.46881663-1.55569638e-15j, 7.04777513-8.63103525e-16j])

        """

        return self._structure_factor

    @property
    def theta(self):
        """Twice the Bragg angle.

        Returns
        -------
        theta : numpy.ndarray
            Filled with ``None`` if :meth:`calculate_theta` hasn't been
            called yet.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]

        Bragg angles are by default not calculated

        >>> rlv.theta
        array([nan, nan])

        >>> rlv.calculate_theta(20e3)
        >>> rlv.theta  # doctest: +SKIP
        array([0.0184036 , 0.02125105])

        """

        return self._theta

    @property
    def allowed(self):
        """Return whether vectors diffract according to diffraction
        selection rules assuming kinematic scattering theory.

        Integer vectors are assumed.

        Returns
        -------
        allowed : numpy.ndarray
            Boolean array.

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> from orix.crystal_map import Phase
        >>> from diffsims.crystallography import ReciprocalLatticeVector
        >>> phase = Phase(
        ...     "al",
        ...     space_group=225,
        ...     structure=Structure(
        ...         lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
        ...         atoms=[Atom("Al", [0, 0, 1])],
        ...     ),
        ... )
        >>> rlv = ReciprocalLatticeVector(
        ...     phase, hkl=[[1, 0, 0], [2, 0, 0]]
        ... )
        >>> rlv.allowed
        array([False,  True])

        """

        self._raise_if_no_space_group()

        # Translational symmetry
        centering = self.phase.space_group.short_name[0]

        hkl = self.hkl.round().astype(int).reshape(-1, 3)

        if centering == "A":  # Centred on A faces only
            return np.isclose(np.mod(hkl[:, 1] + hkl[:, 2], 2), 0)
        elif centering == "B":  # Centred on B faces only
            return np.isclose(np.mod(hkl[:, 0] + hkl[:, 2], 2), 0)
        elif centering == "C":  # Centred on C faces only
            return np.isclose(np.mod(hkl[:, 0] + hkl[:, 1], 2), 0)
        elif centering == "F":  # Face-centred, hkl all odd/even
            selection = np.sum(np.mod(hkl, 2), axis=-1)
            return np.array([i not in [1, 2] for i in selection], dtype=bool)
        elif centering == "I":  # Body-centred, h + k + l = 2n (even)
            return np.isclose(np.mod(np.sum(hkl, axis=-1), 2), 0)
        elif centering in ["R", "H"]:  # Rhombohedral obverse
            # Consider Rhombohedral reverse?
            return np.isclose(np.mod(-hkl[:, 0] + hkl[:, 1] + hkl[:, 2], 3), 0)
        elif centering == "P":  # Primitive
            if self.has_hexagonal_lattice:
                # TODO: See rules in e.g.
                #  https://mcl1.ncifcrf.gov/dauter_pubs/284.pdf, Table 4
                #  http://xrayweb.chem.ou.edu/notes/symmetry.html, Systematic Absences
                raise NotImplementedError
            else:  # Any hkl
                return np.ones(self.shape, dtype=bool)
        else:
            raise ValueError(f"Unknown unit cell centering {centering}")

    # ------------------------- Custom methods ----------------------- #

    def calculate_structure_factor(self, scattering_params="xtables"):
        r"""Populate :attr:`structure_factor` with the complex
        kinematical structure factor :math:`F_{hkl}` for each vector.

        Parameters
        ----------
        scattering_params : str
            Which atomic scattering factors to use, either ``"xtables"``
            (default) or ``"lobato"``.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]

        A unit cell with all asymmetric atom positions is required to
        calculate structure factors

        >>> rlv.phase.structure
        [Al   0.000000 0.000000 1.000000 1.0000]
        >>> rlv.sanitise_phase()
        >>> rlv.phase.structure
        [Al   0.000000 0.000000 0.000000 1.0000,
         Al   0.000000 0.500000 0.500000 1.0000,
         Al   0.500000 0.000000 0.500000 1.0000,
         Al   0.500000 0.500000 0.000000 1.0000]

        >>> rlv.calculate_structure_factor()
        >>> rlv.structure_factor  # doctest: +SKIP
        array([8.46881663-1.55569638e-15j, 7.04777513-8.63103525e-16j])

        Default atomic scattering factors are from the International
        Tables of Crystallography Vol. C Table 4.3.2.3. Alternative
        scattering factors are available from Lobato and Van Dyck
        Acta Cryst. (2014). A70, 636-649
        https://doi.org/10.1107/S205327331401643X

        >>> rlv.calculate_structure_factor("lobato")
        >>> rlv.structure_factor  # doctest: +SKIP
        array([8.44934816-1.55212008e-15j, 7.0387957 -8.62003862e-16j])

        """

        # Compute one structure factor per set {hkl}
        hkl_sets = self.get_hkl_sets()

        # For each set, get the indices of the first vector in the
        # present vectors, accounting for potential multiple dimensions
        # and avoding computing the unique vectors again
        first_idx = []
        for arr in list(hkl_sets.values()):
            i = []
            for arr_i in arr:
                i.append(arr_i[0])
            first_idx.append(i)
        first_idx_arr = np.array(first_idx).T

        # Get 2D array of unique vectors, one for each set
        hkl_unique = self.hkl[tuple(first_idx_arr)]

        structure_factor = _get_kinematical_structure_factor(
            structure=self.phase.structure,
            g_indices=hkl_unique,
            g_hkls_array=self.phase.structure.lattice.rnorm(hkl_unique),
            scattering_params=scattering_params,
        )

        # Set structure factors of all symmetrically equivalent vectors
        for i, idx in enumerate(hkl_sets.values()):
            self._structure_factor[idx] = structure_factor[i]

    def calculate_theta(self, voltage):
        r"""Populate :attr:`theta` with the Bragg angle :math:`theta_B`
        in radians.

        Assumes :attr:`phase.structure` lattice parameters and
        Debye-Waller factors are expressed in Ångströms.

        Parameters
        ----------
        voltage : float
            Beam energy in V.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]

        >>> rlv.calculate_theta(20e3)
        >>> rlv.theta
        array([0.0184036 , 0.02125105])
        >>> rlv.calculate_theta(200e3)
        >>> rlv.theta
        array([0.00537583, 0.00620749])

        """

        wavelength = 10 * get_refraction_corrected_wavelength(self.phase, voltage)
        self._theta = np.arcsin(0.5 * wavelength * self.gspacing)

    def deepcopy(self):
        """Get a deepcopy of the vectors.

        Returns
        -------
        ReciprocalLatticeVector

        """

        return deepcopy(self)

    def get_hkl_sets(self):
        r"""Get unique sets of :math:`{hkl}` for the vectors and the
        indices of vectors in each set.

        Returns
        -------
        hkl_sets : defaultdict
            Dictionary with (h, k, l) as keys and a tuple with
            :class:`numpy.ndarray` with integers of the vectors
            (possibly multi-dimensional) in each set. The keys (h, k, l)
            are rounded to six decimals so that applying integer values
            (h, k, l) as dictionary keys work.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> hkl_sets = rlv.get_hkl_sets()
        >>> hkl_sets
        defaultdict(<class 'tuple'>, {(2.0, 0.0, 0.0): (array([1]),), (1.0, 1.0, 1.0): (array([0]),)})
        >>> hkl_sets[2, 0, 0]
        (array([1]),)
        >>> rlv[hkl_sets[2, 0, 0]]
        ReciprocalLatticeVector (1,), al (m-3m)
        [[2. 0. 0.]]

        """

        # Determine the unique vectors {hkl} representing each set
        rlv_unique = self.unique(use_symmetry=True)

        # Generate all symmetrically equivalent vectors in each set
        # {hkl}, used as a look-up-table for the present vectors
        rlv_symmetrised, mult = rlv_unique.symmetrise(return_multiplicity=True)

        # Find the set for each vector. A Numba function is called,
        # requiring two 2D arrays and one 1D array of float64
        hkl = self.hkl.reshape(-1, 3).astype(np.float64)
        test_hkl = rlv_symmetrised.hkl.reshape(-1, 3).astype(np.float64)
        mult = mult.astype(np.int64)
        hkl_set_idx = _get_set_per_hkl(hkl, test_hkl, mult)

        # Generate dictionary of {hkl} and the indices of vectors in
        # each set
        hkl_sets = defaultdict(tuple)
        for i, hkl_i in enumerate(rlv_unique.hkl):
            mask1d = np.where(hkl_set_idx == i)[0]
            mask = np.unravel_index(mask1d, self.shape)
            hkl_sets[tuple(hkl_i.round(6))] = mask

        return hkl_sets

    def print_table(self):
        r"""Table with indices, structure factor values and multiplicity
        of each set of :math:`{hkl}`.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.print_table()
         h k l      d     |F|_hkl   |F|^2   |F|^2_rel   Mult
         1 1 1    2.332     nan      nan       nan       8
         2 0 0    2.020     nan      nan       nan       6
        >>> rlv.sanitise_phase()
        >>> rlv.calculate_structure_factor()
        >>> rlv.print_table()
         h k l      d     |F|_hkl   |F|^2   |F|^2_rel   Mult
         1 1 1    2.332     8.5     71.7      100.0      8
         2 0 0    2.020     7.0     49.7       69.3      6

        """

        # Column alignment
        align = "^"  # right ">", left "<", or centered "^"

        # Column widths
        width = 6
        hkl_width = width + 2
        d_width = width
        f_hkl_width = width + 1
        f2_hkl_width = width + 1
        f2_hkl_rel_width = width + 2
        mult_width = width

        # Header (note the two-space spacing)
        data = (
            "{:{align}{width}}  ".format(" h k l ", width=hkl_width, align=align)
            + "{:{align}{width}}  ".format("d", width=d_width, align=align)
            + "{:{align}{width}}  ".format("|F|_hkl", width=f_hkl_width, align=align)
            + "{:{align}{width}}  ".format("|F|^2", width=f2_hkl_width, align=align)
            + "{:{align}{width}}  ".format(
                "|F|^2_rel", width=f2_hkl_rel_width, align=align
            )
            + "{:{align}{width}}\n".format("Mult", width=mult_width, align=align)
        )

        v = self.unique(use_symmetry=True)
        structure_factor = v.structure_factor
        f_hkl = abs(structure_factor)
        f2_hkl = abs(structure_factor * structure_factor.conjugate())
        order = np.argsort(f2_hkl)
        v = v[order][::-1]
        f_hkl = f_hkl[order][::-1]
        f2_hkl = f2_hkl[order][::-1]

        size = v.size
        hkl = np.round(v.coordinates).astype(int)
        hkl_string = np.array_str(hkl).replace("[", "").replace("]", "").split("\n")
        d = v.dspacing
        f2_hkl_rel = (f2_hkl / f2_hkl[0]) * 100
        mult = v.multiplicity

        for i in range(size):
            hkl_string_i = hkl_string[i].lstrip(" ")
            data += (
                f"{hkl_string_i:{align}{hkl_width}}  "
                + f"{d[i]:{align}{d_width}.3f}  "
                + f"{f_hkl[i]:{align}{f_hkl_width}.1f}  "
                + f"{f2_hkl[i]:{align}{f2_hkl_width}.1f}  "
                + f" {f2_hkl_rel[i]:{align}{f2_hkl_rel_width}.1f} "
                + f" {mult[i]:{align}{mult_width}}"
            )
            if i != size - 1:
                data += "\n"

        print(data)

    def sanitise_phase(self):
        """Sanitise the :attr:`phase` inplace for calculation of
        structure factors.

        The phase is sanitised when it's
        :attr:`~orix.crystal_map.Phase.structure` has an expanded unit
        cell with all symmetrically atom positions filled, and the atoms
        have their :attr:`~diffpy.structure.Atom.element` set to a
        string, e.g. "Al".

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.phase.structure
        [Al   0.000000 0.000000 1.000000 1.0000]
        >>> rlv.sanitise_phase()
        >>> rlv.phase.structure
        [Al   0.000000 0.000000 0.000000 1.0000,
         Al   0.000000 0.500000 0.500000 1.0000,
         Al   0.500000 0.000000 0.500000 1.0000,
         Al   0.500000 0.500000 0.000000 1.0000]

        """

        self._raise_if_no_space_group()

        space_group = self.phase.space_group
        structure = self.phase.structure

        new_structure = _expand_unit_cell(space_group, structure)
        for atom in new_structure:
            if np.issubdtype(type(atom.element), np.integer):
                atom.element = _get_string_from_element_id(atom.element)

        self.phase.structure = new_structure

    def symmetrise(self, return_multiplicity=False, return_index=False):
        """Unique vectors symmetrically equivalent to the vectors.

        Parameters
        ----------
        return_multiplicity : bool, optional
            Whether to return the multiplicity of each vector. Default
            is ``False``.
        return_index : bool, optional
            Whether to return the index into the vectors for the
            returned symmetrically equivalent vectors. Default is
            ``False``.

        Returns
        -------
        ReciprocalLatticeVector
            Flattened symmetrically equivalent vectors.
        multiplicity : numpy.ndarray
            Multiplicity of each vector. Returned if
            ``return_multiplicity=True``.
        idx : numpy.ndarray
            Index into the vectors for the symmetrically equivalent
            vectors. Returned if ``return_index=True``.

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.symmetrise()
        ReciprocalLatticeVector (14,), al (m-3m)
        [[ 1.  1.  1.]
         [-1.  1.  1.]
         [-1. -1.  1.]
         [ 1. -1.  1.]
         [ 1. -1. -1.]
         [ 1.  1. -1.]
         [-1.  1. -1.]
         [-1. -1. -1.]
         [ 2.  0.  0.]
         [ 0.  2.  0.]
         [-2.  0.  0.]
         [ 0. -2.  0.]
         [ 0.  0.  2.]
         [ 0.  0. -2.]]
        >>> _, mult, idx = rlv.symmetrise(
        ...     return_multiplicity=True, return_index=True
        ... )
        >>> mult
        array([8, 6])
        >>> idx
        array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        """

        out = self.to_miller().symmetrise(
            unique=True, return_multiplicity=return_multiplicity, return_index=True
        )

        if return_multiplicity:
            miller, mult, idx = out
        else:
            miller, idx = out

        new_rlv = self.from_miller(miller)
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

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> from orix.crystal_map import Phase
        >>> from diffsims.crystallography import ReciprocalLatticeVector
        >>> phase = Phase(
        ...     "al",
        ...     space_group=225,
        ...     structure=Structure(
        ...         lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
        ...         atoms=[Atom("Al", [0, 0, 1])],
        ...     ),
        ... )
        >>> rlv = ReciprocalLatticeVector.from_highest_hkl(phase, [3, 3, 3])
        >>> rlv
        ReciprocalLatticeVector (342,), al (m-3m)
        [[ 3.  3.  3.]
         [ 3.  3.  2.]
         [ 3.  3.  1.]
         ...
         [-3. -3. -1.]
         [-3. -3. -2.]
         [-3. -3. -3.]]

        Vectors are included regardless of whether they are
        kinematically allowed or not

        >>> rlv.allowed.all()
        False

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

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> from orix.crystal_map import Phase
        >>> from diffsims.crystallography import ReciprocalLatticeVector
        >>> phase = Phase(
        ...     "al",
        ...     space_group=225,
        ...     structure=Structure(
        ...         lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
        ...         atoms=[Atom("Al", [0, 0, 1])],
        ...     ),
        ... )
        >>> rlv = ReciprocalLatticeVector.from_min_dspacing(phase)
        >>> rlv
        ReciprocalLatticeVector (798,), al (m-3m)
        [[ 5.  2.  2.]
         [ 5.  2.  1.]
         [ 5.  2.  0.]
         ...
         [-5. -2.  0.]
         [-5. -2. -1.]
         [-5. -2. -2.]]
        >>> rlv.dspacing.min()  # doctest: +SKIP
        0.7032737300610338

        Vectors are included regardless of whether they are
        kinematically allowed or not

        >>> rlv = ReciprocalLatticeVector.from_min_dspacing(phase, 1)
        >>> rlv.size
        256
        >>> rlv.allowed.all()
        False

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

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> from orix.crystal_map import Phase
        >>> from orix.vector import Miller
        >>> from diffsims.crystallography import ReciprocalLatticeVector
        >>> phase = Phase(
        ...     "al",
        ...     space_group=225,
        ...     structure=Structure(
        ...         lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
        ...         atoms=[Atom("Al", [0, 0, 1])],
        ...     ),
        ... )
        >>> miller = Miller(hkl=[[1, 1, 1], [2, 0, 0]], phase=phase)
        >>> miller
        Miller (2,), point group m-3m, hkl
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv = ReciprocalLatticeVector.from_miller(miller)
        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.to_miller()
        Miller (2,), point group m-3m, hkl
        [[1. 1. 1.]
         [2. 0. 0.]]

        """

        if miller.coordinate_format not in ["hkl", "hkil"]:
            raise ValueError(
                "`Miller` instance must have `coordinate_format` 'hkl' or 'hkil'"
            )
        return cls(miller.phase, **{miller.coordinate_format: miller.coordinates})

    def to_miller(self):
        """Return the vectors as a ``Miller`` instance.

        Returns
        -------
        orix.vector.Miller

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv.to_miller()
        Miller (2,), point group m-3m, hkl
        [[1. 1. 1.]
         [2. 0. 0.]]

        """

        return Miller(phase=self.phase, **{self.coordinate_format: self.coordinates})

    def _compatible_with(self, other, raise_error=False):
        """Whether the vectors and ``other`` are compatible, i.e. has
        the same crystal lattice and symmetry with vectors in the same
        space.

        Parameters
        ----------
        other : ReciprocalLatticeVector
        raise_error : bool, optional
            Whether to raise a ``ValueError`` if the instances are
            incompatible (default is False).

        Returns
        -------
        bool

        Examples
        --------
        See :class:`ReciprocalLatticeVector` for the creation of ``rlv``

        >>> rlv
        ReciprocalLatticeVector (2,), al (m-3m)
        [[1. 1. 1.]
         [2. 0. 0.]]
        >>> rlv2 = rlv.deepcopy()
        >>> rlv._compatible_with(rlv2)
        1
        >>> rlv2.phase.structure.lattice
        Lattice(a=4.04, b=4.04, c=4.04, alpha=90, beta=90, gamma=90)
        >>> rlv2.phase.structure.lattice.a *= 2
        >>> rlv._compatible_with(rlv2)
        0
        >>> rlv._compatible_with(rlv2, raise_error=True)  # doctest: +SKIP
            raise ValueError(
        ValueError: The crystal lattices and symmetries must be the same, and the vector(s) must be in the same space

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
            Other vectors of compatible shape to the vectors.
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
            Other vectors of compatible shape to the vectors.

        Returns
        -------
        orix.vector.Miller
            Direct lattice vector(s) :math:`[uvw]` or :math:`UVTW`,
            depending on whether the vector's :attr:`coordinate_format`
            is ``hkl`` or ``hkil``, respectively.

        """

        miller = self.to_miller().cross(other.to_miller())
        new_format = {"hkl": "uvw", "hkil": "UVTW"}
        miller.coordinate_format = new_format[self.coordinate_format]
        return miller

    def dot(self, other):
        """Dot product of the vectors with other reciprocal lattice
        vectors.

        Parameters
        ----------
        other : ReciprocalLatticeVector
            Other vectors of compatible shape to the vectors.

        Returns
        -------
        numpy.ndarray

        """

        self._compatible_with(other, raise_error=True)
        return super().dot(other)

    def dot_outer(self, other):
        """Outer dot product of the vectors with other reciprocal
        lattice vectors.

        The dot product for every combination of the vectors are
        computed.

        Parameters
        ----------
        other : ReciprocalLatticeVector
            Other vectors of compatible shape to the vectors.

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

    @classmethod
    def zero(cls, shape=(1,)):
        raise NotImplementedError

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

    def get_random_sample(self, *args, **kwargs):
        raise NotImplementedError

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

        If :attr:`ndim` is originally 2, then order may be undefined.
        In this case the first two dimensions will be transposed.

        Parameters
        ----------
        *axes : int, optional
            Transposed axes order. Only navigation axes need to be
            defined. May be undefined if the vectors only contain two
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
        """The unique vectors.

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

        # TODO: Reduce floating point precision in orix!
        def miller_unique(miller, use_symmetry=False):
            v, idx = Vector3d(miller).unique(return_index=True)

            if use_symmetry:
                operations = miller.phase.point_group
                n_v = v.size
                v2 = operations.outer(v).flatten().reshape(*(n_v, operations.size))
                data = v2.data.round(6)  # Reduced precision
                data_sorted = np.zeros_like(data)
                for i in range(n_v):
                    a = data[i]
                    order = np.lexsort(a.T)  # Sort by column 1, 2, then 3
                    data_sorted[i] = a[order]
                _, idx = np.unique(data_sorted, return_index=True, axis=0)
                v = v[idx[::-1]]

            m = miller.__class__(xyz=v.data, phase=miller.phase)
            m.coordinate_format = miller.coordinate_format
            return m, idx

        #        kwargs = dict(use_symmetry=use_symmetry, return_index=True)
        #        miller, idx = self.to_miller().unique(**kwargs)
        miller, idx = miller_unique(self.to_miller(), use_symmetry)
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


# TODO: Upstream to diffpy.structure.Atom.__eq__()
def _atom_eq(atom1, atom2):
    """Determine whether two atoms are equal.

    Parameters
    ----------
    atom1, atom2 : diffpy.structure.Atom

    Returns
    -------
    bool

    """

    return (
        atom1.element == atom2.element
        and np.allclose(atom1.xyz, atom2.xyz, atol=1e-7)
        and atom1.occupancy == atom2.occupancy
        and np.allclose(atom1.U, atom2.U, atol=1e-7)
        and np.allclose(atom1.Uisoequiv, atom2.Uisoequiv, atol=1e-7)
    )


# TODO: Upstream to orix.crystal_map.Phase.expand_structure()
def _expand_unit_cell(space_group, structure):
    """Expand a unit cell with symmetrically equivalent atoms.

    Parameters
    ----------
    space_group : diffpy.structure.spacegroupmod.SpaceGroup
        Space group describing the symmetry operations of the unit cell.
    structure : diffpy.structure.Structure
        Initial structure with atoms.

    Returns
    -------
    new_structure : diffpy.structure.Structure

    """

    new_structure = Structure(lattice=structure.lattice)

    for atom in structure:
        equal = []
        for atom2 in new_structure:
            equal.append(_atom_eq(atom, atom2))
        if not any(equal):
            new_positions = expandPosition(space_group, atom.xyz)[0]
            for new_position in new_positions:
                new_atom = deepcopy(atom)
                new_atom.xyz = new_position
                new_structure.append(new_atom)

    return new_structure


@nb.njit(
    "int64[:](float64[:, :], float64[:, :], int64[:])",
    cache=True,
    fastmath=True,
    nogil=True,
)
def _get_set_per_hkl(hkl, test_hkl, mult):
    r"""Get which set :math:`{hkl}` each vector :math:`(hkl)` is in.

    Parameters
    ----------
    hkl : numpy.ndarray
        Vectors in 2D array of shape (n, 3) and 64-bit floating point
        data type. Each vector is compared to all vectors in
        ``test_hkl``.
    test_hkl : numpy.ndarray
        Test vectors in 2D array of shape (m, 3) and 64-bit floating
        point data type. These vectors correspond to all symmetrically
        equivalent vectors in the sets :math:`{hkl}`.
    mult : numpy.ndarray
        Multiplicity of the sets :math:`{hkl}`, in the order that the
        equivalent vectors populate ``test_hkl``.

    Returns
    -------
    hkl_set_idx : numpy.ndarray
        1D array of shape (n,) and 64-bit integer data type.

    Notes
    -----
    This is a Numba function so care must be taken with the input
    arrays' shape and data type.

    """

    # Get the index of each hkl into test_hkl, done by floating point
    # comparison of each (h, k, l) with each (test h, test k, test l).
    # A match should always be found, since the sets of test_hkl were
    # generated from the symmetrically unique vectors hkl.
    hkl_size = np.shape(hkl)[0]
    test_hkl_size = np.shape(test_hkl)[0]
    idx = np.zeros(hkl_size, dtype=np.int64)
    for i in nb.prange(hkl_size):
        hkl_i = hkl[i]
        for j in range(test_hkl_size):
            # Compare each index manually since Numba does not support
            # floating point comparison via numpy.isclose()
            test_hkl_j = test_hkl[j]
            for k in range(3):
                if not abs(hkl_i[k] - test_hkl_j[k]) < 1e-6:
                    break
            else:
                idx[i] = j

    # [Start, stop] indices of each set of {hkl} in test_hkl
    family_range = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(mult)))

    # Get which set {hkl} in test_hkl each hkl belongs to
    hkl_set_idx = np.zeros(hkl_size, dtype=np.int64)
    for i in nb.prange(family_range.size - 1):
        mask_i = np.logical_and(idx >= family_range[i], idx < family_range[i + 1])
        hkl_set_idx[mask_i] = i

    return hkl_set_idx
