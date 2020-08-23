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

from diffpy.structure.symmetryutilities import expandPosition, SymmetryConstraints
import numpy as np
from scipy.constants import c, e, h, physical_constants

from diffsims.reciprocal_lattice_point.atomic_scattering_factor import (
    get_kinematical_atomic_scattering_factor,
    get_doyleturner_atomic_scattering_factor,
)


rest_mass = physical_constants["atomic unit of mass"][0]


def find_asymmetric_positions(positions, space_group):
    """Return the asymmetric atom positions among a set of positions
    when considering symmetry operations defined by a space group.

    Parameters
    ----------
    positions : list
        A list of cartesian atom positions.
    space_group : diffpy.structure.spacegroups.SpaceGroup
        Space group describing the symmetry operations.

    Returns
    -------
    np.ndarray
        Asymmetric atom positions.
    """
    asymmetric_positions = SymmetryConstraints(space_group, positions).corepos
    return [
        np.array([np.allclose(xyz, asym_xyz) for xyz in positions])
        for asym_xyz in asymmetric_positions
    ][0]


def get_kinematical_structure_factor(phase, hkl, scattering_parameter):
    """Return the kinematical (X-ray) structure factor for a given family
    of Miller indices.

    Assumes structure's lattice parameters and Debye-Waller factors are
    expressed in Ångströms.

    Parameters
    ----------
    phase : orix.crystal_map.phase_list.Phase
        A phase container with a crystal structure and a space and point
        group describing the allowed symmetry operations.
    hkl : np.ndarray
        Miller indices.
    scattering_parameter : float
        Scattering parameter for these Miller indices.

    Returns
    -------
    structure_factor : float
        Structure factor F.
    """
    # Initialize real and imaginary parts of the structure factor
    structure_factor = 0 + 0j

    structure = phase.structure
    space_group = phase.space_group

    # Loop over asymmetric unit
    asymmetric_positions = find_asymmetric_positions(structure.xyz, space_group)
    for is_asymmetric, atom in zip(asymmetric_positions, structure):
        if not is_asymmetric:
            continue

        # Get atomic scattering factor for this atom
        f = get_kinematical_atomic_scattering_factor(atom, scattering_parameter)

        # Loop over all atoms in the orbit
        equiv_pos = expandPosition(spacegroup=space_group, xyz=atom.xyz)[0]
        for xyz in equiv_pos:
            arg = 2 * np.pi * np.sum(hkl * xyz)
            structure_factor += f * (np.cos(arg) - (np.sin(arg) * 1j))

    return structure_factor.real


def get_doyleturner_structure_factor(phase, hkl, scattering_parameter, voltage):
    """Return the structure factor for a given family of Miller indices
    using Doyle-Turner atomic scattering parameters [Doyle1968]_.

    Assumes structure's lattice parameters and Debye-Waller factors are
    expressed in Ångströms.

    Parameters
    ----------
    phase : orix.crystal_map.phase_list.Phase
        A phase container with a crystal structure and a space and point
        group describing the allowed symmetry operations.
    hkl : np.ndarray
        Miller indices.
    scattering_parameter : float
        Scattering parameter for these Miller indices.
    voltage : float
        Beam energy in V.

    Returns
    -------
    structure_factor : float
        Structure factor F.
    """
    structure = phase.structure
    space_group = phase.space_group

    # Initialize real and imaginary parts of the structure factor
    structure_factor = 0 + 0j

    # Get unit cell volume for the atomic scattering factor calculation
    unit_cell_volume = structure.lattice.volume

    # Loop over all atoms in the asymmetric unit
    asymmetric_positions = find_asymmetric_positions(structure.xyz, space_group)
    for is_asymmetric, atom in zip(asymmetric_positions, structure):
        if not is_asymmetric:
            continue

        # Get atomic scattering factor for this atom
        f = get_doyleturner_atomic_scattering_factor(
            atom, scattering_parameter, unit_cell_volume
        )

        # Loop over all atoms in the orbit
        equiv_pos = expandPosition(spacegroup=space_group, xyz=atom.xyz)[0]
        for xyz in equiv_pos:
            arg = 2 * np.pi * np.sum(hkl * xyz)
            structure_factor += f * np.exp(-arg * 1j)

    # Derived factors
    # TODO: Comment these factors with stuff from Structure of Materials by De Graef
    #  and McHenry
    gamma_relcor = 1 + (2 * e * 0.5 * voltage / rest_mass / (c ** 2))
    v_mod = abs(structure_factor) * gamma_relcor
    v_phase = np.arctan2(structure_factor.imag, structure_factor.real)
    v_g = v_mod * np.exp(v_phase * 1j)
    pre = 2 * (rest_mass * e / h ** 2) * 1e-18

    return (pre * v_g).real
