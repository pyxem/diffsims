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

from diffpy.structure.spacegroups import GetSpaceGroup
from diffpy.structure import Atom, Lattice, Structure
import numpy as np
from orix.crystal_map import Phase
import pytest

from diffsims.structure_factor import (
    find_asymmetric_positions,
    get_kinematical_structure_factor,
    get_doyleturner_structure_factor,
    get_refraction_corrected_wavelength,
)

# Debye-Waller factor in Å^-2: Biosequiv = 8 * np.pi ** 2 * Uisoequiv
nickel = Phase(
    space_group=225,
    structure=Structure(
        lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
        atoms=[Atom(xyz=[0, 0, 0], atype="Ni", Uisoequiv=0.006332)],
    ),
)
ferrite = Phase(
    space_group=229,
    structure=Structure(
        lattice=Lattice(2.8665, 2.8665, 2.8665, 90, 90, 90),
        atoms=[
            Atom(xyz=[0, 0, 0], atype="Fe", Uisoequiv=0.006332),
            Atom(xyz=[0.5, 0.5, 0.5], atype="Fe", Uisoequiv=0.006332),
        ],
    ),
)
sic4h = Phase(
    space_group=186,
    structure=Structure(
        lattice=Lattice(3.073, 3.073, 10.053, 90, 90, 120),
        atoms=[
            Atom(atype="Si", xyz=[0, 0, 0], Uisoequiv=0.006332),
            Atom(atype="Si", xyz=[0.33, 0.667, 0.25], Uisoequiv=0.006332),
            Atom(atype="C", xyz=[0, 0, 0.188], Uisoequiv=0.006332),
            Atom(atype="C", xyz=[0.333, 0.667, 0.438], Uisoequiv=0.006332),
        ],
    ),
)


@pytest.mark.parametrize(
    "positions, space_group, desired_asymmetric",
    [
        ([[0, 0, 0], [0, 0.5, 0.5]], 229, [True, False]),
        ([[0.0349, 0.3106, 0.2607], [0.0349, 0.3106, 0.2607]], 229, [True, True]),
    ],
)
def test_find_asymmetric_positions(positions, space_group, desired_asymmetric):
    assert np.allclose(
        find_asymmetric_positions(
            positions=positions, space_group=GetSpaceGroup(space_group)
        ),
        desired_asymmetric,
    )


@pytest.mark.parametrize(
    "phase, hkl, scattering_parameter, desired_factor",
    [
        (nickel, [1, 1, 1], 0.245778, 79.665427),
        (ferrite, [1, 1, 0], 0.2466795, 35.783295),
        (ferrite, [1, 1, 1], 0.2466795, 0),
        (sic4h, [0, 0, 4], 0.198945, 19.027004),
    ],
)
def test_get_kinematical_structure_factor(
    phase, hkl, scattering_parameter, desired_factor
):
    assert np.allclose(
        get_kinematical_structure_factor(
            phase=phase, hkl=hkl, scattering_parameter=scattering_parameter
        ),
        desired_factor,
    )


@pytest.mark.parametrize(
    "phase, hkl, scattering_parameter, voltage, desired_factor",
    [
        (nickel, [1, 1, 1], 0.245778, 15e3, 8.606363),
        (ferrite, [1, 1, 0], 0.2466795, 20e3, 8.096651),
        (ferrite, [1, 1, 1], 0.2466795, 20e3, 0),
        (sic4h, [0, 0, 4], 0.198945, 200e3, 2.744304),
    ],
)
def test_get_doyleturner_structure_factor(
    phase, hkl, scattering_parameter, voltage, desired_factor
):
    """Tested against EMsoft v5."""
    assert np.allclose(
        get_doyleturner_structure_factor(
            phase=phase,
            hkl=hkl,
            scattering_parameter=scattering_parameter,
            voltage=voltage,
        ),
        desired_factor,
    )


@pytest.mark.parametrize(
    "hkl, scattering_parameter, desired_factor, desired_params",
    [
        ([1, 1, 0], 0.2466795, 8.096628, [1.03, 12.17, 1.22e-16, 12.17 + 2.45e-15j]),
        ([2, 0, 0], 0.348857, 5.581301, [1.03, 8.39, 1.22e-16, 8.39 + 1.02e-15j]),
    ],
)
def test_get_doyleturner_structure_factor_returns(
    ferrite_phase, hkl, scattering_parameter, desired_factor, desired_params
):
    """Tested against EMsoft v5. Only the imaginary part of v_g is off."""
    sf, params = get_doyleturner_structure_factor(
        phase=ferrite_phase,
        hkl=hkl,
        scattering_parameter=scattering_parameter,
        voltage=20e3,
        return_parameters=True,
    )
    assert np.allclose(sf, desired_factor)
    assert np.allclose(list(params.values()), desired_params, atol=1e-2)


@pytest.mark.parametrize(
    "phase, voltage, desired_wavelength",
    [
        (nickel, 20e3, 0.008582),
        (nickel, 200e3, 0.002507),
        (ferrite, 10e3, 0.012186),
        (sic4h, 100e3, 0.003701),
    ],
)
def test_get_refraction_corrected_wavelength(phase, voltage, desired_wavelength):
    """Tested against EMsoft v5."""
    assert np.allclose(
        get_refraction_corrected_wavelength(phase=phase, voltage=voltage),
        desired_wavelength,
        atol=1e-6,
    )
