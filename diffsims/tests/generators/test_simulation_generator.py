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

import numpy as np
import pytest

import diffpy.structure
from orix.crystal_map import Phase

from diffsims.generators.simulation_generator import SimulationGenerator
from diffsims.simulations.simulation import ProfileSimulation
from diffsims.utils.shape_factor_models import (
    linear,
    binary,
    sin2c,
    atanc,
    lorentzian,
    _shape_factor_precession,
)


@pytest.fixture(params=[(300)])
def diffraction_calculator(request):
    return SimulationGenerator(request.param)


@pytest.fixture(scope="module")
def diffraction_calculator_precession_full():
    return SimulationGenerator(300, precession_angle=0.5, approximate_precession=False)


@pytest.fixture(scope="module")
def diffraction_calculator_precession_simple():
    return SimulationGenerator(300, precession_angle=0.5, approximate_precession=True)


def local_excite(excitation_error, maximum_excitation_error, t):
    return (np.sin(t) * excitation_error) / maximum_excitation_error


@pytest.fixture(scope="module")
def diffraction_calculator_custom():
    return SimulationGenerator(300, shape_factor_model=local_excite, t=0.2)


@pytest.fixture(params=[(1, 3), (1,), (False,)])
def precessed(request):
    var = request.param
    return var if len(var) - 1 else var[0]


def make_phase(lattice_parameter=None):
    """
    We construct an Fd-3m silicon (with lattice parameter 5.431 as a default)
    """
    if lattice_parameter is not None:
        a = lattice_parameter
    else:
        a = 5.431
    latt = diffpy.structure.lattice.Lattice(a, a, a, 90, 90, 90)
    # TODO - Make this construction with internal diffpy syntax
    atom_list = []
    for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
        x, y, z = coords[0], coords[1], coords[2]
        atom_list.append(
            diffpy.structure.atom.Atom(atype="Si", xyz=[x, y, z], lattice=latt)
        )  # Motif part A
        atom_list.append(
            diffpy.structure.atom.Atom(
                atype="Si", xyz=[x + 0.25, y + 0.25, z + 0.25], lattice=latt
            )
        )  # Motif part B
    struct = diffpy.structure.Structure(atoms=atom_list, lattice=latt)
    p = Phase(structure=struct, space_group=227)
    return p


@pytest.fixture()
def local_structure():
    return make_phase()


def probe(x, out=None, scale=None):
    if hasattr(x, "shape"):
        return (abs(x[..., 0]) < 6) * (abs(x[..., 1]) < 6)
    else:
        v = abs(x[0].reshape(-1, 1, 1)) < 6
        v = v * abs(x[1].reshape(1, -1, 1)) < 6
        return v + 0 * x[2].reshape(1, 1, -1)


@pytest.mark.parametrize("model", [binary, linear, atanc, sin2c, lorentzian])
def test_shape_factor_precession(model):
    excitation = np.array([-0.1, 0.1])
    r = np.array([1, 5])
    _ = _shape_factor_precession(excitation, r, 0.5, model, 0.1)


def test_linear_shape_factor():
    excitation = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    totest = linear(excitation, 1)
    np.testing.assert_allclose(totest, np.array([0, 0, 0.5, 1, 0.5, 0, 0]))
    np.testing.assert_allclose(linear(0.5, 1), 0.5)


@pytest.mark.parametrize(
    "model, expected",
    [("linear", linear), ("lorentzian", lorentzian), (binary, binary)],
)
def test_diffraction_generator_init(model, expected):
    generator = SimulationGenerator(300, shape_factor_model=model)
    assert generator.shape_factor_model == expected


class TestDiffractionCalculator:
    def test_init(self, diffraction_calculator: SimulationGenerator):
        assert diffraction_calculator.scattering_params == "lobato"
        assert diffraction_calculator.precession_angle == 0
        assert diffraction_calculator.shape_factor_model == lorentzian
        assert diffraction_calculator.approximate_precession == True
        assert diffraction_calculator.minimum_intensity == 1e-20

    def test_matching_results(
        self, diffraction_calculator: SimulationGenerator, local_structure
    ):
        diffraction = diffraction_calculator.calculate_ed_data(
            local_structure, reciprocal_radius=5.0
        )
        assert diffraction.simulations[0].coordinates.size == 72

    def test_precession_simple(
        self, diffraction_calculator_precession_simple, local_structure
    ):
        diffraction = diffraction_calculator_precession_simple.calculate_ed_data(
            local_structure,
            reciprocal_radius=5.0,
        )
        assert diffraction.simulations[0].coordinates.size == 252

    def test_precession_full(
        self, diffraction_calculator_precession_full, local_structure
    ):
        diffraction = diffraction_calculator_precession_full.calculate_ed_data(
            local_structure,
            reciprocal_radius=5.0,
        )
        assert diffraction.simulations[0].coordinates.size == 252

    def test_custom_shape_func(self, diffraction_calculator_custom, local_structure):
        diffraction = diffraction_calculator_custom.calculate_ed_data(
            local_structure,
            reciprocal_radius=5.0,
        )
        assert diffraction.simulations[0].coordinates.size == 52

    def test_appropriate_scaling(self, diffraction_calculator: SimulationGenerator):
        """Tests that doubling the unit cell halves the pattern spacing."""
        silicon = make_phase(5)
        big_silicon = make_phase(10)
        diffraction = diffraction_calculator.calculate_ed_data(
            phase=silicon, reciprocal_radius=5.0
        )
        big_diffraction = diffraction_calculator.calculate_ed_data(
            phase=big_silicon, reciprocal_radius=5.0
        )
        indices = [tuple(i) for i in diffraction.simulations[0].coordinates.hkl]
        big_indices = [tuple(i) for i in big_diffraction.simulations[0].coordinates.hkl]
        assert (2, 2, 0) in indices
        assert (2, 2, 0) in big_indices
        coordinates = diffraction.simulations[0].coordinates[indices.index((2, 2, 0))]
        big_coordinates = big_diffraction.simulations[0].coordinates[
            big_indices.index((2, 2, 0))
        ]
        assert np.allclose(coordinates.data, big_coordinates.data * 2)

    def test_appropriate_intensities(self, diffraction_calculator, local_structure):
        """Tests the central beam is strongest."""
        diffraction = diffraction_calculator.calculate_ed_data(
            local_structure, reciprocal_radius=0.5, with_direct_beam=False
        )  # direct beam doesn't work
        indices = [
            tuple(np.round(i).astype(int))
            for i in diffraction.simulations[0].coordinates.hkl
        ]
        central_beam = indices.index((0, 1, 0))

        smaller = np.greater_equal(
            diffraction.simulations[0].intensities[central_beam],
            diffraction.simulations[0].intensities,
        )
        assert np.all(smaller)

    def test_shape_factor_strings(self, diffraction_calculator, local_structure):
        _ = diffraction_calculator.calculate_ed_data(
            local_structure,
        )

    def test_shape_factor_custom(self, diffraction_calculator, local_structure):
        t1 = diffraction_calculator.calculate_ed_data(
            local_structure, max_excitation_error=0.02
        )
        t2 = diffraction_calculator.calculate_ed_data(
            local_structure, max_excitation_error=0.4
        )

        # softly makes sure the two sims are different
        assert np.sum(t1.simulations[0].intensities) != np.sum(
            t2.simulations[0].intensities
        )

    def test_calculate_profile_class(self, local_structure, diffraction_calculator):
        # tests the non-hexagonal (cubic) case
        profile = diffraction_calculator.calculate_profile_data(
            local_structure, reciprocal_radius=1.0
        )
        assert isinstance(profile, ProfileSimulation)

        latt = diffpy.structure.lattice.Lattice(3, 3, 5, 90, 90, 120)
        atom = diffpy.structure.atom.Atom(atype="Ni", xyz=[0, 0, 0], lattice=latt)
        hexagonal_structure = diffpy.structure.Structure(atoms=[atom], lattice=latt)
        phase = Phase(structure=hexagonal_structure, space_group=194)
        hexagonal_profile = diffraction_calculator.calculate_profile_data(
            phase=phase, reciprocal_radius=1.0
        )
        assert isinstance(hexagonal_profile, ProfileSimulation)


@pytest.mark.parametrize("scattering_param", ["lobato", "xtables"])
def test_param_check(scattering_param):
    generator = SimulationGenerator(300, scattering_params=scattering_param)


@pytest.mark.xfail(raises=NotImplementedError)
def test_invalid_scattering_params():
    scattering_param = "_empty"
    generator = SimulationGenerator(300, scattering_params=scattering_param)


@pytest.mark.xfail(faises=NotImplementedError)
def test_invalid_shape_model():
    generator = SimulationGenerator(300, shape_factor_model="dracula")
