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

import numpy as np
import pytest
from pathlib import Path

import diffpy.structure
from orix.crystal_map import Phase
from orix.quaternion import Rotation

from diffsims.generators.simulation_generator import SimulationGenerator
from diffsims.utils.shape_factor_models import (
    linear,
    binary,
    sin2c,
    atanc,
    lorentzian,
    _shape_factor_precession,
)
from diffsims.simulations import Simulation1D
from diffsims.utils.sim_utils import is_lattice_hexagonal

TEST_DATA_DIR = Path(__file__).parent
FILE1 = TEST_DATA_DIR / "old_simulation.npy"


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

    def test_repr(self, diffraction_calculator):
        assert repr(diffraction_calculator) == (
            "SimulationGenerator(accelerating_voltage=300, "
            "scattering_params=lobato, "
            "approximate_precession=True)"
        )

    def test_matching_results(
        self, diffraction_calculator: SimulationGenerator, local_structure
    ):
        diffraction = diffraction_calculator.calculate_diffraction2d(
            local_structure, reciprocal_radius=5.0
        )
        assert diffraction.coordinates.size == 70

    def test_precession_simple(
        self, diffraction_calculator_precession_simple, local_structure
    ):
        diffraction = diffraction_calculator_precession_simple.calculate_diffraction2d(
            local_structure,
            reciprocal_radius=5.0,
        )
        assert diffraction.coordinates.size == 250

    def test_precession_full(
        self, diffraction_calculator_precession_full, local_structure
    ):
        diffraction = diffraction_calculator_precession_full.calculate_diffraction2d(
            local_structure,
            reciprocal_radius=5.0,
        )
        assert diffraction.coordinates.size == 250

    def test_custom_shape_func(self, diffraction_calculator_custom, local_structure):
        diffraction = diffraction_calculator_custom.calculate_diffraction2d(
            local_structure,
            reciprocal_radius=5.0,
        )
        assert diffraction.coordinates.size == 52

    def test_appropriate_scaling(self, diffraction_calculator: SimulationGenerator):
        """Tests that doubling the unit cell halves the pattern spacing."""
        silicon = make_phase(5)
        big_silicon = make_phase(10)
        diffraction = diffraction_calculator.calculate_diffraction2d(
            phase=silicon, reciprocal_radius=5.0
        )
        big_diffraction = diffraction_calculator.calculate_diffraction2d(
            phase=big_silicon, reciprocal_radius=5.0
        )
        indices = [tuple(i) for i in diffraction.coordinates.hkl]
        big_indices = [tuple(i) for i in big_diffraction.coordinates.hkl]
        assert (2, 2, 0) in indices
        assert (2, 2, 0) in big_indices
        coordinates = diffraction.coordinates[indices.index((2, 2, 0))]
        big_coordinates = big_diffraction.coordinates[big_indices.index((2, 2, 0))]
        assert np.allclose(coordinates.data, big_coordinates.data * 2)

    def test_appropriate_intensities(self, diffraction_calculator, local_structure):
        """Tests the central beam is strongest."""
        diffraction = diffraction_calculator.calculate_diffraction2d(
            local_structure, reciprocal_radius=0.5, with_direct_beam=True
        )
        indices = [tuple(np.round(i).astype(int)) for i in diffraction.coordinates.hkl]
        central_beam = indices.index((0, 0, 0))

        smaller = np.greater_equal(
            diffraction.coordinates.intensity[central_beam],
            diffraction.coordinates.intensity,
        )
        assert np.all(smaller)

    def test_direct_beam(self, diffraction_calculator, local_structure):
        diffraction = diffraction_calculator.calculate_diffraction2d(
            local_structure, reciprocal_radius=0.5, with_direct_beam=False
        )
        indices = [tuple(np.round(i).astype(int)) for i in diffraction.coordinates.hkl]
        with pytest.raises(ValueError):
            indices.index((0, 0, 0))

    def test_shape_factor_strings(self, diffraction_calculator, local_structure):
        _ = diffraction_calculator.calculate_diffraction2d(
            local_structure,
        )

    def test_shape_factor_custom(self, diffraction_calculator, local_structure):
        t1 = diffraction_calculator.calculate_diffraction2d(
            local_structure, max_excitation_error=0.02
        )
        t2 = diffraction_calculator.calculate_diffraction2d(
            local_structure, max_excitation_error=0.4
        )
        # softly makes sure the two sims are different
        assert np.sum(t1.coordinates.intensity) != np.sum(t2.coordinates.intensity)

    @pytest.mark.parametrize("is_hex", [True, False])
    def test_simulate_1d(self, is_hex):
        generator = SimulationGenerator(300)
        phase = make_phase()
        if is_hex:
            phase.structure.lattice.a = phase.structure.lattice.b
            phase.structure.lattice.alpha = 90
            phase.structure.lattice.beta = 90
            phase.structure.lattice.gamma = 120
            assert is_lattice_hexagonal(phase.structure.lattice)
        else:
            assert not is_lattice_hexagonal(phase.structure.lattice)
        sim = generator.calculate_diffraction1d(phase, 0.5)
        assert isinstance(sim, Simulation1D)

        assert len(sim.intensities) == len(sim.reciprocal_spacing)
        assert len(sim.intensities) == len(sim.hkl)
        for h in sim.hkl:
            h = h.replace("-", "")
            if is_hex:
                assert len(h) == 4
            else:
                assert len(h) == 3


def test_multiphase_multirotation_simulation():
    generator = SimulationGenerator(300)
    silicon = make_phase(5)
    big_silicon = make_phase(10)
    rot = Rotation.from_euler([[0, 0, 0], [0.1, 0.1, 0.1]])
    rot2 = Rotation.from_euler([[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    _ = generator.calculate_diffraction2d([silicon, big_silicon], rotation=[rot, rot2])


def test_multiphase_multirotation_simulation_error():
    generator = SimulationGenerator(300)
    silicon = make_phase(5)
    big_silicon = make_phase(10)
    rot = Rotation.from_euler([[0, 0, 0], [0.1, 0.1, 0.1]])
    _ = Rotation.from_euler([[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    with pytest.raises(ValueError):
        _ = generator.calculate_diffraction2d([silicon, big_silicon], rotation=[rot])


@pytest.mark.parametrize("scattering_param", ["lobato", "xtables"])
def test_param_check(scattering_param):
    _ = SimulationGenerator(300, scattering_params=scattering_param)


def test_invalid_scattering_params():
    with pytest.raises(NotImplementedError):
        _ = SimulationGenerator(300, scattering_params="_empty")


def test_invalid_shape_model():
    with pytest.raises(NotImplementedError):
        _ = SimulationGenerator(300, shape_factor_model="dracula")


def test_same_simulation_results():
    # This test is to ensure that the new SimulationGenerator produces the same
    # results as the old DiffractionLibraryGenerator. Based on comments from
    # viljarjf in https://github.com/pyxem/diffsims/pull/201
    # Shared parameters
    latt = diffpy.structure.lattice.Lattice(2.464, 2.464, 6.711, 90, 90, 120)
    atoms = [
        diffpy.structure.atom.Atom(atype="C", xyz=[0.0, 0.0, 0.25], lattice=latt),
        diffpy.structure.atom.Atom(atype="C", xyz=[0.0, 0.0, 0.75], lattice=latt),
        diffpy.structure.atom.Atom(atype="C", xyz=[1 / 3, 2 / 3, 0.25], lattice=latt),
        diffpy.structure.atom.Atom(atype="C", xyz=[2 / 3, 1 / 3, 0.75], lattice=latt),
    ]
    structure_matrix = diffpy.structure.Structure(atoms=atoms, lattice=latt)
    calibration = 0.0262
    reciprocal_radius = 1.6768
    with_direct_beam = False
    max_excitation_error = 0.1
    accelerating_voltage = 200
    shape = (128, 128)
    sigma = 1.4
    generator_kwargs = {
        "accelerating_voltage": accelerating_voltage,
        "scattering_params": "lobato",
        "precession_angle": 0,
        "shape_factor_model": "lorentzian",
        "approximate_precession": True,
        "minimum_intensity": 1e-20,
    }
    # The euler angles are different, as orix.Phase enforces x||a, z||c*
    # euler_angles_old = np.array([[0, 90, 120]])
    euler_angles_new = np.array([[0, 90, 90]])

    # Old way. For creating the old data.
    # struct_library = StructureLibrary(["Graphite"], [structure_matrix], [euler_angles_old])
    # diff_gen = DiffractionGenerator(**generator_kwargs)
    # lib_gen = DiffractionLibraryGenerator(diff_gen)
    # diff_lib = lib_gen.get_diffraction_library(struct_library,
    #                                            calibration=calibration,
    #                                            reciprocal_radius=reciprocal_radius,
    #                                            with_direct_beam=with_direct_beam,
    #                                            max_excitation_error=max_excitation_error,
    #                                            half_shape=shape[0] // 2,
    #                                           )
    # old_data = diff_lib["Graphite"]["simulations"][0].get_diffraction_pattern(shape=shape, sigma=sigma)

    # New
    p = Phase("Graphite", point_group="6/mmm", structure=structure_matrix)
    gen = SimulationGenerator(**generator_kwargs)
    rot = Rotation.from_euler(euler_angles_new, degrees=True)
    sim = gen.calculate_diffraction2d(
        phase=p,
        rotation=rot,
        reciprocal_radius=reciprocal_radius,
        max_excitation_error=max_excitation_error,
        with_direct_beam=with_direct_beam,
    )
    new_data = sim.get_diffraction_pattern(
        shape=shape, sigma=sigma, calibration=calibration
    )
    old_data = np.load(FILE1)
    np.testing.assert_allclose(new_data, old_data, atol=1e-8)
