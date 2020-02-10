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

import pytest
import numpy as np
import diffpy

from diffsims.utils.sim_utils import get_electron_wavelength, \
    get_interaction_constant, get_unique_families, get_kinematical_intensities, \
    get_vectorized_list_for_atomic_scattering_factors, get_points_in_sphere, \
    simulate_kinematic_scattering, is_lattice_hexagonal, uvtw_to_uvw


@pytest.mark.parametrize('accelerating_voltage, wavelength', [
    (100, 0.0370143659),
    (200, 0.0250793403),
    (300, 0.0196874888),
    ('inf', 0),
])
def test_get_electron_wavelength(accelerating_voltage, wavelength):
    val = get_electron_wavelength(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, wavelength)


@pytest.mark.parametrize('accelerating_voltage, interaction_constant', [
    (100, 1.0066772603317773e-16),
    (200, 2.0133545206634971e-16),
    (300, 3.0200317809952176e-16),
])
def test_get_interaction_constant(accelerating_voltage, interaction_constant):
    val = get_interaction_constant(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, interaction_constant)


def test_get_unique_families():
    hkls = ((0, 1, 1), (1, 1, 0))
    unique_families = get_unique_families(hkls)
    assert unique_families == {(1, 1, 0): 2}


def test_get_points_in_sphere():
    latt = diffpy.structure.lattice.Lattice(0.5, 0.5, 0.5, 90, 90, 90)
    ind, cord, dist = get_points_in_sphere(latt, 0.6)
    assert len(ind) == len(cord)
    assert len(ind) == len(dist)
    assert len(dist) == 1 + 6


def test_kinematic_simulator_plane_wave():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32)
    # assert isinstance(sim, ElectronDiffraction)


def test_kinematic_simulator_gaussian_probe():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian_probe')
    # assert isinstance(sim, ElectronDiffraction)


def test_kinematic_simulator_xtables_scattering_params():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian_probe',
                                        scattering_params='xtables')
    # assert isinstance(sim, ElectronDiffraction)


@pytest.mark.xfail(raises=NotImplementedError)
def test_kinematic_simulator_invalid_scattering_params():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian_probe',
                                        scattering_params='_empty')
    # assert isinstance(sim, ElectronDiffraction)


@pytest.mark.xfail(raises=ValueError)
def test_kinematic_simulator_invalid_illumination():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(atomic_coordinates, "Si", 300.,
                                        simulation_size=32,
                                        illumination='gaussian')
    # assert isinstance(sim, ElectronDiffraction)


@pytest.mark.parametrize('uvtw, uvw', [
    ((0, 0, 0, 1), (0, 0, 1)),
    ((1, 0, 0, 1), (2, 1, 1)),
    ((2, 2, 0, 0), (1, 1, 0)),
])
def test_uvtw_to_uvw(uvtw, uvw):
    val = uvtw_to_uvw(uvtw)
    np.testing.assert_almost_equal(val, uvw)
