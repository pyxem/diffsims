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
from pytest import approx
import scipy.constants as sc


from diffsims.utils.sim_utils import (
    get_electron_wavelength,
    get_interaction_constant,
    get_unique_families,
    get_kinematical_intensities,
    get_vectorized_list_for_atomic_scattering_factors,
    get_points_in_sphere,
    simulate_kinematic_scattering,
    is_lattice_hexagonal,
    uvtw_to_uvw,
    get_holz_angle,
    scattering_angle_to_lattice_parameter,
    bst_to_beta,
    beta_to_bst,
    tesla_to_am,
    acceleration_voltage_to_velocity,
    acceleration_voltage_to_relativistic_mass,
    et_to_beta,
    acceleration_voltage_to_wavelength,
    diffraction_scattering_angle,
)


@pytest.mark.parametrize(
    "accelerating_voltage, wavelength",
    [(100, 0.0370143659), (200, 0.0250793403), (300, 0.0196874888), ("inf", 0),],
)
def test_get_electron_wavelength(accelerating_voltage, wavelength):
    val = get_electron_wavelength(accelerating_voltage=accelerating_voltage)
    np.testing.assert_almost_equal(val, wavelength)


@pytest.mark.parametrize(
    "accelerating_voltage, interaction_constant",
    [
        (100, 1.0066772603317773e-16),
        (200, 2.0133545206634971e-16),
        (300, 3.0200317809952176e-16),
    ],
)
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
    sim = simulate_kinematic_scattering(
        atomic_coordinates, "Si", 300.0, simulation_size=32
    )
    # assert isinstance(sim, ElectronDiffraction)


def test_kinematic_simulator_gaussian_probe():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(
        atomic_coordinates,
        "Si",
        300.0,
        simulation_size=32,
        illumination="gaussian_probe",
    )
    # assert isinstance(sim, ElectronDiffraction)


def test_kinematic_simulator_xtables_scattering_params():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(
        atomic_coordinates,
        "Si",
        300.0,
        simulation_size=32,
        illumination="gaussian_probe",
        scattering_params="xtables",
    )
    # assert isinstance(sim, ElectronDiffraction)


@pytest.mark.xfail(raises=NotImplementedError)
def test_kinematic_simulator_invalid_scattering_params():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(
        atomic_coordinates,
        "Si",
        300.0,
        simulation_size=32,
        illumination="gaussian_probe",
        scattering_params="_empty",
    )
    # assert isinstance(sim, ElectronDiffraction)


@pytest.mark.xfail(raises=ValueError)
def test_kinematic_simulator_invalid_illumination():
    atomic_coordinates = np.asarray([[0, 0, 0]])  # structure.cart_coords
    sim = simulate_kinematic_scattering(
        atomic_coordinates, "Si", 300.0, simulation_size=32, illumination="gaussian"
    )
    # assert isinstance(sim, ElectronDiffraction)


@pytest.mark.parametrize(
    "uvtw, uvw",
    [((0, 0, 0, 1), (0, 0, 1)), ((1, 0, 0, 1), (2, 1, 1)), ((2, 2, 0, 0), (1, 1, 0)),],
)
def test_uvtw_to_uvw(uvtw, uvw):
    val = uvtw_to_uvw(uvtw)
    np.testing.assert_almost_equal(val, uvw)

class TestHolzCalibration:
    def test_get_holz_angle(self):
        wavelength = 2.51 / 1000
        lattice_parameter = 0.3905 * 2 ** 0.5
        angle = get_holz_angle(wavelength, lattice_parameter)
        assert approx(95.37805 / 1000) == angle

    def test_scattering_angle_to_lattice_parameter(self):
        wavelength = 2.51 / 1000
        angle = 95.37805 / 1000
        lattice_size = scattering_angle_to_lattice_parameter(wavelength, angle)
        assert approx(0.55225047) == lattice_size

class TestBetaToBst:
    def test_zero(self):
        data = np.zeros((100, 100))
        bst = beta_to_bst(data, 200000)
        assert data.shape == bst.shape
        assert (data == 0.0).all()

    def test_ones(self):
        data = np.ones((100, 100)) * 10
        bst = beta_to_bst(data, 200000)
        assert data.shape == bst.shape
        assert (data != 0.0).all()

    def test_beta_to_bst_to_beta(self):
        beta = 2e-6
        output = bst_to_beta(beta_to_bst(beta, 200000), 200000)
        assert beta == output

    def test_known_value(self):
        # From https://dx.doi.org/10.1016/j.ultramic.2016.03.006
        bst = 10e-9 * 1  # 10 nm, 1 Tesla
        av = 200000  # 200 kV
        beta = bst_to_beta(bst, av)
        assert approx(beta, rel=1e-4) == 6.064e-6


class TestBstToBeta:
    def test_zero(self):
        data = np.zeros((100, 100))
        beta = bst_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data == 0.0).all()

    def test_ones(self):
        data = np.ones((100, 100)) * 10
        beta = bst_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data != 0.0).all()

    def test_bst_to_beta_to_bst(self):
        bst = 10e-6
        output = beta_to_bst(bst_to_beta(bst, 200000), 200000)
        assert bst == output


class TestEtToBeta:
    def test_zero(self):
        data = np.zeros((100, 100))
        beta = et_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data == 0.0).all()

    def test_ones(self):
        data = np.ones((100, 100)) * 10
        beta = bst_to_beta(data, 200000)
        assert data.shape == beta.shape
        assert (data != 0.0).all()

class TeslaToAm:
    def test_zero(self):
        data = np.zeros((100, 100))
        am = tesla_to_am(data)
        assert data.shape == am.shape
        assert (data == 0.00).all()

    def test_ones(self):
        data = np.ones((100, 100)) * 10
        am = tesla_to_am(data)
        assert data.shape == am.shape
        assert (data != 0.0).all()

    def test_known_value(self):
        tesla = 1
        am = tesla_to_am(tesla)
        assert approx(tesla, rel=1e-4) == 795775


class TestAccelerationVoltageToVelocity:
    def test_zero(self):
        assert acceleration_voltage_to_velocity(0) == 0.0

    @pytest.mark.parametrize(
        "av,vel", [(100000, 1.6434e8), (200000, 2.0844e8), (300000, 2.3279e8)]
    )  # V, m/s
    def test_values(self, av, vel):
        v = acceleration_voltage_to_velocity(av)
        assert approx(v, rel=0.001) == vel


class TestAccelerationVoltageToRelativisticMass:
    def test_zero(self):
        mr = acceleration_voltage_to_relativistic_mass(0.0)
        assert approx(mr) == sc.electron_mass

    def test_200kv(self):
        mr = acceleration_voltage_to_relativistic_mass(200000)
        assert approx(mr) == 1.268e-30

@pytest.mark.parametrize(
    "av,wl", [(100000, 3.701e-12), (200000, 2.507e-12), (300000, 1.968e-12)]
)  # V, pm
def test_acceleration_voltage_to_wavelength(av, wl):
    wavelength = acceleration_voltage_to_wavelength(av)
    assert approx(wavelength, rel=0.001, abs=0.0) == wl


def test_acceleration_voltage_to_wavelength_array():
    av = np.array([100000, 200000, 300000])  # In Volt
    wavelength = acceleration_voltage_to_wavelength(av)
    wl = np.array([3.701e-12, 2.507e-12, 1.968e-12])  # In pm
    assert len(wl) == 3
    assert approx(wavelength, rel=0.001, abs=0.0) == wl


class TestDiffractionScatteringAngle:
    def test_simple(self):
        # This should give ~9.84e-3 radians
        acceleration_voltage = 300000
        lattice_size = 2e-10  # 2 Ångstrøm (in meters)
        miller_index = (1, 0, 0)
        scattering_angle = diffraction_scattering_angle(
            acceleration_voltage, lattice_size, miller_index
        )
        assert approx(scattering_angle, rel=0.001) == 9.84e-3

    @pytest.mark.parametrize(
        "mi,sa",
        [
            ((1, 0, 0), 9.84e-3),
            ((0, 1, 0), 9.84e-3),
            ((0, 0, 1), 9.84e-3),
            ((2, 0, 0), 19.68e-3),
            ((0, 2, 0), 19.68e-3),
            ((0, 0, 2), 19.68e-3),
        ],
    )
    def test_miller_index(self, mi, sa):
        acceleration_voltage = 300000
        lattice_size = 2e-10  # 2 Ångstrøm (in meters)
        scattering_angle = diffraction_scattering_angle(
            acceleration_voltage, lattice_size, mi
        )
        assert approx(scattering_angle, rel=0.001) == sa

    def test_array_like(self):
        # This should give ~9.84e-3 radians
        acceleration_voltage = 300000
        lattice_size = np.array([2e-10, 2e-10])
        miller_index = (1, 0, 0)
        scattering_angle = diffraction_scattering_angle(
            acceleration_voltage, lattice_size, miller_index
        )
        assert len(scattering_angle) == 2
        sa_known = np.array([9.84e-3, 9.84e-3])
        assert approx(scattering_angle, rel=0.001) == sa_known
