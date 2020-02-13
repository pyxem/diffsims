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

import math

import numpy as np
from scipy.constants import h, m_e, e, c, pi
import collections
import diffpy.structure

from transforms3d.axangles import axangle2mat
from transforms3d.euler import mat2euler
from transforms3d.euler import euler2mat

from .atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
from .lobato_scattering_params import ATOMIC_SCATTERING_PARAMS_LOBATO

# from diffsims.sims.diffraction_simulation import DiffractionSimulation
from diffsims.utils.vector_utils import get_angle_cartesian


def get_electron_wavelength(accelerating_voltage):
    """Calculates the (relativistic) electron wavelength in Angstroms for a
    given accelerating voltage in kV.

    Parameters
    ----------
    accelerating_voltage : float or 'inf'
        The accelerating voltage in kV. Values `numpy.inf` and 'inf' are
        also accepted.

    Returns
    -------
    wavelength : float
        The relativistic electron wavelength in Angstroms.

    """
    if accelerating_voltage in (np.inf, 'inf'):
        return 0
    E = accelerating_voltage * 1e3
    wavelength = h / math.sqrt(2 * m_e * e * E *
                               (1 + (e / (2 * m_e * c * c)) * E)) * 1e10
    return wavelength


def get_interaction_constant(accelerating_voltage):
    """Calculates the interaction constant, sigma, for a given
    acelerating voltage.

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage in V.

    Returns
    -------
    sigma : float
        The relativistic electron wavelength in m.

    """
    E = accelerating_voltage
    wavelength = get_electron_wavelength(accelerating_voltage)
    sigma = (2 * pi * (m_e + e * E))

    return sigma


def get_unique_families(hkls):
    """Returns unique families of Miller indices, which must be permutations of
    each other.

    Parameters
    ----------
    hkls : list
        List of Miller indices ([h, k, l])

    Returns
    -------
    pretty_unique : dictionary
        A dict with unique hkl and multiplicity {hkl: multiplicity}.
    """

    def is_perm(hkl1, hkl2):
        h1 = np.abs(hkl1)
        h2 = np.abs(hkl2)
        return all([i == j for i, j in zip(sorted(h1), sorted(h2))])

    unique = collections.defaultdict(list)
    for hkl1 in hkls:
        found = False
        for hkl2 in unique.keys():
            if is_perm(hkl1, hkl2):
                found = True
                unique[hkl2].append(hkl1)
                break
        if not found:
            unique[hkl1].append(hkl1)

    pretty_unique = {}
    for k, v in unique.items():
        pretty_unique[sorted(v)[-1]] = len(v)

    return pretty_unique


def get_scattering_params_dict(scattering_params):
    """Get scattering parameter dictionary from name.

    Parameters
    ----------
    scattering_params : string
        Name of scattering factors. One of 'lobato', 'xtables'.

    Returns
    -------
    scattering_params_dict : dict
        Dictionary of scattering parameters mapping from element name.
    """
    if scattering_params == 'lobato':
        scattering_params_dict = ATOMIC_SCATTERING_PARAMS_LOBATO
    elif scattering_params == 'xtables':
        scattering_params_dict = ATOMIC_SCATTERING_PARAMS
    else:
        raise NotImplementedError("The scattering parameters `{}` are not implemented. "
                                  "See documentation for available "
                                  "implementations.".format(scattering_params))
    return scattering_params_dict


def get_vectorized_list_for_atomic_scattering_factors(structure,
                                                      debye_waller_factors,
                                                      scattering_params):
    """ Create a flattened array of coeffs, fcoords and occus for vectorized
    computation of atomic scattering factors.

    Note: The dimensions of the returned objects are not necessarily the same
    size as the number of atoms in the structure as each partially occupied
    specie occupies its own position in the flattened array.


    Parameters
    ----------
    structure : diffpy.structure
        The atomic structure for which scattering factors are required.
    debye_waller_factors : list
        List of Debye-Waller factors for atoms in structure.

    Returns
    -------
    coeffs : np.array()
        Coefficients of atomic scattering factor parameterization for each atom.
    fcoords : np.array()
        Fractional coordinates of each atom in structure.
    occus : np.array()
        Occupancy of each atomic site.
    dwfactors : np.array()
        Debye-Waller factors for each atom in the structure.
    """

    scattering_params_dict = get_scattering_params_dict(scattering_params)

    n_structures = len(structure)
    coeffs = np.empty((n_structures, 5, 2))
    fcoords = np.empty((n_structures, 3))
    occus = np.empty(n_structures)
    dwfactors = np.empty(n_structures)

    for i, site in enumerate(structure):
        coeffs[i] = scattering_params_dict[site.element]
        dwfactors[i] = debye_waller_factors.get(site.element, 0)
        fcoords[i] = site.xyz
        occus[i] = site.occupancy

    return coeffs, fcoords, occus, dwfactors


def get_atomic_scattering_factors(g_hkl_sq, coeffs, scattering_params):
    """Calculate atomic scattering factors for n atoms.

    Parameters
    ----------
    g_hkl_sq : ndarray
        One-dimensional array of g-vector lengths squared.
    coeffs : ndarray
        Three-dimensional array [n, 5, 2] of coefficients corresponding to the n atoms.
    scattering_params : string
        Type of scattering factor calculation to use. One of 'lobato', 'xtables'.

    Returns
    -------
    scattering_factors : ndarray
        The calculated atomic scattering parameters.
    """
    g_sq_coeff_1 = np.outer(g_hkl_sq, coeffs[:, :, 1]).reshape(g_hkl_sq.shape + coeffs[:, :, 1].shape)
    if scattering_params == 'lobato':
        f = (2 + g_sq_coeff_1) * (1 / np.square(1 + g_sq_coeff_1))
    elif scattering_params == 'xtables':
        f = np.exp(-0.25 * g_sq_coeff_1)
    return np.sum(coeffs[:, :, 0] * f, axis=-1)


def get_kinematical_intensities(structure,
                                g_indices,
                                g_hkls,
                                excitation_error,
                                maximum_excitation_error,
                                debye_waller_factors,
                                scattering_params='lobato'):
    """Calculates peak intensities.

    The peak intensity is a combination of the structure factor for a given
    peak and the position the Ewald sphere intersects the relrod. In this
    implementation, the intensity scales linearly with proximity.

    Parameters
    ----------
    structure : Structure
        The structure for which to derive the structure factors.
    indices : array-like
        The fractional coordinates of the peaks for which to calculate the
        structure factor.
    proximities : array-like
        The distances between the Ewald sphere and the peak centers.

    Returns
    -------
    peak_intensities : array-like
        The intensities of the peaks.

    """
    coeffs, fcoords, occus, dwfactors = get_vectorized_list_for_atomic_scattering_factors(
        structure=structure, debye_waller_factors=debye_waller_factors,
        scattering_params=scattering_params)

    # Store array of g_hkls^2 values since used multiple times.
    g_hkls_sq = g_hkls ** 2

    # Create array containing atomic scattering factors.
    fs = get_atomic_scattering_factors(g_hkls_sq, coeffs, scattering_params)

    # Change the coordinate system of fcoords to align with that of g_indices
    fcoords = np.dot(fcoords, np.linalg.inv(np.dot(structure.lattice.stdbase,
                                                   structure.lattice.recbase)))

    # Calculate structure factors for all excited g-vectors.
    f_hkls = np.sum(fs * occus * np.exp(
        2j * np.pi * np.dot(g_indices, fcoords.T) -
        0.25 * np.outer(g_hkls_sq, dwfactors)),
        axis=-1)

    # Define an intensity scaling that is linear with distance from Ewald sphere
    # along the beam direction.
    shape_factor = 1 - (excitation_error / maximum_excitation_error)

    # Calculate the peak intensities from the structure factor and excitation
    # error.
    peak_intensities = (f_hkls * f_hkls.conjugate()).real * shape_factor
    return peak_intensities


def simulate_kinematic_scattering(atomic_coordinates,
                                  element,
                                  accelerating_voltage,
                                  simulation_size=256,
                                  max_k=1.5,
                                  illumination='plane_wave',
                                  sigma=20,
                                  scattering_params='lobato'):
    """Simulate electron scattering from arrangement of atoms comprising one
    elemental species.

    Parameters
    ----------
    atomic_coordinates : array
        Array specifying atomic coordinates in structure.
    element : string
        Element symbol, e.g. "C".
    accelerating_voltage : float
        Accelerating voltage in keV.
    simulation_size : int
        Simulation size, n, specifies the n x n array size for
        the simulation calculation.
    max_k : float
        Maximum scattering vector magnitude in reciprocal angstroms.
    illumination : string
        Either 'plane_wave' or 'gaussian_probe' illumination
    sigma : float
        Gaussian probe standard deviation, used when illumination == 'gaussian_probe'
    scattering_params : string
        Type of scattering factor calculation to use. One of 'lobato', 'xtables'.

    Returns
    -------
    simulation : ElectronDiffraction
        ElectronDiffraction simulation.
    """
    # Get atomic scattering parameters for specified element.
    coeffs = np.array(get_scattering_params_dict(scattering_params)[element])

    # Calculate electron wavelength for given keV.
    wavelength = get_electron_wavelength(accelerating_voltage)

    # Define a 2D array of k-vectors at which to evaluate scattering.
    l = np.linspace(-max_k, max_k, simulation_size)
    kx, ky = np.meshgrid(l, l)

    # Convert 2D k-vectors into 3D k-vectors accounting for Ewald sphere.
    k = np.array((kx, ky, (wavelength / 2) * (kx ** 2 + ky ** 2)))

    # Calculate scattering vector squared for each k-vector.
    gs_sq = np.linalg.norm(k, axis=0) ** 2

    # Get the scattering factors for this element.
    fs = get_atomic_scattering_factors(gs_sq, coeffs[np.newaxis, :], scattering_params)

    # Evaluate scattering from all atoms
    scattering = np.zeros_like(gs_sq)
    if illumination == 'plane_wave':
        for r in atomic_coordinates:
            scattering = scattering + (fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    elif illumination == 'gaussian_probe':
        for r in atomic_coordinates:
            probe = (1 / (np.sqrt(2 * np.pi) * sigma)) * \
                np.exp((-np.abs(((r[0] ** 2) - (r[1] ** 2)))) / (4 * sigma ** 2))
            scattering = scattering + (probe * fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    else:
        raise ValueError("User specified illumination '{}' not defined.".format(illumination))

    # Calculate intensity
    intensity = (scattering * scattering.conjugate()).real

    return intensity


def get_points_in_sphere(reciprocal_lattice, reciprocal_radius):
    """Finds all reciprocal lattice points inside a given reciprocal sphere.
    Utilised within the DiffractionGenerator.

    Parameters
    ----------
    reciprocal_lattice : diffpy.Structure.Lattice
        The reciprocal crystal lattice for the structure of interest.
    reciprocal_radius  : float
        The radius of the sphere in reciprocal space (units of reciprocal
        Angstroms) within which reciprocal lattice points are returned.

    Returns
    -------
    spot_indicies : numpy.array
        Miller indices of reciprocal lattice points in sphere.
    spot_coords : numpy.array
        Cartesian coordinates of reciprocal lattice points in sphere.
    spot_distances : numpy.array
        Distance of reciprocal lattice points in sphere from the origin.
    """
    a, b, c = reciprocal_lattice.a, reciprocal_lattice.b, reciprocal_lattice.c
    h_max = np.floor(reciprocal_radius / a)
    k_max = np.floor(reciprocal_radius / b)
    l_max = np.floor(reciprocal_radius / c)
    from itertools import product
    h_list = np.arange(-h_max, h_max + 1)  # arange has a non-inclusive endpoint
    k_list = np.arange(-k_max, k_max + 1)
    l_list = np.arange(-l_max, l_max + 1)
    potential_points = np.asarray(list(product(h_list, k_list, l_list)))
    in_sphere = np.abs(reciprocal_lattice.dist(potential_points, [0, 0, 0])) < reciprocal_radius
    spot_indicies = potential_points[in_sphere]
    spot_coords = reciprocal_lattice.cartesian(spot_indicies)
    spot_distances = reciprocal_lattice.dist(spot_indicies, [0, 0, 0])

    return spot_indicies, spot_coords, spot_distances


def is_lattice_hexagonal(latt):
    """Determines if a diffpy lattice is hexagonal or trigonal.
    Parameters
    ----------
    latt : diffpy.Structure.lattice
        The diffpy lattice object to be determined as hexagonal or not.
    Returns
    -------
    is_true : bool
        True if hexagonal or trigonal.
    """
    truth_list = []
    truth_list.append(latt.a == latt.b)
    truth_list.append(latt.alpha == 90)
    truth_list.append(latt.beta == 90)
    truth_list.append(latt.gamma == 120)
    return len(truth_list) == np.sum(truth_list)


def uvtw_to_uvw(uvtw):
    """Convert 4-index direction to a 3-index direction.

    Parameters
    ----------
    uvtw : array-like with 4 floats

    Returns
    -------
    uvw : tuple of 4 floats
    """
    u, v, t, w = uvtw
    u, v, w = 2 * u + v, 2 * v + u, w
    common_factor = math.gcd(math.gcd(u, v), w)
    return tuple((int(x / common_factor)) for x in (u, v, w))
