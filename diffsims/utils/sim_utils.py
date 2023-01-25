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

import collections
import math

import diffpy.structure
import numpy as np
from scipy.constants import h, m_e, e, c, pi, mu_0

from diffsims.utils.atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
from diffsims.utils.lobato_scattering_params import ATOMIC_SCATTERING_PARAMS_LOBATO


def get_electron_wavelength(accelerating_voltage):
    """Calculates the (relativistic) electron wavelength in Angstroms
    for a given accelerating voltage in kV.

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
    if accelerating_voltage in (np.inf, "inf"):
        return 0
    E = accelerating_voltage * 1e3
    wavelength = (
        h / math.sqrt(2 * m_e * e * E * (1 + (e / (2 * m_e * c * c)) * E)) * 1e10
    )
    return wavelength


def get_interaction_constant(accelerating_voltage):
    """Calculates the interaction constant, sigma, for a given
    accelerating voltage.

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage in V.

    Returns
    -------
    sigma : float
        The relativistic electron wavelength in m.

    """
    sigma = 2 * pi * (m_e + e * accelerating_voltage)
    return sigma


def get_unique_families(hkls):
    """Returns unique families of Miller indices, which must be
    permutations of each other.

    Parameters
    ----------
    hkls : list
        List of Miller indices ([h, k, l])

    Returns
    -------
    pretty_unique : dict
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
            unique[tuple(hkl1)].append(hkl1)

    pretty_unique = {}
    for k, v in unique.items():
        pretty_unique[tuple(sorted(v)[-1])] = len(v)

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
    if scattering_params == "lobato":
        scattering_params_dict = ATOMIC_SCATTERING_PARAMS_LOBATO
    elif scattering_params == "xtables":
        scattering_params_dict = ATOMIC_SCATTERING_PARAMS
    else:
        raise NotImplementedError(
            "The scattering parameters `{}` are not implemented. "
            "See documentation for available "
            "implementations.".format(scattering_params)
        )
    return scattering_params_dict


def get_vectorized_list_for_atomic_scattering_factors(
    structure, debye_waller_factors, scattering_params
):
    """Create a flattened array of coeffs, fcoords and occus for
    vectorized computation of atomic scattering factors.

    Note: The dimensions of the returned objects are not necessarily
    the same size as the number of atoms in the structure as each
    partially occupied specie occupies its own position in the flattened
    array.

    Parameters
    ----------
    structure : diffpy.structure.Structure
        The atomic structure for which scattering factors are required.
    debye_waller_factors : dist
        Debye-Waller factors for atoms in the structure.
    scattering_params: string
        The type of scattering params to use. "lobato", "xtables", and
        None are supported.

    Returns
    -------
    coeffs : numpy.ndarray
        Coefficients of atomic scattering factor parameterization for
        each atom.
    fcoords : numpy.ndarray
        Fractional coordinates of each atom in structure.
    occus : numpy.ndarray
        Occupancy of each atomic site.
    dwfactors : numpy.ndarray
        Debye-Waller factors for each atom in the structure.
    """

    if scattering_params is not None:
        scattering_params_dict = get_scattering_params_dict(scattering_params)
    else:
        scattering_params_dict = {}

    n_structures = len(structure)
    coeffs = np.empty((n_structures, 5, 2))
    fcoords = np.empty((n_structures, 3))
    occus = np.empty(n_structures)
    dwfactors = np.empty(n_structures)
    default = np.zeros((5, 2))

    for i, site in enumerate(structure):
        coeffs[i] = scattering_params_dict.get(site.element, default)
        dwfactors[i] = debye_waller_factors.get(site.element, 0)
        fcoords[i] = site.xyz
        occus[i] = site.occupancy

    return coeffs, fcoords, occus, dwfactors


def get_atomic_scattering_factors(g_hkl_sq, coeffs, scattering_params):
    """Calculate atomic scattering factors for n atoms.

    Parameters
    ----------
    g_hkl_sq : numpy.ndarray
        One-dimensional array of g-vector lengths squared.
    coeffs : numpy.ndarray
        Three-dimensional array [n, 5, 2] of coefficients corresponding
        to the n atoms.
    scattering_params : str
        Type of scattering factor calculation to use. One of 'lobato',
        'xtables'.

    Returns
    -------
    scattering_factors : numpy.ndarray
        The calculated atomic scattering parameters.
    """
    g_sq_coeff_1 = np.outer(g_hkl_sq, coeffs[:, :, 1]).reshape(
        g_hkl_sq.shape + coeffs[:, :, 1].shape
    )
    if scattering_params == "lobato":
        f = (2 + g_sq_coeff_1) * (1 / np.square(1 + g_sq_coeff_1))
    elif scattering_params == "xtables":
        f = np.exp(-0.25 * g_sq_coeff_1)
    return np.sum(coeffs[:, :, 0] * f, axis=-1)


def _get_kinematical_structure_factor(
    structure,
    g_indices,
    g_hkls_array,
    debye_waller_factors=None,
    scattering_params="lobato",
):
    """See docstring of :func:`get_kinematical_intensities`."""
    if debye_waller_factors is None:
        debye_waller_factors = {}

    (
        coeffs,
        xyz,
        occupancy,
        dwfactors,
    ) = get_vectorized_list_for_atomic_scattering_factors(
        structure=structure,
        debye_waller_factors=debye_waller_factors,
        scattering_params=scattering_params,
    )

    gspacing_squared = g_hkls_array ** 2

    if scattering_params is not None:
        atomic_scattering_factor = get_atomic_scattering_factors(
            gspacing_squared, coeffs, scattering_params
        )
    else:
        # Set all atomic scattering factors to 1
        atomic_scattering_factor = np.ones((gspacing_squared.shape[0], coeffs.shape[0]))

    # Express the atom positions in the same reference frame as the
    # Miller indices
    mat = np.linalg.inv(np.dot(structure.lattice.stdbase, structure.lattice.recbase))
    xyz = np.dot(xyz, mat)

    # Calculate the complex structure factor
    structure_factor = np.sum(
        atomic_scattering_factor
        * occupancy
        * np.exp(
            2j * np.pi * np.dot(g_indices, xyz.T)
            - 0.25 * np.outer(gspacing_squared, dwfactors)
        ),
        axis=-1,
    )

    return structure_factor


def get_kinematical_intensities(
    structure,
    g_indices,
    g_hkls_array,
    debye_waller_factors=None,
    scattering_params="lobato",
    prefactor=1,
):
    """Calculates peak intensities.

    The peak intensity is a combination of the structure factor for a
    given peak and the position the Ewald sphere intersects the relrod.
    In this implementation, the intensity scales linearly with
    proximity.

    Parameters
    ----------
    structure : diffpy.structure.Structure
        The structure for which to derive the structure factors.
    g_indices : numpy.ndarray
        Indicies of spots to be considered.
    g_hkls_array : numpy.ndarray
        Coordinates of spots to be considered.
    debye_waller_factors : dict
        Maps element names to their temperature-dependent Debye-Waller
        factors.
    scattering_params : str
        "lobato", "xtables" or None
    prefactor : float or numpy.ndarray
        Multiplciation factor for structure factor.

    Returns
    -------
    peak_intensities : numpy.ndarray
        The intensities of the peaks.

    """
    f_hkls = _get_kinematical_structure_factor(
        structure=structure,
        g_indices=g_indices,
        g_hkls_array=g_hkls_array,
        debye_waller_factors=debye_waller_factors,
        scattering_params=scattering_params,
    )

    # Calculate the peak intensities from the structure factor and prefactor
    peak_intensities = prefactor * (f_hkls * f_hkls.conjugate()).real
    return peak_intensities


def simulate_kinematic_scattering(
    atomic_coordinates,
    element,
    accelerating_voltage,
    simulation_size=256,
    max_k=1.5,
    illumination="plane_wave",
    sigma=20,
    scattering_params="lobato",
):
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
    if illumination == "plane_wave":
        for r in atomic_coordinates:
            scattering = scattering + (fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    elif illumination == "gaussian_probe":
        for r in atomic_coordinates:
            probe = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
                (-np.abs(((r[0] ** 2) - (r[1] ** 2)))) / (4 * sigma ** 2)
            )
            scattering = scattering + (probe * fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    else:
        raise ValueError(
            "User specified illumination '{}' not defined.".format(illumination)
        )

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
    spot_indices : numpy.array
        Miller indices of reciprocal lattice points in sphere.
    cartesian_coordinates : numpy.array
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
    in_sphere = (
        np.abs(reciprocal_lattice.dist(potential_points, [0, 0, 0])) < reciprocal_radius
    )
    spot_indices = potential_points[in_sphere]
    cartesian_coordinates = reciprocal_lattice.cartesian(spot_indices)
    spot_distances = reciprocal_lattice.dist(spot_indices, [0, 0, 0])

    return spot_indices, cartesian_coordinates, spot_distances


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


def get_intensities_params(reciprocal_lattice, reciprocal_radius):

    """Calculates the variables needed for get_kinematical_intensities

    Parameters
    ----------
    reciprocal_lattice : diffpy.Structure.Lattice
        The reciprocal crystal lattice for the structure of interest.
    reciprocal_radius  : float
        The radius of the sphere in reciprocal space (units of reciprocal
        Angstroms) within which reciprocal lattice points are returned.

    Returns
    -------
    unique_hkls : array-like
        The unique plane families which lie in the given reciprocal sphere.

    multiplicites : array-like
        The multiplicites of the given unqiue planes in the sphere.

    g_hkls : list
        The g vector length of the given hkl in the sphere.

    """

    spot_indices, _, spot_distances = get_points_in_sphere(
        reciprocal_lattice, reciprocal_radius
    )

    dict_i_to_d = {}
    for i, d in zip(spot_indices, spot_distances):
        dict_i_to_d[tuple(i)] = d

    list_hkls = spot_indices.tolist()

    unique_hkls_dict = get_unique_families(list_hkls)

    multiplicites = np.fromiter(unique_hkls_dict.values(), dtype=float)
    unique_hkls = np.array(list(unique_hkls_dict))

    g_hkls = []
    for unique_hkl in unique_hkls:
        g_hkls.append(dict_i_to_d[tuple(unique_hkl)])

    return unique_hkls, multiplicites, g_hkls


def get_holz_angle(electron_wavelength, lattice_parameter):
    """Converts electron wavelength and lattice paramater to holz angle
    Parameters
    ----------
    electron_wavelength : scalar
        In nanometers
    lattice_parameter : scalar
        In nanometers

    Returns
    -------
    scattering_angle : scalar
        Scattering angle in radians

    Examples
    --------
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> lattice_size = 0.3905 # STO-(001) in nm
    >>> wavelength = 2.51/1000 # Electron wavelength for 200 kV
    >>> angle = sim_utils.get_holz_angle(wavelength, lattice_size)

    """
    k0 = 1.0 / electron_wavelength
    kz = 1.0 / lattice_parameter
    in_root = kz * ((2 * k0) - kz)
    sin_angle = (in_root ** 0.5) / k0
    angle = np.arcsin(sin_angle)
    return angle


def scattering_angle_to_lattice_parameter(electron_wavelength, angle):
    """Convert scattering angle data to lattice parameter sizes.

    Parameters
    ----------
    electron_wavelength : float
        Wavelength of the electrons in the electron beam. In nm.
        For 200 kV electrons: 0.00251 (nm)
    angle : NumPy array
        Scattering angle, in radians.

    Returns
    -------
    lattice_parameter : NumPy array
        Lattice parameter, in nanometers

    Examples
    --------
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> angle_list = [0.1, 0.1, 0.1, 0.1] # in radians
    >>> wavelength = 2.51/1000 # Electron wavelength for 200 kV
    >>> lattice_size = sim_utils.scattering_angle_to_lattice_parameter(
    ...     wavelength, angle_list)

    """

    k0 = 1.0 / electron_wavelength
    kz = k0 - (k0 * ((1 - (np.sin(angle) ** 2)) ** 0.5))
    return 1 / kz


def bst_to_beta(bst, acceleration_voltage):
    """Calculate beam deflection (beta) values from Bs * t.

    Parameters
    ----------
    bst : NumPy array
        Saturation induction Bs times thickness t of the sample. In Tesla*meter
    acceleration_voltage : float
        In Volts

    Returns
    -------
    beta : NumPy array
        Beam deflection in radians

    Examples
    --------
    >>> import numpy as np
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> data = np.random.random((100, 100))  # In Tesla*meter
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> beta = sim_utils.bst_to_beta(data, acceleration_voltage)

    """
    wavelength = acceleration_voltage_to_wavelength(acceleration_voltage)
    beta = e * wavelength * bst / h
    return beta


def beta_to_bst(beam_deflection, acceleration_voltage):
    """Calculate Bs * t values from beam deflection (beta).

    Parameters
    ----------
    beam_deflection : NumPy array
        In radians
    acceleration_voltage : float
        In Volts

    Returns
    -------
    bst : NumPy array
        In Tesla * meter

    Examples
    --------
    >>> import numpy as np
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> data = np.random.random((100, 100))  # In radians
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> bst = sim_utils.beta_to_bst(data, 200000)

    """
    wavelength = acceleration_voltage_to_wavelength(acceleration_voltage)
    beta = beam_deflection

    mag_field = beta * h / (wavelength * e)
    return mag_field


def tesla_to_am(data):
    """Convert data from Tesla to A/m

    Parameters
    ----------
    data : NumPy array
        Data in Tesla

    Returns
    -------
    output_data : NumPy array
        In A/m

    Examples
    --------
    >>> import numpy as np
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> data_T = np.random.random((100, 100))  # In tesla
    >>> data_am = sim_utils.tesla_to_am(data_T)

    """
    return data / mu_0


def acceleration_voltage_to_velocity(acceleration_voltage):
    """Get relativistic velocity of electron from acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float
        In Volt

    Returns
    -------
    v : float
        In m/s

    Example
    -------
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> v = sim_utils.acceleration_voltage_to_velocity(200000) # 200 kV
    >>> round(v)
    208450035

    """

    part1 = (1 + (acceleration_voltage * e) / (m_e * c ** 2)) ** 2
    v = c * (1 - (1 / part1)) ** 0.5
    return v


def acceleration_voltage_to_relativistic_mass(acceleration_voltage):
    """Get relativistic mass of electron as function of acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float
        In Volt

    Returns
    -------
    mr : float
        Relativistic electron mass

    Example
    -------
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> mr = sim_utils.acceleration_voltage_to_relativistic_mass(200000) # 200 kV

    """
    v = acceleration_voltage_to_velocity(acceleration_voltage)
    part1 = 1 - (v ** 2) / (c ** 2)
    mr = m_e / (part1) ** 0.5
    return mr


def et_to_beta(et, acceleration_voltage):
    """Calculate beam deflection (beta) values from E * t.

    Parameters
    ----------
    et : NumPy array
        Electric field times thickness t of the sample.
    acceleration_voltage : float
        In Volts

    Returns
    -------
    beta: NumPy array
        Beam deflection in radians

    Examples
    --------
    >>> import numpy as np
    >>> import diffsims.utils.sim_utils as sim_utils
    >>> data = np.random.random((100, 100))
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> beta = sim_utils.et_to_beta(data, acceleration_voltage)

    """
    wavelength = acceleration_voltage_to_wavelength(acceleration_voltage)
    m = acceleration_voltage_to_relativistic_mass(acceleration_voltage)

    beta = e * (wavelength ** 2) * m * et / (h ** 2)
    return beta


def acceleration_voltage_to_wavelength(acceleration_voltage):
    """Get electron wavelength from the acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float or array-like
        In Volt

    Returns
    -------
    wavelength : float or array-like
        In meters

    """
    energy = acceleration_voltage * e
    wavelength = h / (2 * m_e * energy * (1 + (energy / (2 * m_e * c ** 2)))) ** 0.5
    return wavelength


def diffraction_scattering_angle(acceleration_voltage, lattice_size, miller_index):
    """Get electron scattering angle from a crystal lattice.

    Returns the total scattering angle, as measured from the middle of the
    direct beam (0, 0, 0) to the given Miller index.

    Miller index: h, k, l = miller_index
    Interplanar distance: d = a / (h**2 + k**2 + l**2)**0.5
    Bragg's law: theta = arcsin(electron_wavelength / (2 * d))
    Total scattering angle (phi):  phi = 2 * theta

    Parameters
    ----------
    acceleration_voltage : float
        In Volt
    lattice_size : float or array-like
        In meter
    miller_index : tuple
        (h, k, l)

    Returns
    -------
    angle : float
        Scattering angle in radians.

    """
    wavelength = acceleration_voltage_to_wavelength(acceleration_voltage)
    h, k, l = miller_index
    a = lattice_size
    d = a / (h ** 2 + k ** 2 + l ** 2) ** 0.5
    scattering_angle = 2 * np.arcsin(wavelength / (2 * d))
    return scattering_angle
