# -*- coding: utf-8 -*-
# Copyright 2017-2021 The diffsims developers
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

"""Electron diffraction pattern simulation."""

import numpy as np
from scipy.integrate import quad
from transforms3d.euler import euler2mat

from diffsims.sims.diffraction_simulation import DiffractionSimulation
from diffsims.sims.diffraction_simulation import ProfileSimulation
from diffsims.utils.sim_utils import (
    get_electron_wavelength,
    get_kinematical_intensities,
    get_points_in_sphere,
    is_lattice_hexagonal,
    get_intensities_params,
)
from diffsims.utils.fourier_transform import from_recip
from diffsims.utils.shape_factor_models import (
    linear,
    atanc,
    lorentzian,
    sinc,
    sin2c,
    lorentzian_precession,
)


_shape_factor_model_mapping = {
    "linear": linear,
    "atanc": atanc,
    "sinc": sinc,
    "sin2c": sin2c,
    "lorentzian": lorentzian,
}


def _z_sphere_precession(theta, r_spot, wavelength, phi):
    """
    Returns the z-coordinate in reciprocal space of the Ewald sphere at
    distance r_spot from the origin

    Parameters
    ----------
    theta : float
        The azimuthal angle in degrees
    r_spot : float
        The projected length of the reciprocal lattice vector onto the plane
        perpendicular to the optical axis in A^-1
    wavelength : float
        The electron wavelength in A
    phi : float
        The precession angle in degrees (angle between beam and optical axis)

    Returns
    -------
    z : float
        The height of the ewald sphere at the point r in A^-1

    Notes
    -----
    * The azimuthal angle is the angle the beam is currently precessed to.
    It is the angle between the projection of the beam and the projection of
    the relevant diffraction spot both onto the x-y plane.
    * In the derivation of this formula we assume that we will always integrate
    over a full precession circle, because we do not explicitly take into
    consideration x-y coordinates of reflections.
    """
    theta = np.deg2rad(theta)
    r = 1 / wavelength
    phi = np.deg2rad(phi)
    return -np.sqrt(
        r ** 2 * (1 - np.sin(phi) ** 2 * np.sin(theta) ** 2)
        - (r_spot - r * np.sin(phi) * np.cos(theta)) ** 2
    ) + r * np.cos(phi)


def _shape_factor_precession(
    z_spot, r_spot, wavelength, phi, shape_function, max_excitation, **kwargs
):
    """
    The rel-rod shape factors for reflections taking into account
    precession

    Parameters
    ----------
    z_spot : np.ndarray (N,)
        An array representing the z-coordinates of the reflections in A^-1
    r_spot : np.ndarray (N,)
        An array representing the distance of spots from the z-axis in A^-1
    wavelength : float
        The electron wavelength in A
    phi : float
        The precession angle in degrees
    shape_function : callable
        A function that describes the influence from the rel-rods. Should be
        in the form func(excitation_error: np.ndarray, max_excitation: float,
        **kwargs)
    max_excitation : float
        Parameter to describe the "extent" of the rel-rods.

    Other parameters
    ----------------
    ** kwargs: passed directly to shape_function

    Notes
    -----
    * We calculate excitation_error as z_spot - z_sphere so that it is
    negative when the spot is outside the ewald sphere and positive when inside
    conform W&C chapter 12, section 12.6
    * We assume that the sample is a thin infinitely wide slab perpendicular
    to the optical axis, so that the shape factor function only depends on the
    distance from each spot to the Ewald sphere parallel to the optical axis.
    """
    shf = np.zeros(z_spot.shape)
    # loop over all spots
    for i, (z_spot_i, r_spot_i) in enumerate(zip(z_spot, r_spot)):

        def integrand(phi):
            z_sph = _z_sphere_precession(phi, r_spot_i, wavelength, phi)
            return shape_function(z_spot_i - z_sph, max_excitation, **kwargs)

        # average factor integrated over the full revolution of the beam
        shf[i] = (1 / (360)) * quad(integrand, 0, 360)[0]
    return shf


def _average_excitation_error_precession(z_spot, r_spot, wavelength, precession_angle):
    """
    Calculate the average excitation error for spots
    """
    ext = np.zeros(z_spot.shape)
    # loop over all spots
    for i, (z_spot_i, r_spot_i) in enumerate(zip(z_spot, r_spot)):

        def integrand(phi):
            z_sph = _z_sphere_precession(phi, r_spot_i, wavelength, precession_angle)
            return z_spot_i - z_sph

        # average factor integrated over the full revolution of the beam
        ext[i] = (1 / (360)) * quad(integrand, 0, 360)[0]
    return ext


class DiffractionGenerator(object):
    """Computes electron diffraction patterns for a crystal structure.

    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`

    3. The intensity of each reflection is then given in the kinematic
       approximation as the modulus square of the structure factor.
       :math:`I_{hkl} = F_{hkl}F_{hkl}^*`

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage of the microscope in kV.
    scattering_params : str
        "lobato" or "xtables"
    minimum_intensity : float
        Minimum intensity for a peak to be considered visible in the pattern
    precession_angle : float
        Angle about which the beam is precessed. Default is no precession.
    approximate_precession : boolean
        When using precession, whether to precisely calculate average
        excitation errors and intensities or use an approximation. See notes.
    shape_factor_model : function or string
        A function that takes excitation_error and
        `max_excitation_error` (and potentially kwargs) and returns
        an intensity scaling factor. If None defaults to
        `shape_factor_models.linear`. A number of pre-programmed functions
        are available via strings.
    kwargs
        Keyword arguments passed to `shape_factor_model`.

    Notes
    -----
    * A full calculation is much slower and is not recommended for calculating
      a diffraction library for precession diffraction patterns.
    * When using precession and approximate_precession=True, the shape factor
    model defaults to Lorentzian; shape_factor_model is ignored. Only with
    approximate_precession=False the custom shape_factor_model is used.
    """

    def __init__(
        self,
        accelerating_voltage,
        scattering_params="lobato",
        precession_angle=0,
        shape_factor_model="lorentzian",
        approximate_precession=True,
        minimum_intensity=1e-20,
        **kwargs,
    ):
        self.wavelength = get_electron_wavelength(accelerating_voltage)
        self.precession_angle = precession_angle
        self.approximate_precession = approximate_precession
        if isinstance(shape_factor_model, str):
            if shape_factor_model in _shape_factor_model_mapping.keys():
                self.shape_factor_model = _shape_factor_model_mapping[
                    shape_factor_model
                ]
            else:
                raise NotImplementedError(
                    f"{shape_factor_model} is not a recognized shape factor "
                    f"model, choose from: {_shape_factor_model_mapping.keys()} "
                    f"or provide your own function."
                )
        else:
            self.shape_factor_model = shape_factor_model
        self.minimum_intensity = minimum_intensity
        self.shape_factor_kwargs = kwargs
        if scattering_params in ["lobato", "xtables"]:
            self.scattering_params = scattering_params
        else:
            raise NotImplementedError(
                "The scattering parameters `{}` is not implemented. "
                "See documentation for available "
                "implementations.".format(scattering_params)
            )

    def calculate_ed_data(
        self,
        structure,
        reciprocal_radius,
        rotation=(0, 0, 0),
        with_direct_beam=True,
        max_excitation_error=1e-2,
        debye_waller_factors={},
    ):
        """Calculates the Electron Diffraction data for a structure.

        Parameters
        ----------
        structure : diffpy.structure.structure.Structure
            The structure for which to derive the diffraction pattern.
            Note that the structure must be rotated to the appropriate
            orientation and that testing is conducted on unit cells
            (rather than supercells).
        reciprocal_radius : float
            The maximum radius of the sphere of reciprocal space to
            sample, in reciprocal Angstroms.
        rotation : tuple
            Euler angles, in degrees, in the rzxz convention. Default is
            (0, 0, 0) which aligns 'z' with the electron beam.
        with_direct_beam : bool
            If True, the direct beam is included in the simulated
            diffraction pattern. If False, it is not.
        max_excitation_error : float
            The extinction distance for reflections, in reciprocal
            Angstroms. Roughly equal to 1/thickness.
        debye_waller_factors : dict of str:value pairs
            Maps element names to their temperature-dependent Debye-Waller factors.

        Returns
        -------
        diffsims.sims.diffraction_simulation.DiffractionSimulation
            The data associated with this structure and diffraction setup.
        """
        # Specify variables used in calculation
        wavelength = self.wavelength
        latt = structure.lattice

        # Obtain crystallographic reciprocal lattice points within `reciprocal_radius` and
        # g-vector magnitudes for intensity calculations.
        recip_latt = latt.reciprocal()
        spot_indices, cartesian_coordinates, spot_distances = get_points_in_sphere(
            recip_latt, reciprocal_radius
        )

        ai, aj, ak = (
            np.deg2rad(rotation[0]),
            np.deg2rad(rotation[1]),
            np.deg2rad(rotation[2]),
        )
        R = euler2mat(ai, aj, ak, axes="rzxz")
        cartesian_coordinates = np.matmul(R, cartesian_coordinates.T).T

        # Identify points intersecting the Ewald sphere within maximum
        # excitation error and store the magnitude of their excitation error.
        r_sphere = 1 / wavelength
        r_spot = np.sqrt(np.sum(np.square(cartesian_coordinates[:, :2]), axis=1))
        z_spot = cartesian_coordinates[:, 2]

        if self.precession_angle > 0 and not self.approximate_precession:
            # We find the average excitation error - this step can be
            # quite expensive
            excitation_error = _average_excitation_error_precession(
                z_spot,
                r_spot,
                wavelength,
                self.precession_angle,
            )
        else:
            z_sphere = -np.sqrt(r_sphere ** 2 - r_spot ** 2) + r_sphere
            excitation_error = z_sphere - z_spot
        # Mask parameters corresponding to excited reflections.
        intersection = np.abs(excitation_error) < max_excitation_error
        intersection_coordinates = cartesian_coordinates[intersection]
        excitation_error = excitation_error[intersection]
        r_spot = r_spot[intersection]
        g_indices = spot_indices[intersection]
        g_hkls = spot_distances[intersection]
        # take into consideration rel-rods
        if self.precession_angle > 0 and not self.approximate_precession:
            shape_factor = _shape_factor_precession(
                intersection_coordinates[:, 2],
                r_spot,
                wavelength,
                self.precession_angle,
                self.shape_factor_model,
                max_excitation_error,
                **self.shape_factor_kwargs,
            )
        elif self.precession_angle > 0 and self.approximate_precession:
            shape_factor = lorentzian_precession(
                excitation_error,
                max_excitation_error,
                r_spot,
                self.precession_angle,
            )
        else:
            shape_factor = self.shape_factor_model(
                excitation_error, max_excitation_error, **self.shape_factor_kwargs
            )

        # Calculate diffracted intensities based on a kinematical model.
        intensities = get_kinematical_intensities(
            structure,
            g_indices,
            g_hkls,
            prefactor=shape_factor,
            scattering_params=self.scattering_params,
            debye_waller_factors=debye_waller_factors,
        )

        # Threshold peaks included in simulation based on minimum intensity.
        peak_mask = intensities > self.minimum_intensity
        intensities = intensities[peak_mask]
        intersection_coordinates = intersection_coordinates[peak_mask]
        g_indices = g_indices[peak_mask]

        return DiffractionSimulation(
            coordinates=intersection_coordinates,
            indices=g_indices,
            intensities=intensities,
            with_direct_beam=with_direct_beam,
        )

    def calculate_profile_data(
        self,
        structure,
        reciprocal_radius=1.0,
        minimum_intensity=1e-3,
        debye_waller_factors={},
    ):
        """Calculates a one dimensional diffraction profile for a
        structure.

        Parameters
        ----------
        structure : diffpy.structure.structure.Structure
            The structure for which to calculate the diffraction profile.
        reciprocal_radius : float
            The maximum radius of the sphere of reciprocal space to
            sample, in reciprocal angstroms.
        minimum_intensity : float
            The minimum intensity required for a diffraction peak to be
            considered real. Deals with numerical precision issues.
        debye_waller_factors : dict of str:value pairs
            Maps element names to their temperature-dependent Debye-Waller factors.

        Returns
        -------
        diffsims.sims.diffraction_simulation.ProfileSimulation
            The diffraction profile corresponding to this structure and
            experimental conditions.
        """
        wavelength = self.wavelength
        latt = structure.lattice

        # Obtain crystallographic reciprocal lattice points within range
        recip_latt = latt.reciprocal()
        spot_indices, _, spot_distances = get_points_in_sphere(
            recip_latt, reciprocal_radius
        )

        ##spot_indicies is a numpy.array of the hkls allowd in the recip radius
        g_indices, multiplicities, g_hkls = get_intensities_params(
            recip_latt, reciprocal_radius
        )

        i_hkl = get_kinematical_intensities(
            structure,
            g_indices,
            np.asarray(g_hkls),
            prefactor=multiplicities,
            scattering_params=self.scattering_params,
            debye_waller_factors=debye_waller_factors,
        )

        if is_lattice_hexagonal(latt):
            # Use Miller-Bravais indices for hexagonal lattices.
            g_indices = (
                g_indices[0],
                g_indices[1],
                -g_indices[0] - g_indices[1],
                g_indices[2],
            )

        hkls_labels = ["".join([str(int(x)) for x in xs]) for xs in g_indices]

        peaks = {}
        for l, i, g in zip(hkls_labels, i_hkl, g_hkls):
            peaks[l] = [i, g]

        # Scale intensities so that the max intensity is 100.

        max_intensity = max([v[0] for v in peaks.values()])
        x = []
        y = []
        hkls = []
        for k in peaks.keys():
            v = peaks[k]
            if v[0] / max_intensity * 100 > minimum_intensity and (k != "000"):
                x.append(v[1])
                y.append(v[0])
                hkls.append(k)

        y = np.asarray(y) / max(y) * 100

        return ProfileSimulation(x, y, hkls)


class AtomicDiffractionGenerator:
    """
    Computes electron diffraction patterns for an atomic lattice.

    Parameters
    ----------
    accelerating_voltage : float, 'inf'
        The accelerating voltage of the microscope in kV
    detector : list of 1D float-type arrays
        List of mesh vectors defining the (flat) detector size and sensor positions
    reciprocal_mesh : bool, optional
        If True then `detector` is assumed to be a reciprocal grid, else
        (default) it is assumed to be a real grid.

    """

    def __init__(self, accelerating_voltage, detector, reciprocal_mesh=False):
        self.wavelength = get_electron_wavelength(accelerating_voltage)
        # Always store a 'real' mesh
        self.detector = detector if not reciprocal_mesh else from_recip(detector)

    def calculate_ed_data(
        self,
        structure,
        probe,
        slice_thickness,
        probe_centre=None,
        z_range=200,
        precessed=False,
        dtype="float64",
        ZERO=1e-14,
        mode="kinematic",
        **kwargs,
    ):
        """
        Calculates single electron diffraction image for particular atomic
        structure and probe.

        Parameters
        ----------
        structure : Structure
            The structure for upon which to perform the calculation
        probe : instance of probeFunction
            Function representing 3D shape of beam
        slice_thickness : float
            Discretisation thickness in the z-axis
        probe_centre : ndarray (or iterable), shape [3] or [2]
            Translation vector for the probe. Either of the same dimension of the
            space or the dimension of the detector. default=None focusses the
            probe at [0,0,0]
        zrange : float
            z-thickness to discretise. Only required if sample is not thick enough to
            fully resolve the Ewald-sphere. Default value is 200.
        precessed : bool, float, or (float, int)
            Dictates whether beam precession is simulated. If False or the float is
            0 then no precession is computed. If <precessed> = (alpha, n) then the
            precession arc of tilt alpha (in degrees) is discretised into n
            projections. If n is not provided then default of 30 is used.
        dtype : str or numpy.dtype
            Defines the precision to use whilst computing diffraction image.
        ZERO : float > 0
            Rounding error permitted in computation of atomic density. This value is
            the smallest value rounded to 0. Default is 1e-14.
        mode : str
            Only <mode>='kinematic' is currently supported.
        kwargs : dictionary
            Extra key-word arguments to pass to child simulator.
            For kinematic: **GPU** (bool): Flag to use GPU if available,
            default is True. **pointwise** (bool): Flag to evaluate charge
            pointwise on voxels rather than average, default is False.

        Returns
        -------
        ndarray
            Diffraction data to be interpreted as a discretisation on the original
            detector mesh.

        """

        species = structure.element
        coordinates = structure.xyz_cartn.reshape(species.size, -1)
        dim = coordinates.shape[1]  # guarenteed to be 3

        if not ZERO > 0:
            raise ValueError("The value of the ZERO argument must be greater than 0")

        if probe_centre is None:
            probe_centre = np.zeros(dim)
        elif len(probe_centre) == (dim - 1):
            probe_centre = np.array(list(probe_centre) + [0])

        coordinates = coordinates - probe_centre[None]

        if not precessed:
            precessed = (float(0), 1)
        elif np.isscalar(precessed):
            precessed = (float(precessed), 30)

        dtype = np.dtype(dtype)
        dtype = round(dtype.itemsize / (1 if dtype.kind == "f" else 2))
        dtype = "f" + str(dtype), "c" + str(2 * dtype)

        # Filter list of atoms
        for d in range(dim - 1):
            ind = coordinates[:, d] >= self.detector[d].min() - 20
            coordinates, species = coordinates[ind, :], species[ind]
            ind = coordinates[:, d] <= self.detector[d].max() + 20
            coordinates, species = coordinates[ind, :], species[ind]

        # Add z-coordinate
        z_range = max(
            z_range, coordinates[:, -1].ptp()
        )  # enforce minimal resolution in reciprocal space
        x = [
            self.detector[0],
            self.detector[1],
            np.arange(
                coordinates[:, -1].min() - 20,
                coordinates[:, -1].min() + z_range + 20,
                slice_thickness,
            ),
        ]

        if mode == "kinematic":
            from diffsims.utils import kinematic_simulation_utils as simlib
        else:
            raise NotImplementedError(
                "<mode> = %s is not currently supported" % repr(mode)
            )

        kwargs["dtype"] = dtype
        kwargs["ZERO"] = ZERO
        return simlib.get_diffraction_image(
            coordinates, species, probe, x, self.wavelength, precessed, **kwargs
        )
