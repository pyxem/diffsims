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


def _shape_factor_precession(
    excitation_error, r_spot, phi, shape_function, max_excitation, **kwargs
):
    """
    The rel-rod shape factors for reflections taking into account
    precession

    Parameters
    ----------
    excitation_error : np.ndarray (N,)
        An array of excitation errors
    r_spot : np.ndarray (N,)
        An array representing the distance of spots from the z-axis in A^-1
    phi : float
        The precession angle in radians
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
    shf = np.zeros(excitation_error.shape)
    # loop over all spots
    for i, (excitation_error_i, r_spot_i) in enumerate(zip(excitation_error, r_spot)):

        def integrand(theta):
            # Equation 8 in L.Palatinus et al. Acta Cryst. (2019) B75, 512-522
            S_zero = excitation_error_i
            variable_term = r_spot_i * (phi) * np.cos(theta)
            return shape_function(S_zero + variable_term, max_excitation, **kwargs)

        # average factor integrated over the full revolution of the beam
        shf[i] = (1 / (2 * np.pi)) * quad(integrand, 0, 2 * np.pi)[0]
    return shf


class DiffractionGenerator(object):
    r"""Computes electron diffraction patterns for a crystal structure.

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
        "lobato", "xtables" or None. If None is provided then atomic
        scattering is not taken into consideration.
    precession_angle : float
        Angle about which the beam is precessed in degrees. Default is
        no precession.
    shape_factor_model : func or str
        A function that takes excitation_error and
        `max_excitation_error` (and potentially kwargs) and returns
        an intensity scaling factor. If None defaults to
        `shape_factor_models.linear`. A number of pre-programmed
        functions are available via strings.
    approximate_precession : bool
        When using precession, whether to precisely calculate average
        excitation errors and intensities or use an approximation.
    minimum_intensity : float
        Minimum intensity for a peak to be considered visible in the
        pattern (fractional from the maximum).
    kwargs
        Keyword arguments passed to `shape_factor_model`.

    Notes
    -----
    When using precession and approximate_precession=True, the shape
    factor model defaults to Lorentzian; shape_factor_model is ignored.
    Only with `approximate_precession=False` the
    `custom shape_factor_model` is used.
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
        self.precession_angle = np.abs(precession_angle)
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
        if scattering_params in ["lobato", "xtables", None]:
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
        shape_factor_width=None,
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
            The cut-off for geometric excitation error in the z-direction
            in units of reciprocal Angstroms. Spots with a larger distance
            from the Ewald sphere are removed from the pattern.
            Related to the extinction distance and roungly equal to 1/thickness.
        shape_factor_width : float
            Determines the width of the reciprocal rel-rod, for fine-grained
            control. If not set will be set equal to max_excitation_error.
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
        g_indices, cartesian_coordinates, g_hkls = get_points_in_sphere(
            recip_latt, reciprocal_radius
        )

        ai, aj, ak = (
            np.deg2rad(rotation[0]),
            np.deg2rad(rotation[1]),
            np.deg2rad(rotation[2]),
        )
        R = euler2mat(ai, aj, ak, axes="rzxz")
        cartesian_coordinates = np.matmul(R, cartesian_coordinates.T).T

        # Identify the excitation errors of candidate points
        r_sphere = 1 / wavelength
        r_spot = np.sqrt(np.sum(np.square(cartesian_coordinates[:, :2]), axis=1))
        z_spot = cartesian_coordinates[:, 2]

        z_sphere = -np.sqrt(r_sphere ** 2 - r_spot ** 2) + r_sphere
        excitation_error = z_sphere - z_spot

        # determine the pre-selection reflections
        if self.precession_angle == 0:
            intersection = np.abs(excitation_error) < max_excitation_error
        else:
            # only consider points that intersect the ewald sphere at some point
            # the center point of the sphere
            P_z = r_sphere * np.cos(np.deg2rad(self.precession_angle))
            P_t = r_sphere * np.sin(np.deg2rad(self.precession_angle))
            # the extremes of the ewald sphere
            z_surf_up = P_z - np.sqrt(r_sphere ** 2 - (r_spot + P_t) ** 2)
            z_surf_do = P_z - np.sqrt(r_sphere ** 2 - (r_spot - P_t) ** 2)
            intersection = (z_spot - max_excitation_error <= z_surf_up) & (
                z_spot + max_excitation_error >= z_surf_do
            )

        # select these reflections
        intersection_coordinates = cartesian_coordinates[intersection]
        excitation_error = excitation_error[intersection]
        r_spot = r_spot[intersection]
        g_indices = g_indices[intersection]
        g_hkls = g_hkls[intersection]

        if shape_factor_width is None:
            shape_factor_width = max_excitation_error
        # select and evaluate shape factor model
        if self.precession_angle == 0:
            # calculate shape factor
            shape_factor = self.shape_factor_model(
                excitation_error, shape_factor_width, **self.shape_factor_kwargs
            )
        else:
            if self.approximate_precession:
                shape_factor = lorentzian_precession(
                    excitation_error,
                    shape_factor_width,
                    r_spot,
                    np.deg2rad(self.precession_angle),
                )
            else:
                shape_factor = _shape_factor_precession(
                    excitation_error,
                    r_spot,
                    np.deg2rad(self.precession_angle),
                    self.shape_factor_model,
                    shape_factor_width,
                    **self.shape_factor_kwargs,
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

        # Threshold peaks included in simulation as factor of maximum intensity.
        peak_mask = intensities > np.max(intensities) * self.minimum_intensity
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
