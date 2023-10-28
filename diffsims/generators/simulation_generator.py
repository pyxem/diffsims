from typing import Union, Sequence
import numpy as np

from orix.quaternion import Rotation
from orix.crystal_map import Phase

from diffsims.crystallography import ReciprocalLatticeVector
from diffsims.simulations.simulation import Simulation, ProfileSimulation
from diffsims.utils.shape_factor_models import (
    linear,
    atanc,
    lorentzian,
    sinc,
    sin2c,
    lorentzian_precession,
    _shape_factor_precession,
)

from diffsims.utils.sim_utils import (
    get_electron_wavelength,
    get_kinematical_intensities,
    is_lattice_hexagonal,
)

_shape_factor_model_mapping = {
    "linear": linear,
    "atanc": atanc,
    "sinc": sinc,
    "sin2c": sin2c,
    "lorentzian": lorentzian,
}


class SimulationGenerator:
    def __init__(
        self,
        accelerating_voltage: float = 200,
        scattering_params: str = "lobato",
        precession_angle: float = 0,
        shape_factor_model: str = "lorentzian",
        approximate_precession: bool = True,
        minimum_intensity: float = 1e-20,
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
        phase: Union[Phase, Sequence[Phase]],
        rotation: Union[Rotation, Sequence[Rotation]] = Rotation.from_euler(
            (0, 0, 0), degrees=True
        ),
        reciprocal_radius: float = 1.0,
        with_direct_beam: bool = True,
        max_excitation_error: float = 1e-2,
        shape_factor_width: float = None,
        debye_waller_factors: dict = None,
    ):
        """Calculates the Electron Diffraction data for a structure.

        Parameters
        ----------
        phase:
            The phase(s) for which to derive the diffraction pattern.
        reciprocal_radius : float
            The maximum radius of the sphere of reciprocal space to
            sample, in reciprocal Angstroms.
        rotation
            The Rotation object(s) to apply to the structure and then
            calculate the diffraction pattern.
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
        if isinstance(phase, Phase):
            phase = [phase]
        if isinstance(rotation, Rotation):
            rotation = [rotation]
        if debye_waller_factors is None:
            debye_waller_factors = {}
        # Specify variables used in calculation
        wavelength = self.wavelength

        # Rotate using all the rotations in the list
        vectors = []
        for p, rotate in zip(phase, rotation):
            recip = ReciprocalLatticeVector.from_min_dspacing(
                p,
                min_dspacing=1 / reciprocal_radius,
                include_zero_beam=with_direct_beam,
            )
            phase_vectors = []
            for rot in rotate.to_matrix():
                # Calculate the reciprocal lattice vectors that intersect the Ewald sphere.
                intersected_vectors, shape_factor = self.get_intersecting_reflections(
                    recip,
                    rot,
                    wavelength,
                    max_excitation_error,
                    shape_factor_width=shape_factor_width,
                )

                # Calculate diffracted intensities based on a kinematic model.
                intensities = get_kinematical_intensities(
                    p.structure,
                    intersected_vectors.hkl,
                    intersected_vectors.gspacing,
                    prefactor=shape_factor,
                    scattering_params=self.scattering_params,
                    debye_waller_factors=debye_waller_factors,
                )

                # Threshold peaks included in simulation as factor of maximum intensity.
                peak_mask = intensities > np.max(intensities) * self.minimum_intensity
                intensities = intensities[peak_mask]
                intersected_vectors = intersected_vectors[peak_mask]
                intersected_vectors.intensity = intensities
                phase_vectors.append(intersected_vectors)
            vectors.append(phase_vectors)

        if len(phase) == 1:
            vectors = vectors[0]
            phase = phase[0]
            rotation = rotation[0]
        if rotation.size == 1:
            vectors = vectors[0]

        # Create a simulation object
        sim = Simulation(
            phases=phase,
            coordinates=vectors,
            rotations=rotation,
            simulation_generator=self,
        )
        return sim

    def get_intersecting_reflections(
        self,
        recip: ReciprocalLatticeVector,
        rot: np.ndarray,
        wavelength: float,
        max_excitation_error: float,
        shape_factor_width: float = None,
    ):
        """Calculates the reciprocal lattice vectors that intersect the Ewald sphere."""
        rotated_vectors = recip.rotate_from_matrix(rot)
        # Identify the excitation errors of all points (distance from point to Ewald sphere)
        r_sphere = 1 / wavelength
        r_spot = np.sqrt(np.sum(np.square(rotated_vectors.data[:, :2]), axis=1))
        z_spot = rotated_vectors.data[:, 2]

        z_sphere = -np.sqrt(r_sphere**2 - r_spot**2) + r_sphere
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
            z_surf_up = P_z - np.sqrt(r_sphere**2 - (r_spot + P_t) ** 2)
            z_surf_do = P_z - np.sqrt(r_sphere**2 - (r_spot - P_t) ** 2)
            intersection = (z_spot - max_excitation_error <= z_surf_up) & (
                z_spot + max_excitation_error >= z_surf_do
            )

        # select these reflections
        intersected_vectors = rotated_vectors[intersection]
        excitation_error = excitation_error[intersection]
        r_spot = r_spot[intersection]

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
        return intersected_vectors, shape_factor

    def calculate_profile_data(
        self,
        phase: Phase,
        reciprocal_radius: float = 1.0,
        minimum_intensity: float = 1e-3,
        debye_waller_factors: dict = None,
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
        latt = phase.structure.lattice

        # Obtain crystallographic reciprocal lattice points within range
        vectors = ReciprocalLatticeVector.from_min_dspacing(
            phase,
            min_dspacing=1 / reciprocal_radius,
        )

        unique_vectors = vectors.unique(use_symmetry=True).symmetrise()

        multiplicity = unique_vectors.multiplicity
        g_indices = unique_vectors.hkl
        g_hkls = unique_vectors.gspacing

        i_hkl = get_kinematical_intensities(
            phase.structure,
            g_indices,
            np.asarray(g_hkls),
            prefactor=multiplicity,
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
