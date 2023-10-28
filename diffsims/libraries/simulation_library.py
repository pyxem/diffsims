from typing import NamedTuple, Sequence, TYPE_CHECKING

from orix.quaternion import Rotation
from orix.crystal_map import Phase
from orix.vector import Vector3d
import numpy as np

from diffsims.simulations.simulation import Simulation as DiffractionSimulation

if TYPE_CHECKING:
    from diffsims.generators.simulation_generator import SimulationGenerator


class SimulationLibrary(NamedTuple):
    phase: Sequence[Phase]
    rotations: Sequence[Rotation]
    diffraction_generator: "SimulationGenerator"
    simulations: Sequence[DiffractionSimulation]
    str_rotations: Sequence[str] = None
    calibration: float = None

    def __repr__(self):
        return (
            f"DiffractionPhaseLibrary(phase={self.phase.name},"
            f" No. Rotations={self.__len__()})"
        )

    def __post_init__(self):
        if len(self.rotations) != len(self.simulations):
            raise ValueError("Number of rotations and simulations must be the same")
        if self.str_rotations is not None and len(self.rotations) != len(
            self.str_rotations
        ):
            raise ValueError("Number of rotations and str_rotations must be the same")
        self.simulation_index = 0  # for interactive plotting

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.str_rotations.index(item)
        return SimulationLibrary(
            self.phase, self.rotations[item], self.simulations[item]
        )

    def get_library_entry(
        self, rotation: Rotation, angle_cutoff: float = 1e-2
    ) -> "DiffractionPhaseLibrary":
        angles = self.rotations.angle_with(rotation)
        is_in_range = np.sum(np.abs(angles), axis=1) < angle_cutoff
        return self[is_in_range]

    def rotations_to_vectors(self, beam_direction: Vector3d = None) -> Vector3d:
        """Converts the rotations to vectors

        Parameters
        ----------
        beam_direction
            The beam direction used to determine the vectors based on the rotations
        """
        if beam_direction is None:
            beam_direction = Vector3d.zvector()
        vectors = self.rotations * beam_direction
        return vectors

    def max_num_spots(self):
        """Returns the maximum number of spots in the library"""
        return max([i.intensities.shape[0] for i in self.simulations])

    def plot_rotations(self, beam_direction: Vector3d = None, **kwargs):
        """Plots all the diffraction patterns in the library

        Parameters
        ----------
        beam_direction
            The beam direction used to determine the vectors based on the rotations
        """
        vectors = self.rotations_to_vectors(beam_direction)
        vectors.scatter(**kwargs)

    def polar_flatten_simulations(self):
        """Flatten the simulations into arrays of shape (n_simulations, max(num_diffraction_spots))
        for the polar coordinates (r,theta, intensity) of the diffraction spots
        """
        max_num_spots = self.max_num_spots()

        r = np.zeros((len(self), max_num_spots))
        theta = np.zeros((len(self), max_num_spots))
        intensity = np.zeros((len(self), max_num_spots))

        for i, sim in enumerate(self.simulations):
            (
                r[i, : sim.intensities.shape[0]],
                theta[i, : sim.intensities.shape[0]],
            ) = sim.get_polar_coordinates()
            intensity[i, : sim.intensities.shape[0]] = sim.intensities
        return r, theta, intensity


class SimulationLibraries(dict):
    """
    A dictionary containing all the structures and their associated rotations
    """

    def __init__(self, libraries: Sequence[SimulationLibrary]):
        super().__init__()
        for library in libraries:
            self[library.phase.name] = library

    def __repr__(self):
        return f"DiffractionLibrary<Phases:{list(self.keys())}>)"
