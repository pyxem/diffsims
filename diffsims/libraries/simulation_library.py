from typing import NamedTuple, Sequence

from orix.quaternion import Rotation
from orix.crystal_map import Phase
from orix.vector import Vector3d
import numpy as np

from diffsims.sims.diffraction_simulation import DiffractionSimulation
from diffsims.generators.diffraction_generator import DiffractionGenerator


class SimulationLibrary(NamedTuple):
    phase: Phase
    rotations: Rotation
    diffraction_generator: DiffractionGenerator
    simulations: Sequence[DiffractionSimulation]
    str_rotations: Sequence[str] = None

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

    def rotations_to_vectors(self,
                             beam_direction: Vector3d = None) -> Vector3d:
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

    def plot_rotations(self,
                       beam_direction: Vector3d = None,
                       **kwargs):
        """Plots all the diffraction patterns in the library

        Parameters
        ----------
        beam_direction
            The beam direction used to determine the vectors based on the rotations
        """
        vectors = self.rotations_to_vectors(beam_direction)
        vectors.scatter(**kwargs)


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
