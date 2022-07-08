from __future__ import annotations

import numpy as np
from orix.quaternion import Rotation
from orix.vector import Vector3d


class ConstrainedRotation(Rotation):
    """A rotation where phi1 = 0 in the Euler angle representation for the
    bunge convention.
    """

    @classmethod
    def from_vector(
        cls,
        vector: Vector3d,
    ) -> ConstrainedRotation:
        """Construct ConstrainedRotation from the 3D vector that should be
        where the z-axis points after rotation

        Parameters
        ----------
        vector
            The vector(s) that should be aligned with the beam direction after
            rotation.

        Returns
        -------
        ConstrainedRotation
        """
        grid = cls._vectors_to_euler_angles(vector)
        rotations = cls.from_euler(grid)
        return rotations

    @property
    def corresponding_beam_direction(self) -> Vector3d:
        return self * Vector3d.zvector()

    @classmethod
    def _vectors_to_euler_angles(
        cls,
        vector: Vector3d,
    ) -> np.ndarray:
        Phi = vector.polar
        phi2 = (np.pi/2 - vector.azimuth) % (2*np.pi)
        phi1 = np.zeros(phi2.shape[0])
        return np.vstack([phi1, Phi, phi2]).T

    def to_euler(
        self,
    ) -> np.ndarray:
        vector = self.corresponding_beam_direction
        return self._vectors_to_euler_angles(vector)
