from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from orix.quaternion import Rotation
from orix.vector.vector3d import Vector3d
from orix.projections import StereographicProjection


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

    def to_stereographic(self,
                         pole: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """Convert the rotation to a stereographic projection along some
        pole direction.

        Parameters
        ----------
        pole
            The pole of the stereographic projection
        """
        s = StereographicProjection(pole=pole)
        rot_reg_test = self * Vector3d.zvector()
        x, y = s.vector2xy(rot_reg_test)
        return x, y

    def plot(self,
             ax: plt.Axes = None,
             pole: int = -1,
             **kwargs,
             ):
        """Plot the stereographic projection of the rotation

        Parameters
        ----------
        ax
            The axis to plot on. If None, a new figure is created.
        pole
            The pole of the stereographic projection
        kwargs
            Additional keyword arguments to pass to ax.scatter

        """
        x, y = self.to_stereographic(pole=pole)
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(x, y, **kwargs)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        max_v = np.max([xlim[1], ylim[1]])
        min_v = np.min([xlim[0], ylim[0]])
        ax.set_xlim((min_v, max_v))
        ax.set_ylim((min_v, max_v))
        return ax

