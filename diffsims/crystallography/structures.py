from orix.crystal_map import Phase
from diffsims.utils.sampling_utils import get_reduced_fundamental_zone_grid
from diffsims.utils.sampling_utils import generate_zap_rotations
from diffsims.crystallography import ReciprocalLatticeVector


class CrystalPhase(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._radius = 0
        self._reciprocal_space_lattice = None

    def zap_rotations(self,
                      density: str = "3"):
        """
        Returns a list of rotations that can be used to generate ZAP diffraction
        patterns.

        Parameters
        ----------
        density
            The density of directions to use. Options are '3' or '7' referring to
            3 for only the corners of the fundamental zone or 7 for the corners,
            midpoints and centroids.
        """

        return generate_zap_rotations(self,
                                   density=density)

    def constrained_rotation(self,
                             resolution: float = 1.0,
                             mesh: str = None,):
        """
        Returns all rotations for some crystal structure reduced to only the unique rotations
        by symmetry.

        Parameters
        ----------
        resolution : float
            The resolution of the grid (degrees).
        mesh : str
            The mesh type to use. Options are 'cuboctahedron', 'icosahedron', 'octahedron',

        """

        return get_reduced_fundamental_zone_grid(resolution,
                                                 mesh,
                                                 self.point_group)

    def reciprocal_lattice_vectors(self,
                                   reciprocal_radius: float = 10):
        """
        Returns the reciprocal space lattice vectors for a given radius for the structure.

        This is a cached property, so the first time it is called it will be slow, but
        subsequent calls will be fast.

        Parameters
        ----------
        reciprocal_radius
            The radius of the sphere in reciprocal space (units of reciprocal
            Angstroms) within which reciprocal lattice points are returned
        """
        if reciprocal_radius >= self._radius:
            # recalculate to a higher radius
            # this could be more efficient if we just calculated only the new points
            self._radius = reciprocal_radius
            self._reciprocal_lattice_vectors = ReciprocalLatticeVector.from_min_dspacing(self,
                                                                                      1/reciprocal_radius)
            return self._reciprocal_lattice_vectors
        else:
            # return the existing lattice sliced to the radius
            in_sphere = self._reciprocal_lattice_vectors.gspacing <= reciprocal_radius
            return self._reciprocal_lattice_vectors[in_sphere]


