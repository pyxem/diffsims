from diffsims.crystallography import ReciprocalLatticeVector
from diffsims.crystallography._diffracting_vector import DiffractingVector
from orix.quaternion import Rotation

import pytest
import numpy as np


class TestDiffractingVector:
    def test_init(self, ferrite_phase):
        rlv = DiffractingVector(
            ferrite_phase, hkl=[[1, 1, 1], [2, 0, 0]], intensity=[1, 2]
        )
        assert rlv.phase == ferrite_phase
        assert rlv.shape == (2,)
        assert rlv.hkl.shape == (2, 3)
        assert np.allclose(rlv.hkl, [[1, 1, 1], [2, 0, 0]])
        assert np.allclose(rlv.intensity, [1, 2])

    def test_init_wrong_intensity_length(self, ferrite_phase):
        with pytest.raises(ValueError):
            DiffractingVector(ferrite_phase, hkl=[[1, 1, 1], [2, 0, 0]], intensity=[1])

    def test_add_intensity(self, ferrite_phase):
        rlv = DiffractingVector.from_min_dspacing(ferrite_phase, 1.5)
        rlv.intensity = 1
        assert isinstance(rlv.intensity, np.ndarray)
        assert np.allclose(rlv.intensity, np.ones(rlv.size))

    def test_add_intensity_error(self, ferrite_phase):
        rlv = DiffractingVector.from_min_dspacing(ferrite_phase, 1.5)
        with pytest.raises(ValueError):
            rlv.intensity = [0, 1]

    def test_slicing(self, ferrite_phase):
        rlv = DiffractingVector.from_min_dspacing(ferrite_phase, 1.5)
        rlv.intensity = 1
        rlv_slice = rlv[0:3]
        assert rlv_slice.size == 3
        assert np.allclose(rlv_slice.intensity, np.ones(3))

    def test_structure_factor(self, ferrite_phase):
        rlv = DiffractingVector.from_min_dspacing(ferrite_phase, 1.5)
        with pytest.raises(NotImplementedError):
            rlv.calculate_structure_factor()

    def test_hkl(self, ferrite_phase):
        rlv = ReciprocalLatticeVector(ferrite_phase, hkl=[[1, 1, 1], [2, 0, 0]])
        rot = Rotation.from_euler([90, 90, 0], degrees=True)
        rotated_vectors = (~rot * rlv.to_miller()).data
        ferrite_phase2 = ferrite_phase.deepcopy()
        ferrite_phase2.structure.lattice.setLatPar(baserot=rot.to_matrix()[0])
        dv = DiffractingVector(ferrite_phase2, xyz=rotated_vectors)
        assert np.allclose(rlv.hkl, dv.hkl)

    def test_flat_polar(self, ferrite_phase):
        dv = DiffractingVector(ferrite_phase, xyz=[[1, 1, 1], [0.5, -0.5, 0]])
        r, t = dv.to_flat_polar()
        assert np.allclose(r, [np.sqrt(2), 0.70710678])
        assert np.allclose(t, [np.pi / 4, -np.pi / 4])
        dv = DiffractingVector(
            ferrite_phase,
            xyz=[[[1, 1, 1], [0.5, -0.5, 0]], [[1, 1, 1], [0.5, -0.5, 0]]],
        )
        r, t = dv.to_flat_polar()
        assert np.allclose(r, [np.sqrt(2), np.sqrt(2), 0.70710678, 0.70710678])
        assert np.allclose(t, [np.pi / 4, np.pi / 4, -np.pi / 4, -np.pi / 4])

    def test_get_lattice_basis_rotation(self, ferrite_phase):
        """Rotation matrix to align the lattice basis with the Cartesian
        basis is correct.
        """
        rlv = DiffractingVector(ferrite_phase, hkl=[[1, 1, 1], [2, 0, 0]])
        rot = rlv.basis_rotation
        assert np.allclose(rot.to_matrix(), np.eye(3))

    def test_rotation_with_basis_raises(self, ferrite_phase):
        rlv = DiffractingVector(ferrite_phase, hkl=[[1, 1, 1], [2, 0, 0]])
        rot = Rotation.from_euler([[90, 90, 0], [90, 90, 1]], degrees=True)
        with pytest.raises(ValueError):
            rlv.rotate_with_basis(rotation=rot)
