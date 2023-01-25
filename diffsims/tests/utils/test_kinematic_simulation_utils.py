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

import pytest
import numpy as np
from diffsims.utils.kinematic_simulation_utils import (
    get_diffraction_image,
    precess_mat,
    grid2sphere,
)


def to_mesh(x):
    y = np.meshgrid(*x, indexing="ij")
    return np.concatenate([z[..., None] for z in y], axis=-1)


def create_atoms(n, shape):
    coords = np.concatenate(
        [np.linspace(0, s / 2, len(n)).reshape(-1, 1) for s in shape], axis=1
    )
    species = np.array(n)
    return coords, species


def probe(x, out=None, scale=None):
    if len(x) == 3:
        v = abs(x[0].reshape(-1, 1, 1)) < 6
        v = v * abs(x[1].reshape(1, -1, 1)) < 6
        v = v + 0 * x[2].reshape(1, 1, -1)
    else:
        v = abs(x[..., :2]).max(-1) < 6
    if scale is not None:
        v = v * scale
    if out is None:
        return v
    else:
        out[...] = v
        return out


@pytest.mark.parametrize(
    "n, vol_shape, grid_shape, precession, wavelength",
    [
        ([0], (0.7, 0.7, 0.7), (10, 11, 12), False, 1e-8),
        ([10, 14], (10, 20, 30), (10, 10, 10), False, 0),
        ([14], (5, 10, 15), (10,) * 3, True, 0),
        ([14], (17, 30, 25), (10,) * 3, True, 1e-8),
    ],
)
def test_get_diffraction_image(n, vol_shape, grid_shape, precession, wavelength):
    coords, species = create_atoms(n, vol_shape)
    x = [np.linspace(0, vol_shape[i], grid_shape[i]) for i in range(3)]
    if precession:
        precession = (1e-3, 20)
    else:
        precession = (0, 1)

    params = {"dtype": ("f4", "c8"), "ZERO": 1e-10, "GPU": False, "pointwise": True}
    val1 = get_diffraction_image(
        coords, species, probe, x, wavelength, precession, **params
    )
    val2 = get_diffraction_image(coords, species, probe, x, 0, (0, 1), **params)

    if precession[0] > 0:
        val1 = val1[2:-2, 2:-2]
        val2 = val2[2:-2, 2:-2]

    assert val1.shape == val2.shape
    if precession[0] == 0:
        assert val1.shape == grid_shape[:2]
    np.testing.assert_allclose(val1, val2, 1e-2, 1e-4)


@pytest.mark.parametrize(
    "alpha, theta, x",
    [
        (0, 10, (-1, 0, 1)),
        (10, 0, (1, 2, 3)),
        (5, 10, (-1, 1, -1)),
    ],
)
def test_precess_mat(alpha, theta, x):
    R = precess_mat(alpha, theta)
    Ra = precess_mat(alpha, 0)
    Rt = precess_mat(theta, 0)[::-1, ::-1].T
    x = np.array(x)

    def angle(v1, v2):
        return (v1 * v2).sum() / np.linalg.norm(v1) / np.linalg.norm(v2)

    np.testing.assert_allclose(
        np.cos(np.deg2rad(theta)), angle(x[:2], Rt.dot(x)[:2]), 1e-5, 1e-5
    )
    np.testing.assert_allclose(
        np.cos(np.deg2rad(alpha)), angle(x[1:], Ra.dot(x)[1:]), 1e-5, 1e-5
    )
    assert abs(R[2, 2] - np.cos(np.deg2rad(alpha))) < 1e-5
    assert abs(R - Rt.T.dot(Ra.dot(Rt))).max() < 1e-5


@pytest.mark.parametrize(
    "shape, rad",
    [
        ((10,) * 3, 100),
        ((10, 20, 20), 200),
        ((10, 20, 30), 300),
        ((10, 20, 1), 1e10),
    ],
)
def test_grid2sphere(shape, rad):
    x = [np.linspace(-1, 1, s) if s > 1 else np.array([0]) for s in shape]
    X = to_mesh(x)
    Y = to_mesh((x[0], x[1], np.array([0]))).reshape(-1, 3)
    w = 1 / (1 + (Y ** 2).sum(-1) / rad ** 2)
    Y *= w[..., None]
    Y[:, 2] = rad * (1 - w)
    Y = Y.reshape(shape[0], shape[1], 3)

    for i in range(3):
        np.testing.assert_allclose(
            Y[..., i], grid2sphere(X[..., i], x, None, rad), 1e-4, 1e-4
        )
        if X.shape[i] == 1:
            S = [slice(None)] * X.ndim
            S[i], S[-1] = 0, i
            np.testing.assert_allclose(
                Y[..., i], grid2sphere(X[tuple(S)], x, None, rad), 1e-4, 1e-4
            )
