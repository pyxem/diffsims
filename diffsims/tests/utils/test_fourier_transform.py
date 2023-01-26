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
from diffsims.utils.fourier_transform import (
    plan_fft,
    plan_ifft,
    fftn,
    ifftn,
    ifftshift,
    fftshift,
    fftshift_phase,
    fast_abs,
    to_recip,
    from_recip,
    get_recip_points,
    get_DFT,
    convolve,
)
from diffsims.utils.discretise_utils import get_atoms


def _toMesh(x):
    y = np.meshgrid(*x, indexing="ij")
    return np.concatenate([z[..., None] for z in y], axis=-1)


def _random_array(shape, n=0):
    x = np.zeros(np.prod(shape), dtype="float64")
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
    ]
    primes = np.array(primes, dtype=int)[::3]
    for i, j in enumerate(primes[n:]):
        x[::j] += i * (-1) ** j
    return x.reshape(shape)


standard_test = ([(20,), 1], [(38,), 2], [(10, 12), 3])


@pytest.mark.parametrize("shape, n", standard_test)
def test_plan(shape, n):
    x = _random_array(shape, n)
    f1, x1 = plan_fft(x)
    f2, x2 = plan_ifft(x)
    x1[...], x2[...] = x, x

    np.testing.assert_allclose(f1(), fftn(x), 1e-5)
    np.testing.assert_allclose(f2(), ifftn(x), 1e-5)


@pytest.mark.parametrize("shape, n", standard_test)
def test_fftshift(shape, n):
    x = _random_array(shape, n) + 1
    y = fftshift(fftn(x))
    z = ifftn(ifftshift(y))
    Y = x.copy()
    fftshift_phase(Y)
    Y = fftn(Y)

    np.testing.assert_allclose(x, z, 1e-5, 1e-5)
    np.testing.assert_allclose(y, Y, 1e-5, 1e-5)


@pytest.mark.parametrize("shape, n", standard_test)
def test_fast_abs(shape, n):
    x = _random_array(shape, n)
    y = np.empty(shape, dtype=x.dtype)
    z = fast_abs(x, y)

    np.testing.assert_allclose(abs(x), fast_abs(x), 1e-5)
    np.testing.assert_allclose(abs(x), y, 1e-5)
    assert y is z


@pytest.mark.parametrize(
    "shape, dX, rX, dY, rY",
    [
        ([200], (0.1,), (1,), (0.01,), (2,)),
        ([200], (0.1,), (1,), (6,), (60,)),
        ([None], (0.1,), (1,), (0.01,), (2,)),
        ([5, 20], (0.1,) * 2, (1,) * 2, (0.01,) * 2, (2,) * 2),
        ([10] * 3, (0.1, 0.1, 0.2), (1, 2, 1), (0.01,) * 3, (2,) * 3),
    ],
)
def test_freq(shape, dX, rX, dY, rY):
    x, y = get_recip_points(len(shape), shape, dX, rX, dY, rY)
    X, Y = from_recip(y), to_recip(x)

    assert len(x) == len(shape)
    assert len(y) == len(shape)
    assert len(X) == len(shape)
    assert len(Y) == len(shape)

    for i in range(len(shape)):
        assert y[i].size == x[i].size
        if shape[i] is None:
            if dX[i] is not None:
                assert abs(x[i].item(1) - x[i].item(0)) <= dX[i] + 1e-8
            if rY[i] is not None:
                assert y[i].ptp() >= rY[i] - 1e-8
        if rX[i] is not None:
            assert x[i].ptp() >= rX[i] - 1e-8
        if dY[i] is not None:
            assert abs(y[i].item(1) - y[i].item(0)) <= dY[i] + 1e-8

        np.testing.assert_allclose(x[i], X[i], 1e-5)
        np.testing.assert_allclose(y[i], Y[i], 1e-5)


@pytest.mark.parametrize("shape", [[200], [1, 10], [10, 1]])
def test_freq2(shape):
    x = [np.linspace(0, 1, s) if s > 1 else np.array([0]) for s in shape]
    y = to_recip(from_recip(x))
    z = from_recip(to_recip(x))

    for i in range(len(shape)):
        assert abs(x[i] - y[i] + y[i].min()).max() < 1e-6
        assert abs(x[i] - z[i] + z[i].min()).max() < 1e-6


@pytest.mark.parametrize(
    "rX, rY",
    [
        ([1], 1000),
        ([1] * 2, 1000),
        ([1] * 3, 1000),
    ],
)
def test_DFT(rX, rY):
    x, y = get_recip_points(len(rX), rX=rX, rY=rY)
    axes = 0 if len(x) == 1 else None

    f, g = get_atoms(0, returnFunc=True)
    f, g = f(_toMesh(x)), g(_toMesh(y))

    ft, ift = get_DFT(x, y)
    ft1, ift1 = get_DFT(X=x)
    ft2, ift2 = get_DFT(Y=y)

    for FT in (ft, ft1, ft2):
        np.testing.assert_allclose(g, FT(f, axes=axes), 1e-5, 1e-5)
    for IFT in (ift, ift1, ift2):
        np.testing.assert_allclose(f, IFT(g, axes=axes), 1e-5, 1e-5)


@pytest.mark.xfail(raises=ValueError)
def test_fail_DFT():
    get_DFT()


@pytest.mark.parametrize(
    "shape1, shape2, n1, n2, dx",
    [
        ([10], [10], 1, 5, None),
        ([10, 1], [10, 10], 2, 6, 1),
        ([10, 10], [10, 1], 3, 7, (1, 1)),
        ([10, 11], [10, 11], 4, 8, None),
        ([10], [10, 11], 5, 9, 1),
    ],
)
def test_convolve(shape1, shape2, n1, n2, dx):
    x = _random_array(shape1, n1)
    y = _random_array(shape2, n2)

    c1 = convolve(x, y, dx)
    s1 = shape1 + [1] * (len(shape2) - len(shape1))
    s2 = shape2 + [1] * (len(shape1) - len(shape2))
    if len(shape2) <= len(shape1):
        c2 = ifftn(fftn(x).reshape(s1) * fftn(ifftshift(y)).reshape(s2))
    else:
        c2 = ifftn(fftn(ifftshift(x)).reshape(s1) * fftn(y).reshape(s2))

    assert c1.shape == c2.shape
    np.testing.assert_allclose(c1, c2, 1e-5, 1e-5)
