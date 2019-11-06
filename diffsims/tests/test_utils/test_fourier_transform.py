'''
Created on 2 Nov 2019

@author: Rob Tovey
'''

import pytest
import numpy as np
from diffsims.utils.fourier_transform import (plan_fft, plan_ifft, fftn, ifftn,
                                              ifftshift, fftshift, fftshift_phase,
                                              fast_abs, toFreq, fromFreq,
                                              getFTpoints, getDFT, convolve)
from diffsims.utils.discretise_utils import getA


def _toMesh(x):
    y = np.meshgrid(*x, indexing='ij')
    return np.concatenate([z[..., None] for z in y], axis=-1)


def _random_array(shape, n=0):
    x = np.zeros(np.prod(shape), dtype='float64')
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
              67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
              139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    primes = np.array(primes, dtype=int)[::3]
    for i, j in enumerate(primes[n:]):
        x[::j] += i * (-1) ** j
    return x.reshape(shape)


standard_test = ([(20,), 1], [(38,), 2],
                 [(10, 12), 3])


@pytest.mark.parametrize('shape, n', standard_test)
def test_plan(shape, n):
    x = _random_array(shape, n)
    f1, x1 = plan_fft(x)
    f2, x2 = plan_ifft(x)
    x1[...], x2[...] = x, x

    np.testing.assert_allclose(f1(), fftn(x), 1e-5)
    np.testing.assert_allclose(f2(), ifftn(x), 1e-5)


@pytest.mark.parametrize('shape, n', standard_test)
def test_fftshift(shape, n):
    x = _random_array(shape, n) + 1
    y = fftshift(fftn(x))
    z = ifftn(ifftshift(y))
    Y = x.copy()
    fftshift_phase(Y)
    Y = fftn(Y)

    np.testing.assert_allclose(x, z, 1e-5, 1e-5)
    np.testing.assert_allclose(y, Y, 1e-5, 1e-5)


@pytest.mark.parametrize('shape, n', standard_test)
def test_fast_abs(shape, n):
    x = _random_array(shape, n)
    y = np.empty(shape, dtype=x.dtype)
    z = fast_abs(x, y)

    np.testing.assert_allclose(abs(x), y, 1e-5)
    assert y is z


@pytest.mark.parametrize('shape, dX, rX, dY, rY', [
    ([200], (.1,), (1,), (.01,), (2,)),
    ([None], (.1,), (1,), (.01,), (2,)),
    ([5, 20], (.1,) * 2, (1,) * 2, (.01,) * 2, (2,) * 2),
    ([10] * 3, (.1, .1, .2), (1, 2, 1), (.01,) * 3, (2,) * 3),
])
def test_freq(shape, dX, rX, dY, rY):
    x, y = getFTpoints(len(shape), shape, dX, rX, dY, rY)
    X, Y = fromFreq(y), toFreq(x)

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


@pytest.mark.parametrize('rX, rY', [
    ([1], 1000),
    ([1] * 2, 1000),
])
def test_DFT(rX, rY):
    x, y = getFTpoints(len(rX), rX=rX, rY=rY)

    f, g = getA(0, returnFunc=True)
    f, g = f(_toMesh(x)), g(_toMesh(y))

    ft, ift = getDFT(x, y)

    np.testing.assert_allclose(f, ift(g), 1e-5, 1e-5)
    np.testing.assert_allclose(g, ft(f), 1e-5, 1e-5)


@pytest.mark.parametrize('shape1, shape2, n1, n2', [
    ([10], [10], 1, 5),
    ([10, 1], [10, 10], 2, 6),
    ([10, 10], [10, 1], 3, 7),
    ([10, 11], [10, 11], 4, 8),
])
def test_convolve(shape1, shape2, n1, n2):
    x = _random_array(shape1, n1)
    y = _random_array(shape2, n2)

    c1 = convolve(x, y)
    c2 = ifftn(fftn(x) * fftn(ifftshift(y)))

    assert c1.shape == c2.shape
    np.testing.assert_allclose(c1, c2, 1e-5, 1e-5)

