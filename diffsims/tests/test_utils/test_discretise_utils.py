'''
Created on 1 Nov 2019

@author: Rob Tovey
'''

import pytest
import numpy as np
from diffsims.utils.discretise_utils import (getA, getDiscretisation, _CUDA,
    rebin)
dtype, ZERO = ('f4', 'c8'), 1e-10
params = {'dtype':('f4', 'c8'), 'ZERO':1e-10, 'GPU':False}


def _toMesh(x):
    y = np.meshgrid(*x, indexing='ij')
    return np.concatenate([z[..., None] for z in y], axis=-1)


def create_atoms(n, shape):
    coords = np.concatenate(
        [np.linspace(0, s, len(n)).reshape(-1, 1) for s in shape],
        axis=1)
    species = np.array(n)
    return coords, species


@pytest.mark.parametrize('Z, returnFunc', [(i, bool(1 + (-1) ** i)) for i in range(21)])
def test_getA(Z, returnFunc):
    a, b = getA(Z, returnFunc)
    if returnFunc:
        x = [np.linspace(0, 1, 10), np.linspace(0, 1, 21), np.linspace(0, 1, 32)]
        x = _toMesh(x)
        f = a(x)
        FT = b(x)
        assert x.shape == (10, 21, 32, 3)
        assert f.shape == (10, 21, 32)
        assert FT.shape == (10, 21, 32)
        assert f.min() >= 0
    else:
        assert a.min() >= 0 or Z == 19
        assert b.min() >= 0
        assert len(a) == len(b)


@pytest.mark.xfail(raises=ValueError)
def test_fail_getA(): getA(np.array([0, 1]))


@pytest.mark.parametrize('r', [.5, pytest.param(.1, marks=pytest.mark.xfail), pytest.param(.1, marks=pytest.mark.xfail)])
def test_rebin(r):
    x = [np.linspace(0, 1, 10), np.linspace(0, 1, 21), np.linspace(0, 1, 32)]
    loc = np.concatenate([0 * np.linspace(0, 1, 500)[:, None], ] * 3, axis=1)
    k = 1
    mem = 1e5
    rebin(x, loc, r, k, mem)


@pytest.mark.parametrize('n, shape', [
    ([0], (1, 1, 1)),
    ([1, 14], (10, 20, 30)),
    ([14, 14, 14], (10, 20, 30)),
])
def test_getDiscretisation(n, shape):
    atoms, species = create_atoms(n, shape)
    x = [np.linspace(0, .1, g) for g in (10, 21, 32)]
    X = _toMesh(x)

    f1 = getDiscretisation(atoms, species, x, pointwise=True, FT=False, **params)
    FT1 = getDiscretisation(atoms, species, x, pointwise=True, FT=True, **params)
    f2, FT2 = 0, 0
    for i in range(len(n)):
        a, b = getA(n[i], True)
        f2 = f2 + a(X - atoms[i].reshape(1, 1, -1))
        FT2 = FT2 + b(X) * np.exp(-1j * (X * atoms[i].reshape(1, 1, -1)).sum(-1))

    np.testing.assert_allclose(f1, f2, 1e-1)
    np.testing.assert_allclose(FT1, FT2, 1e-1)


@pytest.mark.parametrize('n, shape, grid', [
    ([0], (1, 1, 1), (3, 4, 1)),
    ([1, 14], (10, 20, 30), (3, 1, 5)),
    ([14, 14, 14], (10, 20, 30), (1, 4, 5)),
])
def test_getDiscretisation2(n, shape, grid):
    atoms, species = create_atoms(n, shape)
    x = [np.linspace(0, .1, g) for g in grid]

    for b1 in (True, False):
        for b2 in (True, False):
            f = getDiscretisation(atoms, species, x, pointwise=b1, FT=b2, **params)
            assert f.shape == grid


@pytest.mark.parametrize('n, shape', [
    ([10, 15, 20], (1, 2, 3)),
    ([14, 14, 14], (1, 2, 3)),
])
def test_pointwise(n, shape):
    atoms, species = create_atoms(n, shape)
    x = [np.linspace(0, .01, g) for g in (10, 21, 32)]

    pw_f = getDiscretisation(atoms, species, x, pointwise=True, FT=False, **params)
    pw_FT = getDiscretisation(atoms, species, x, pointwise=True, FT=True, **params)
    av_f = getDiscretisation(atoms, species, x, pointwise=False, FT=False, **params)
    av_FT = getDiscretisation(atoms, species, x, pointwise=True, FT=True, **params)

    np.testing.assert_allclose(pw_f, av_f, 1e-1)
    np.testing.assert_allclose(pw_FT, av_FT, 1e-1)


if _CUDA:

    @pytest.mark.parametrize('n, shape', [
        ([20, 14], (10, 20, 30)),
        ([14, 14, 14], (10, 20, 30)),
    ])
    def test_CUDA(n, shape):
        atoms, species = create_atoms(n, shape)
        x = [np.linspace(0, .01, g) for g in (10, 21, 32)]

        params['GPU'] = False
        cpu_f = getDiscretisation(atoms, species, x, pointwise=True, FT=False, **params)
        cpu_FT = getDiscretisation(atoms, species, x, pointwise=True, FT=True, **params)
        params['GPU'] = True
        gpu_f = getDiscretisation(atoms, species, x, pointwise=True, FT=False, **params)
        gpu_FT = getDiscretisation(atoms, species, x, pointwise=True, FT=True, **params)

        np.testing.assert_allclose(cpu_f, gpu_f, 1e-4)
        np.testing.assert_allclose(cpu_FT, gpu_FT, 1e-4)
