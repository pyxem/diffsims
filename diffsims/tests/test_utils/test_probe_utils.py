'''
Created on 6 Nov 2019

@author: Rob Tovey
'''

import pytest
import numpy as np
from diffsims.utils.atomic_diffraction_generator_support.probe_utils import (ProbeFunction, BesselProbe)


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


@pytest.mark.xfail(raises=NotImplementedError)
def test_null_probe():
    p = ProbeFunction()
    p(1)


@pytest.fixture(params=[(10, 11, 12)])
def simple_mesh(request): return [np.linspace(-1, 1, n) for n in request.param]


@pytest.fixture(params=[(lambda x: (x.max(-1) < 1).astype(float),)])
def simple_probe(request): return ProbeFunction(*request.param)


class TestSimpleProbe:

    def test_init(self, simple_probe: ProbeFunction):
        assert simple_probe._func is not None

    def test_func(self, simple_probe, simple_mesh):
        arr1 = simple_probe(simple_mesh)
        arr2 = simple_probe(_toMesh(simple_mesh))
        assert arr1.shape == tuple(X.size for X in simple_mesh)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)

    def test_FT(self, simple_probe, simple_mesh):
        arr1 = simple_probe.FT(simple_mesh)
        arr2 = simple_probe.FT(_toMesh(simple_mesh))
        assert arr1.shape == tuple(X.size for X in simple_mesh)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)

    def test_func_kwargs(self, simple_probe, simple_mesh):
        out = np.empty([X.size for X in simple_mesh])
        scale = _random_array(out.shape, 0)
        arr1 = simple_probe(simple_mesh) * scale
        arr2 = simple_probe(simple_mesh, out=out, scale=scale)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)
        assert arr2 is out

    def test_FT_kwargs(self, simple_probe, simple_mesh):
        out = np.empty([X.size for X in simple_mesh], dtype='c16')
        arr1 = simple_probe.FT(simple_mesh)
        arr2 = simple_probe.FT(simple_mesh, out=out)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)
        assert arr2 is out


@pytest.fixture(params=[(10,), (10, 11), (10, 11, 12)])
def bess_mesh(request): return [np.linspace(-1, 1, n) for n in request.param]


@pytest.fixture(params=[(1,), (2,)])
def bess_probe(request): return BesselProbe(*request.param)


class TestBessProbe:

    def test_init(self, bess_probe: BesselProbe):
        assert bess_probe._func is None
        assert hasattr(bess_probe, 'r')
        assert hasattr(bess_probe, '_r')

    def test_func(self, bess_probe, bess_mesh):
        arr1 = bess_probe(bess_mesh)
        arr2 = bess_probe(_toMesh(bess_mesh))
        assert arr1.shape == tuple(X.size for X in bess_mesh)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)

    def test_FT(self, bess_probe, bess_mesh):
        arr1 = bess_probe.FT(bess_mesh)
        arr2 = bess_probe.FT(_toMesh(bess_mesh))
        assert arr1.shape == tuple(X.size for X in bess_mesh)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)

    def test_func_kwargs(self, bess_probe, bess_mesh):
        out = np.empty([X.size for X in bess_mesh])
        scale = _random_array(out.shape, 0)
        arr1 = bess_probe(bess_mesh) * scale
        arr2 = bess_probe(bess_mesh, out=out, scale=scale)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)
        assert arr2 is out

    def test_FT_kwargs(self, bess_probe, bess_mesh):
        out = np.empty([X.size for X in bess_mesh], dtype='c16')
        arr1 = bess_probe.FT(bess_mesh)
        arr2 = bess_probe.FT(bess_mesh, out=out)
        np.testing.assert_allclose(arr1, arr2, 1e-5, 1e-5)
        assert arr2 is out
