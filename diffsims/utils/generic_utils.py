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

"""
Created on 31 Oct 2019

Generic tools for all areas of code.

@author: Rob Tovey
"""
from numpy import isscalar, zeros, array
import numba

# Coverage: Cuda code is not tested by travis
try:  # pragma: no cover
    from numba import cuda

    __CUDA = cuda.is_available()
except Exception:  # pragma: no cover
    cuda = None
    __CUDA = False


class GLOBAL_BOOL:
    """
    An object which behaves like a bool but can be changed in-place by `set`
    or by calling as a function.
    """

    def __init__(self, val):
        self.val = bool(val)

    def __call__(self, val):
        self.set(val)

    def set(self, val):
        self.val = bool(val)

    def __bool__(self):
        return self.val

    def __str__(self):
        return str(self.val)


_CUDA = GLOBAL_BOOL(__CUDA)


def get_grid(sz, tpb=None):
    dim = len(sz)

    if tpb is None:
        # Complete guess that feels reasonable
        tpb = min(int(512 ** (1 / dim)), 256)

    tpb = [tpb] * dim if isscalar(tpb) else list(tpb)
    grid = [0] * dim

    for i in range(dim):
        if tpb[i] > sz[i]:
            tpb[i] = sz[i]
            grid[i] = 1
        else:
            while tpb[i] * (sz[i] // tpb[i]) != sz[i]:
                tpb[i] -= 1
            grid[i] = sz[i] // tpb[i]

    return grid, tpb


# Coverage: Numba code does not register when code is run
@numba.njit(parallel=True, fastmath=True, cache=False)
def __toMesh2d(x0, x1, dx0, dx1, out):  # pragma: no cover
    for i0 in numba.prange(x0.size):
        X00 = x0[i0] * dx0[0]
        X01 = x0[i0] * dx0[1]
        for i1 in range(x1.size):
            out[i0, i1, 0] = X00 + x1[i1] * dx1[0]
            out[i0, i1, 1] = X01 + x1[i1] * dx1[1]


# Coverage: Numba code does not register when code is run
@numba.njit(parallel=True, fastmath=True, cache=False)
def __toMesh3d(x0, x1, x2, dx0, dx1, dx2, out):  # pragma: no cover
    for i0 in numba.prange(x0.size):
        X00 = x0[i0] * dx0[0]
        X01 = x0[i0] * dx0[1]
        X02 = x0[i0] * dx0[2]
        for i1 in range(x1.size):
            X10 = x1[i1] * dx1[0]
            X11 = x1[i1] * dx1[1]
            X12 = x1[i1] * dx1[2]
            for i2 in range(x2.size):
                out[i0, i1, i2, 0] = X00 + X10 + x2[i2] * dx2[0]
                out[i0, i1, i2, 1] = X01 + X11 + x2[i2] * dx2[1]
                out[i0, i1, i2, 2] = X02 + X12 + x2[i2] * dx2[2]


def to_mesh(x, dx=None, dtype=None):
    """
    Generates dense meshes from grid vectors, broadly:
        to_mesh(x)[i,j,...] = (x[0][i], x[1][j], ...)

    Parameters
    ----------
    x : `list` [`numpy.ndarray`], of shape [(nx,), (ny,), ...]
        List of grid vectors
    dx : `list` [`numpy.ndarray`] or `None`, optional
        Basis in which to expand mesh, default is the canonical basis
    dtype : `str` or `None`, optional
        String representing the `numpy` type of output, default inherits from `x`

    Returns
    -------
    X : `numpy.ndarray` [dtype], (x[0].size, x[1].size, ..., len(x))
        X[i,j,..., k] = x[0][i]*dx[0][k] + x[1][j]*dx[1][k] + ...

    """
    shape = [xi.size for xi in x]
    if dtype is None:
        dtype = x[0].dtype
    else:
        x = [xi.astype(dtype, copy=False) for xi in x]
    dim = len(shape)
    X = zeros(shape + [dim], dtype=dtype)

    if dim == 2:
        dx = [array([1, 0]), array([0, 1])] if dx is None else list(dx)
        __toMesh2d(*x, *dx, X)
        return X
    elif dim == 3:
        dx = (
            [array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])]
            if dx is None
            else list(dx)
        )
        __toMesh3d(*x, *dx, X)
        return X

    # TODO: this ignores dx, probably just raise error on dim=4?
    dim = X.shape[-1]
    for i in range(len(x)):
        sz = [1] * dim
        sz[i] = -1
        X[..., i] += x[i].reshape(sz)
    return X
