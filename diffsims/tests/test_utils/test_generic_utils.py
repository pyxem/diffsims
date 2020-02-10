'''
Created on 3 Dec 2019

@author: Rob Tovey
'''

import pytest
from diffsims.utils.atomic_diffraction_generator_support.generic_utils import GLOBAL_BOOL, get_grid, to_mesh
import numpy as np


@pytest.mark.parametrize('start, end', [(True, True), (True, False), (False, True), (False, False)])
def test_global_bool(start, end):
    x = GLOBAL_BOOL(start)
    assert bool(x) == start

    if end:
        x.set(end)
    else:
        x(end)
    assert bool(x) == end
    assert str(x) == str(end)


@pytest.mark.parametrize('sz, tpb_in, grid_out, tpb_out',
                         [((10,), None, [1], [10]),
                          ((10, 18), 7, [2, 3], [5, 6])])
def test_getGrid(sz, tpb_in, grid_out, tpb_out):
    grid, tpb = get_grid(sz, tpb_in)
    assert grid == grid_out
    assert tpb == tpb_out


@pytest.mark.parametrize('shape, dx, dtype',
                         [((5,), False, 'float32'),
                          ((5, 8), True, 'float64'),
                          ((8, 5), False, 'float64'),
                          ((5, 8, 13), False, 'complex64')])
def test_toMesh(shape, dx, dtype):
    x = [np.linspace(0, 1, s + 1)[1:] for s in shape]

    if dx:
        dx = list(np.eye(len(shape), len(shape))[::-1])
    else:
        dx = None

    var1 = to_mesh(x, dx, dtype)
    var2 = to_mesh(x, dx).astype(dtype)

    if len(shape) == 2:
        flag = dx is None
        for i in range(shape[0]):
            for j in range(shape[1]):
                assert abs(var1[i, j, 0 if flag else 1] - x[0][i]) < 1e-16
                assert abs(var1[i, j, 1 if flag else 0] - x[1][j]) < 1e-16

    assert var1.dtype == var2.dtype
    assert var1.shape == var2.shape
    np.testing.assert_allclose(var1, var2, 1e-16, 1e-16)
