'''
Created on 5 Nov 2019

@author: Rob Tovey
'''
from diffsims.utils.fourier_transform import getDFT, fromFreq
from diffsims.utils.generic_utils import toMesh
import numba
from math import sqrt as c_sqrt
from numpy import empty, maximum, sqrt, arange, pi, linspace, ones
from scipy.special import jv


class probeFunction:
    '''
    Object representing a probe function.

    Parameters
    ----------
    func : function
        Function which takes in an array, r, of shape [nx, ny, nz, 3] and
        returns an array of shape [nx, ny, nz]. r[...,0] corresponds to the
        x coordinate, r[..., 1] to y etc. If not provided (or None) then the
        __call__ and FT methods must be overrided.

    Attributes
    ----------
    __call__ : method(x, out=None, scale=None)
        Returns func(x)*scale. If out is provided then it is used as
        preallocated storage. If scale is not provided then it is assumed
        to be 1. If x is a list of arrays it is converted into a mesh first.

    FT : method(y, out=None)
        Returns the Fourier transform of func on the mesh y. Again, if out is
        provided then computation is inplace. If y is a list of arrays then
        it is converted into a mesh first. If this function is not overridden
        then an approximation is made using func and the fft.


    '''

    def __init__(self, func=None):
        self._func = func

    def __call__(self, x, out=None, scale=None):
        '''
        Parameters
        ----------
        x : array of shape [nx, ny, nz, 3] or list of arrays size nx, ny and nz
            Mesh points at which to evaluate the probe density
        out : array of shape [nx, ny, nz]
            If provided then computation is performed inplace
        scale : array of shape [nx, ny, nz]
            If provided then the probe density is scaled by this before being
            returned.

        Returns
        -------
        out : array of shape [nx, ny, nz]
            An array equal to probe(x)*scale

        '''
        if self._func is None:
            raise NotImplementedError

        if not(hasattr(x, 'shape')):
            x = toMesh(x)

        if out is None:
            out = self._func(x)
        else:
            out[...] = self._func(x)

        if scale is not None:
            out *= scale
        return out

    def FT(self, y, out=None):
        '''
        Parameters
        ----------
        y : array of shape [nx, ny, nz, 3] or list of arrays size nx, ny and nz
            Mesh of Fourier coordinates at which to evaluate the probe density.
        out : array of shape [nx, ny, nz]
            If provided then computation is performed inplace

        Returns
        -------
        out : array of shape [nx, ny, nz]
            An array equal to FourierTransform(probe)(y)

        '''
        if hasattr(y, 'shape'):
            y_start = y[(0,) * (y.ndim - 1)]
            y_end = y[(-1,) * (y.ndim - 1)]
            y = [linspace(y_start[i], y_end[i], y.shape[i], endpoint=True)
                  for i in range(3)]
        x = fromFreq(y)
        ft = getDFT(x, y)[0]
        tmp = ft(self(x, out=out))
        if out is None:
            out = tmp
        else:
            out[...] = tmp
        return out


class BesselProbe(probeFunction):
    '''
    Probe function given by a radially scaled Bessel function of the first kind.

    Parameters
    ----------
    r : float
        Width of probe at the surface of the sample. More specifically, the smallest
        0 of the probe.

    Attributes
    ----------
    __call__ : method(x, out=None, scale=None)
        If X = sqrt(x[...,0]**2+x[...,1]**2)/r then returns J_1(X)/X*scale.
        If out is provided then this is computed inplace. If scale is not
        provided then it is assumed to be 1. If x is a list of arrays it is
        converted into a mesh first.

    FT : method(y, out=None)
        If Y = sqrt(y[...,0]**2 + y[...,1]**2)*r then returns an indicator
        function on the disc Y < 1, y[2]==0. Again, if out is provided then
        computation is inplace. If y is a list of arrays then it is converted
        into a mesh first.
    '''

    def __init__(self, r):
        probeFunction.__init__(self)
        self.r = r
        self._r = r / 3.83170597020751

    def __call__(self, x, out=None, scale=None):
        '''
        Parameters
        ----------
        x : array of shape [nx, ny, nz, 3] or list of arrays size nx, ny and nz
            Mesh points at which to evaluate the probe density.
            As a plotting utility, if a lower dimensional mesh is provided then
            the remaining coordinates are assumed to be 0 and so only the
            respective 1D/2D slice of the probe is returned.
        out : array of shape [nx, ny, nz]
            If provided then computation is performed inplace
        scale : array of shape [nx, ny, nz]
            If provided then the probe density is scaled by this before being
            returned.

        Returns
        -------
        out : array of shape [nx, ny, nz]
            An array equal to probe(x)*scale. If ny=0 or nz=0 then array is of
            smaller dimension.

        '''
        if not hasattr(x, 'shape'):
            x = toMesh(x)
        scale = ones(1, dtype=x.dtype) if scale is None else scale
        if out is None:
            out = empty(x.shape[:-1], dtype=scale.dtype)
        if x.shape[-1] == 1 or x.ndim == 1:
            x = maximum(1e-16, abs(x)).reshape(-1)
            out[...] = jv(1, x) / x * scale
        elif x.shape[-1] == 2:
            x = maximum(1e-16, sqrt(abs(x * x).sum(-1) / self._r ** 2))
            out[...] = jv(1, x) / x * scale
        else:
            d = abs(x[1, 1, 0, :2] - x[0, 0, 0, :2])
            h = d.min() / 10
            s = ((d[0] * x.shape[0]) ** 2 + (d[1] * x.shape[1]) ** 2) ** .5

            fine_grid = arange(h / 2, s / self._r + h, h)
            j = jv(1, fine_grid) / fine_grid

            _bess(x.reshape(-1, 3), 1 / self._r, 1 / h, j, scale.reshape(-1), out.reshape(-1))
        return out

    def FT(self, y, out=None):
        '''
        Parameters
        ----------
        y : array of shape [nx, ny, nz, 3] or list of arrays size nx, ny and nz
            Mesh of Fourier coordinates at which to evaluate the probe density.
            As a plotting utility, if a lower dimensional mesh is provided then
            the remaining coordinates are assumed to be 0 and so only the
            respective 1D/2D slice of the probe is returned.
        out : array of shape [nx, ny, nz]
            If provided then computation is performed inplace

        Returns
        -------
        out : array of shape [nx, ny, nz]
            An array equal to FourierTransform(probe)(y). If ny=0 or nz=0 then
            array is of smaller dimension.

        '''
        if not hasattr(y, 'shape'):
            y = toMesh(y)
        r = self._r
        if y.shape[-1] == 1 or y.ndim == 1:
            y = (y * r).reshape(-1)
            y[abs(y) > 1] = 1
            if out is None:
                out = (2 * r) * sqrt(1 - y * y)
            else:
                out[...] = (2 * r) * sqrt(1 - y * y)
        else:
            if y.shape[-1] == 3:
                dy2 = []
                for i in range(y.ndim - 1):
                    tmp = tuple(0 if j != i else 1 for j in range(y.ndim - 1)) + (2,)
                    try:
                        dy2.append(abs(y[tmp] - y[..., 2].item(0)))
                    except Exception:
                        dy2.append(1)
                eps = max(1e-16, max(dy2) * .5)
                if out is None:
                    out = empty(y.shape[:3], dtype=y.dtype)

                _bessFT(
                    y.reshape(-1, 3), 1 / r ** 2, 2 * pi * r ** 2, eps, out.reshape(-1))

            else:
                if out is None:
                    out = (2 * pi * r ** 2) * (abs(y * y).sum(-1) <= 1 / r ** 2)
                else:
                    out[...] = (2 * pi * r ** 2) * (abs(y * y).sum(-1) <= 1 / r ** 2)
        return out


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _bess(X, R, H, J, scale, out):
    if scale.size == 1:
        for i in numba.prange(X.shape[0]):
            rad = c_sqrt(X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]) * R
            ind = int(rad * H)
            if ind < J.size:
                out[i] = J[ind]
            else:
                out[i] = 0
    else:
        for i in numba.prange(X.shape[0]):
            rad = c_sqrt(X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]) * R
            ind = int(rad * H)
            if ind < J.size:
                out[i] = scale[i] * J[ind]
            else:
                out[i] = 0


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _bessFT(X, R, s, eps, out):
    for i in numba.prange(X.shape[0]):
        rad = X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]
        if rad > R or abs(X[i, 2]) > eps:
            out[i] = 0
        else:
            out[i] = s
