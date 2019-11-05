# -*- coding: utf-8 -*-
# Copyright 2017-2019 The diffsims developers
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
from numpy import unique, require, array, arange, ones, pi, zeros, empty, \
    ascontiguousarray, sqrt, random, isscalar
from diffsims.utils.generic_utils import getGrid

"""Utils for converting lists of atoms to a discretised volume

"""

from psutil import virtual_memory
import numpy as np
from scipy.special import erf
import numba
from .generic_utils import _CUDA, cuda
c_abs = abs
from math import (sqrt as c_sqrt, exp as c_exp, erf as c_erf, ceil, cos as c_cos,
                  sin as c_sin, floor)
LOG0 = -20  # TODO: in prismatic this is a user parameter
# TODO: decide these precisions
FTYPE, CTYPE = 'f8', 'c16'
c_FTYPE = numba.float64 if True else numba.float32


##########
# Look-up table of atoms
##########
def getA(Z, returnFunc=True):
    '''
    This function returns an approximation of the atom
    with atomic number Z using a list of Gaussians.

    Parameters
    ----------
    Z : int
        Atomic number of atom
    returnFunc: bool (default=True)
        If True then returns functions for real/reciprocal space discretisation
        else returns the vectorial representation of the approximating Gaussians.

    Returns
    -------
    obj1, obj2 : depends on value of returnFunc
        Continuous atom is represented by: y\mapsto sum_i a[i]*exp(-b[i]*|y|^2)


    This data table is from 'Robust Parameterization of
    Elastic and Absorptive Electron Atomic Scattering
    Factors' by L.-M. Peng, G. Ren, S. L. Dudarev and
    M. J. Whelan, 1996
    '''
    if isscalar(Z):
        Z = array([Z])
    if Z.dtype.char not in 'US':
        Z = Z.astype(int)
    Z = np.unique(Z)
    if Z.size > 1:
        raise ValueError('Only 1 atom can generated at once')
    else:
        Z = Z[0]
    if Z == 0:
        # Mimics a Dirac spike
        a = [1] * 5 + [.1] * 5
    elif Z in (1, 'H'):
        a = [0.0349, 0.1201, 0.1970, 0.0573, 0.1195,
             0.5347, 3.5867, 12.3471, 18.9525, 38.6269]
    elif Z in (2, 'He'):
        a = [0.0317, 0.0838, 0.1526, 0.1334, 0.0164,
             0.2507, 1.4751, 4.4938, 12.6646, 31.1653]
    elif Z in (3, 'Li'):
        a = [0.0750, 0.2249, 0.5548, 1.4954, 0.9354,
             0.3864, 2.9383, 15.3829, 53.5545, 138.7337]
    elif Z in (4, 'Be'):
        a = [0.0780, 0.2210, 0.6740, 1.3867, 0.6925,
             0.3131, 2.2381, 10.1517, 30.9061, 78.3273]
    elif Z in (5, 'B'):
        a = [0.0909, 0.2551, 0.7738, 1.2136, 0.4606,
             0.2995, 2.1155, 8.3816, 24.1292, 63.1314]
    elif Z in (6, 'C'):
        a = [0.0893, 0.2563, 0.7570, 1.0487, 0.3575,
             0.2465, 1.7100, 6.4094, 18.6113, 50.2523]
    elif Z in (7, 'N'):
        a = [0.1022, 0.3219, 0.7982, 0.8197, 0.1715,
             0.2451, 1.7481, 6.1925, 17.3894, 48.1431]
    elif Z in (8, 'O'):
        a = [0.0974, 0.2921, 0.6910, 0.6990, 0.2039,
             0.2067, 1.3815, 4.6943, 12.7105, 32.4726]
    elif Z in (9, 'F'):
        a = [0.1083, 0.3175, 0.6487, 0.5846, 0.1421,
             0.2057, 1.3439, 4.2788, 11.3932, 28.7881]
    elif Z in (10, 'Ne'):
        a = [0.1269, 0.3535, 0.5582, 0.4674, 0.1460,
             0.2200, 1.3779, 4.0203, 9.4934, 23.1278]
    elif Z in (11, 'Na'):
        a = [0.2142, 0.6853, 0.7692, 1.6589, 1.4482,
             0.3334, 2.3446, 10.0830, 48.3037, 138.2700]
    elif Z in (12, 'Mg'):
        a = [0.2314, 0.6866, 0.9677, 2.1882, 1.1339,
             0.3278, 2.2720, 10.9241, 39.2898, 101.9748]
    elif Z in (13, 'Al'):
        a = [0.2390, 0.6573, 1.2011, 2.5586, 1.2312,
             0.3138, 2.1063, 10.4163, 34.4552, 98.5344]
    elif Z in (14, 'Si'):
        a = [0.2519, 0.6372, 1.3795, 2.5082, 1.0500,
             0.3075, 2.0174, 9.6746, 29.3744, 80.4732]
    elif Z in (15, 'P'):
        a = [0.2548, 0.6106, 1.4541, 2.3204, 0.8477,
             0.2908, 1.8740, 8.5176, 24.3434, 63.2996]
    elif Z in (16, 'S'):
        a = [0.2497, 0.5628, 1.3899, 2.1865, 0.7715,
             0.2681, 1.6711, 7.0267, 19.5377, 50.3888]
    elif Z in (17, 'Cl'):
        a = [0.2443, 0.5397, 1.3919, 2.0197, 0.6621,
             0.2468, 1.5242, 6.1537, 16.6687, 42.3086]
    elif Z in (18, 'Ar'):
        a = [0.2385, 0.5017, 1.3428, 1.8899, 0.6079,
             0.2289, 1.3694, 5.2561, 14.0928, 35.5361]
    elif Z in (19, 'K'):
        a = [0.4115, -1.4031, 2.2784, 2.6742, 2.2162,
             0.3703, 3.3874, 13.1029, 68.9592, 194.4329]
    elif Z in (20, 'Ca'):
        a = [0.4054, 1.3880, 2.1602, 3.7532, 2.2063,
             0.3499, 3.0991, 11.9608, 53.9353, 142.3892]

    a, b = np.array(a[:5], dtype=FTYPE), np.array(a[5:], dtype=FTYPE)
    b /= (4 * np.pi) ** 2  # Weird scaling in initial paper

    def myAtom(x):
        dim = x.shape[-1]
        x = abs(x * x).sum(-1)
        y = 0
        for i in range(5):
            y += (a[i] / (4 * np.pi * b[i]) ** (dim / 2)) * np.exp(-x / (4 * b[i]))

        return y

    def myAtomFT(x):
        x = abs(x * x).sum(-1)
        y = 0
        for i in range(5):
            y += a[i] * np.exp(-b[i] * x)
        return y

    if returnFunc:
        return myAtom, myAtomFT
    else:
        return a, b


##########
# Evaluate single atom intensities
##########
def __atom(a, b, x, dx, pointwise=False):
    '''
    Compute atomic intensities.

    Parameters
    ----------
    a, b : 1D arrays of same length
        Continuous atom is represented by: y\mapsto sum_i a[i]*exp(-b[i]*|y|^2)
    x : array or array-like
        Coordinate(s) at which to evaluate the atom intensity. len(x) should be the
        spatial dimension. Each x[i] should have the same shape.
    dx : array-like
        Physical dimensions of a single pixel. Depending on the <pointwise> flag,
        returned value approximates the intensity on the box [x-dx/2, x+dx/2]
    pointwise: bool (default=False)
        If true, the evaluation is pointwise. If false, returns average intensity
        over box defined by <dx>.

    Returns
    -------
    out : array
        Atomic intensities evaluated as detailed above. out.shape = x[i].shape
        for each i.
    '''
    if pointwise:
        n = sum(X * X for X in x)
        return sum(a[i] * np.exp(-b[i] * n) for i in range(a.size))
    else:
        Sum = 0
        for i in range(a.size):
            B = b.item(i) ** .5  # force to scalar
            prod = 1
            for j in range(len(x)):
                if dx[j] == 0:
                    prod *= 2 / B
                else:
                    prod *= (erf(B * (x[j] + dx[j] / 2))
                             -erf(B * (x[j] - dx[j] / 2))) * np.pi ** .5 / (2 * dx[j] * B)
            Sum += a[i] * prod
        return Sum


def __precomp(a, b, dx, pointwise=False):
    '''
    Helper for computing atomic intensities. This function precomputes
    values so that __atom(a,b,x,dx,True) = __atom_pw_cpu(*x,*__precomp(a,b,dx,True))
    etc.

    Parameters
    ----------
    a, b : 1D arrays of same length
        Continuous atom is represented by: y\mapsto sum_i a[i]*exp(-b[i]*|y|^2)
    dx : array-like
        Physical dimensions of a single pixel. Depending on the <pointwise> flag,
        returned value approximates the intensity on the box [x-dx/2, x+dx/2]
    pointwise: bool (default=False)
        If true, the evaluation is pointwise. If false, returns average intensity
        over box defined by <dx>.

    Returns
    -------
    precomp : array
        Radial profile of atom intensities
    params : array
        Grid spacings to convert real positions to indices
    Rmax : float
        The cut-off radius for this atom
    '''
    if pointwise:
        f = lambda x: sum(a[i] * c_exp(-b[i] * x ** 2) for i in range(a.size))
        Rmax = 1
        while f(Rmax) > c_exp(LOG0) * f(0):
            Rmax *= 1.1
        h = max(Rmax / 200, max(dx) / 10)
        pms = np.array([Rmax ** 2, 1 / h])
        precomp = np.array([f(x) for x in np.arange(0, Rmax + 2 * h, h)], dtype=FTYPE)
    else:

        def f(i, j, x):
            A = a[i] ** (1 / 3)  # factor spread evenly over 3 dimensions
            B = b[i] ** .5
            if dx[j] == 0:
                return A * 2 / B
            return A * (c_erf(B * (x + dx[j] / 2))
                        -c_erf(B * (x - dx[j] / 2))) / (2 * dx[j] * B) * np.pi ** .5

        h = [D / 10 for D in dx]
        Rmax = np.ones([a.size, 3], dtype=FTYPE)
        L = 1
        for i in range(a.size):
            for j in range(3):
                if dx[j] == 0:
                    Rmax[i, j] = 1e5
                    continue
                while f(i, j, Rmax[i, j]) > c_exp(LOG0) * f(i, j, 0):
                    Rmax[i, j] *= 1.1
                    L = max(L, Rmax[i, j] / h[j] + 2)
        L = min(200, int(ceil(L)))
        precomp, grid = np.zeros([a.size, 3, L], dtype=FTYPE), np.arange(L)
        for i in range(a.size):
            for j in range(3):
                h[j] = Rmax[:, j].max() / (L - 2)
                precomp[i, j] = np.array([f(i, j, x * h[j]) for x in grid], dtype=FTYPE)
        pms = np.array([Rmax.max(0), 1 / np.array(h)], dtype=FTYPE)
        Rmax = Rmax.max(0).min()
    return precomp, pms, Rmax


@numba.jit(fastmath=True, nopython=True)
def __atom_pw_cpu(x0, x1, x2, pc, h):
    n = x0 * x0 + x1 * x1 + x2 * x2
    if n >= h[0]:
        return 0
    else:
        n = h[1] * c_sqrt(n)
        i = int(n)
        n -= i
        return (1 - n) * pc[i] + n * pc[i + 1]


@numba.jit(fastmath=True, nopython=True)
def __atom_av_cpu(x0, x1, x2, pc, h):
    x0, x1, x2 = c_abs(x0), c_abs(x1), c_abs(x2)
    if x0 > h[0, 0] or x1 > h[0, 1] or x2 > h[0, 2]:
        return 0

    x0, x1, x2 = h[1, 0] * x0, h[1, 1] * x1, h[1, 2] * x2
    i0, i1, i2 = int(x0), int(x1), int(x2)
    x0, x1, x2 = x0 - i0, x1 - i1, x2 - i2
    X0, X1, X2 = 1 - x0, 1 - x1, 1 - x2
    s = 0
    for i in range(pc.shape[0]):
        v = X0 * pc[i, 0, i0] + x0 * pc[i, 0, i0 + 1]
        v *= X1 * pc[i, 1, i1] + x1 * pc[i, 1, i1 + 1]
        v *= X2 * pc[i, 2, i2] + x2 * pc[i, 2, i2 + 1]

        s += v

    return s


if _CUDA:

    @cuda.jit(device=True, inline=True)
    def __atom_pw_gpu(x0, x1, x2, pc, h):
        n = x0 * x0 + x1 * x1 + x2 * x2
        if n >= h[0]:
            return 0
        else:
            n = h[1] * c_sqrt(n)
            i = int(n)
            n -= i
            return (1 - n) * pc[i] + n * pc[i + 1]

    @cuda.jit(device=True, inline=True)
    def __atom_av_gpu(x0, x1, x2, pc, h):
        x0, x1, x2 = c_abs(x0), c_abs(x1), c_abs(x2)
        if x0 > h[0, 0] or x1 > h[0, 1] or x2 > h[0, 2]:
            return 0

        x0, x1, x2 = h[1, 0] * x0, h[1, 1] * x1, h[1, 2] * x2
        i0, i1, i2 = int(x0), int(x1), int(x2)
        x0, x1, x2 = x0 - i0, x1 - i1, x2 - i2
        X0, X1, X2 = 1 - x0, 1 - x1, 1 - x2
        s = 0
        for i in range(pc.shape[0]):
            v = X0 * pc[i, 0, i0] + x0 * pc[i, 0, i0 + 1]
            v *= X1 * pc[i, 1, i1] + x1 * pc[i, 1, i1 + 1]
            v *= X2 * pc[i, 2, i2] + x2 * pc[i, 2, i2 + 1]

            s += v

        return s


##########
# Binning list of atoms into a grid for efficiency
##########
@numba.jit(cache=True)
def __countbins(x0, x1, x2, loc, r, s, Len, MAX):
    for j0 in range(loc.shape[0]):
        bin0 = int((loc[j0, 0] - x0) / r[0])
        bin1 = int((loc[j0, 1] - x1) / r[1])
        bin2 = int((loc[j0, 2] - x2) / r[2])
        for i in range(max(0, bin0 - s), min(Len.shape[0], bin0 + s + 1)):
            for j in range(max(0, bin1 - s), min(Len.shape[1], bin1 + s + 1)):
                for k in range(max(0, bin2 - s), min(Len.shape[2], bin2 + s + 1)):
                    Len[i, j, k] += 1
                    if Len[i, j, k] == MAX:
                        return


@numba.jit(cache=True)
def __rebin(x0, x1, x2, loc, sublist, r, s, Len):
    for j0 in range(loc.shape[0]):
        bin0 = int((loc[j0, 0] - x0) / r[0])
        bin1 = int((loc[j0, 1] - x1) / r[1])
        bin2 = int((loc[j0, 2] - x2) / r[2])
        for i in range(max(0, bin0 - s), min(Len.shape[0], bin0 + s + 1)):
            for j in range(max(0, bin1 - s), min(Len.shape[1], bin1 + s + 1)):
                for k in range(max(0, bin2 - s), min(Len.shape[2], bin2 + s + 1)):
                    sublist[i, j, k, Len[i, j, k]] = j0
                    Len[i, j, k] += 1

    for b0 in range(sublist.shape[0]):
        for b1 in range(sublist.shape[1]):
            for b2 in range(sublist.shape[2]):
                j0 = Len[b0, b1, b2]
                if j0 < sublist.shape[3]:
                    sublist[b0, b1, b2, j0] = -1


def rebin(x, loc, r, k, mem):
    '''
    x is the grid for the discretisation, used for bounding box
    loc is the locations of each Atom
    '''
    assert len(x) == 3, 'x must represent a 3 dimensional grid'
    mem = virtual_memory().available if mem is None else mem

    if np.isscalar(r):
        r = np.array([r, r, r], dtype='f4')
    else:
        r = r.copy()
    xmin = np.array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=x[0].dtype)
    nbins = [int(ceil((x[i].item(-1) - x[i].item(0)) / r[i])) + 1
             for i in range(3)]
    if np.prod(nbins) * 32 * 10 > mem:
        raise MemoryError
    Len = np.zeros(nbins, dtype='i4')
    L = int(mem / (Len.size * Len.itemsize)) + 2
    __countbins(xmin[0], xmin[1], xmin[2], loc, r, k, Len, L)

    L = Len.max()
    if Len.size * Len.itemsize * L > mem:
        raise MemoryError
    subList = np.zeros(nbins + [L], dtype='i4')
    Len.fill(0)
    __rebin(xmin[0], xmin[1], xmin[2], loc, subList, r, k, Len)

    return subList


def doBinning(x, loc, Rmax, d, GPU):
    # Bin the atoms
    k = int(Rmax / max(d)) + 1
    # TODO: consider desired memory usage more.
    try:
        if not (GPU and _CUDA): raise Exception
        cuda.current_context().deallocations.clear()
        mem = cuda.current_context().get_memory_info()[0]  # amount of free memory
    except Exception:
        mem = virtual_memory().total / 10  # can use swap space if needed

    while k > 0:
        # Proposed coarse grid size
        r = np.array([2e5 if D == 0 else max(Rmax / k, D) for D in d], dtype='f4')

        subList = None
        try:
            subList = rebin(x, loc, r, k, mem=.25 * mem)
            if subList.size * subList.itemsize > .25 * mem:
                subList = None  # treat like memory error
        except MemoryError:
            pass

        if subList is None and k == 1:
            # Memory error at smallest k is a failure
            return None, r, mem
        elif subList is None:
            k -= 1  # No extra information
        else:
            return subList, r, mem


##########
# Discretise whole crystal
##########
@numba.jit(parallel=True, fastmath=True, nopython=True)
def __density3D_pw_cpu(x0, x1, x2, xmin,
                  loc, sublist, r, a, d, B, precomp, h, out):
    X0 = x0
    bin0 = int(floor((X0 - xmin[0]) / r[0]))
    if bin0 >= sublist.shape[0]:
        return
    for i1 in numba.prange(x1.size):
        X1 = x1[i1]
        bin1 = int(floor((X1 - xmin[1]) / r[1]))
        if bin1 >= sublist.shape[1]:
            continue
        for i2 in range(x2.size):
            X2 = x2[i2]
            bin2 = int(floor((X2 - xmin[2]) / r[2]))
            if bin2 >= sublist.shape[2]:
                continue

            Sum = 0
            for bb in range(sublist.shape[3]):
                j0 = sublist[bin0, bin1, bin2, bb]
                if j0 < 0:
                    break
                Y0 = loc[j0, 0] - X0
                Y1 = loc[j0, 1] - X1
                Y2 = loc[j0, 2] - X2
                Sum += __atom_pw_cpu(Y0, Y1, Y2, precomp, h)
            out[i1, i2] = Sum


@numba.jit(parallel=True, fastmath=True, nopython=True)
def __density3D_av_cpu(x0, x1, x2, xmin,
                  loc, sublist, r, a, d, B, precomp, h, out):
    X0 = x0
    bin0 = int(floor((X0 - xmin[0]) / r[0]))
    if bin0 >= sublist.shape[0]:
        return
    for i1 in numba.prange(x1.size):
        X1 = x1[i1]
        bin1 = int(floor((X1 - xmin[1]) / r[1]))
        if bin1 >= sublist.shape[1]:
            continue
        for i2 in range(x2.size):
            X2 = x2[i2]
            bin2 = int(floor((X2 - xmin[2]) / r[2]))
            if bin2 >= sublist.shape[2]:
                continue

            Sum = 0
            for bb in range(sublist.shape[3]):
                j0 = sublist[bin0, bin1, bin2, bb]
                if j0 < 0:
                    break
                Y0 = loc[j0, 0] - X0
                Y1 = loc[j0, 1] - X1
                Y2 = loc[j0, 2] - X2
                Sum += __atom_av_cpu(Y0, Y1, Y2, precomp, h)
            out[i1, i2] = Sum


@numba.jit(parallel=True, fastmath=True, nopython=True)
def __FT3D_pw_cpu(x0, x1, x2, loc, a, b, d, B, precomp, h, out):
    X0 = x0
    for i1 in numba.prange(x1.size):
        X1 = x1[i1]
        for i2 in range(x2.size):
            X2 = x2[i2]
            scale = __atom_pw_cpu(X0, X1, X2, precomp, h)

            IP, Sum = 0, complex(0, 0)
            for j0 in range(loc.shape[0]):
                IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
                Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

            out[i1, i2] = Sum


@numba.jit(parallel=True, fastmath=True, nopython=True)
def __FT3D_av_cpu(x0, x1, x2, loc, a, b, d, B, precomp, h, out):
    X0 = x0
    for i1 in numba.prange(x1.size):
        X1 = x1[i1]
        for i2 in range(x2.size):
            X2 = x2[i2]
            scale = __atom_av_cpu(X0, X1, X2, precomp, h)

            IP, Sum = 0, complex(0, 0)
            for j0 in range(loc.shape[0]):
                IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
                Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

            out[i1, i2] = Sum


if _CUDA:

    @cuda.jit
    def __density3D_pw_gpu(x0, x1, x2, xmin,
                      loc, sublist, r, a, d, B, precomp, h, out):

        i0, i1, i2 = cuda.grid(3)
        if i0 >= x0.size or i1 >= x1.size or i2 >= x2.size:
            return
        X0, X1, X2 = x0[i0], x1[i1], x2[i2]
        Y0, Y1, Y2 = 0, 0, 0

        bin0 = int((X0 - xmin[0]) / r[0])
        bin1 = int((X1 - xmin[1]) / r[1])
        bin2 = int((X2 - xmin[2]) / r[2])
        if bin0 >= sublist.shape[0] or bin1 >= sublist.shape[1] or bin2 >= sublist.shape[2]:
            return

        Sum = 0
        for bb in range(sublist.shape[3]):
            j0 = sublist[bin0, bin1, bin2, bb]
            if j0 < 0:
                break
            Y0 = loc[j0, 0] - X0
            Y1 = loc[j0, 1] - X1
            Y2 = loc[j0, 2] - X2
            Sum += __atom_pw_gpu(Y0, Y1, Y2, precomp, h)
        out[i0, i1, i2] = Sum

    @cuda.jit
    def __density3D_av_gpu(x0, x1, x2, xmin,
                      loc, sublist, r, a, d, B, precomp, h, out):

        i0, i1, i2 = cuda.grid(3)
        if i0 >= x0.size or i1 >= x1.size or i2 >= x2.size:
            return
        X0, X1, X2 = x0[i0], x1[i1], x2[i2]
        Y0, Y1, Y2 = 0, 0, 0

        bin0 = int((X0 - xmin[0]) / r[0])
        bin1 = int((X1 - xmin[1]) / r[1])
        bin2 = int((X2 - xmin[2]) / r[2])
        if bin0 >= sublist.shape[0] or bin1 >= sublist.shape[1] or bin2 >= sublist.shape[2]:
            return

        Sum = 0
        for bb in range(sublist.shape[3]):
            j0 = sublist[bin0, bin1, bin2, bb]
            if j0 < 0:
                break
            Y0 = loc[j0, 0] - X0
            Y1 = loc[j0, 1] - X1
            Y2 = loc[j0, 2] - X2
            Sum += __atom_av_gpu(Y0, Y1, Y2, precomp, h)
        out[i0, i1, i2] = Sum

    @cuda.jit
    def __FT3D_pw_gpu(x0, x1, x2, loc, a, b, d, B, precomp, h, out):
        i0, i1, i2 = cuda.grid(3)
        if i0 >= x0.size or i1 >= x1.size or i2 >= x2.size:
            return
        X0, X1, X2 = x0[i0], x1[i1], x2[i2]
        scale = __atom_pw_gpu(X0, X1, X2, precomp, h)

        IP, Sum = 0, complex(0, 0)
        for j0 in range(loc.shape[0]):
            IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
            Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

        out[i0, i1, i2] = Sum

    @cuda.jit
    def __FT3D_av_gpu(x0, x1, x2, loc, a, b, d, B, precomp, h, out):
        i0, i1, i2 = cuda.grid(3)
        if i0 >= x0.size or i1 >= x1.size or i2 >= x2.size:
            return
        X0, X1, X2 = x0[i0], x1[i1], x2[i2]

        scale = __atom_av_gpu(X0, X1, X2, precomp, h)

        IP, Sum = 0, complex(0, 0)
        for j0 in range(loc.shape[0]):
            IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
            Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

        out[i0, i1, i2] = Sum


def _precomp_atom(x, a, b, d, pw):

    if pw:
        n_zeros = sum(1 for D in d if D == 0)
        f = lambda x: sum(a[i] * c_exp(-b[i] * x ** 2) * (pi / b[i]) ** (n_zeros / 2)
                          for i in range(a.size))
        Rmax = 1
        while f(Rmax) > c_exp(LOG0) * f(0):
            Rmax *= 1.1
        h = max(Rmax / 200, max(d) / 10)
        pms = array([Rmax ** 2, 1 / h], dtype=FTYPE)
        precomp = array([f(x) for x in arange(0, Rmax + 2 * h, h)], dtype=FTYPE)

    else:

        def f(i, j, x):
            A = a[i] ** (1 / 3)  # factor spread evenly over 3 dimensions
            B = b[i] ** .5
            if d[j] == 0:
                return A * 2 / B
            return A * (c_erf(B * (x + d[j] / 2))
                        -c_erf(B * (x - d[j] / 2))) / (2 * d[j] * B) * pi ** .5

        h = [D / 10 for D in d]
        Rmax = ones([a.size, 3], dtype=FTYPE)
        L = 1
        for i in range(a.size):
            for j in range(3):
                if d[j] == 0:
                    Rmax[i, j] = 1e5
                    continue
                while f(i, j, Rmax[i, j]) > c_exp(LOG0) * f(i, j, 0):
                    Rmax[i, j] *= 1.1
                    L = max(L, Rmax[i, j] / h[j] + 2)
        L = min(200, int(ceil(L)))
        precomp, grid = zeros([a.size, 3, L], dtype=FTYPE), arange(L)
        for i in range(a.size):
            for j in range(3):
                h[j] = Rmax[:, j].max() / (L - 2)
                precomp[i, j] = array([f(i, j, x * h[j]) for x in grid], dtype=FTYPE)
        pms = array([Rmax.max(0), 1 / array(h)], dtype=FTYPE)
        Rmax = Rmax.max(0).min()

    return precomp, pms, Rmax


def _bin_atoms(x, loc, Rmax, d, GPU):
    # Bin the atoms
    k = int(Rmax / max(d)) + 1
    try:
        if not (GPU and _CUDA): raise Exception
        cuda.current_context().deallocations.clear()
        mem = cuda.current_context().get_memory_info()[0]
    except Exception:
        mem = 1e12

    while k > 0:
        r = array([2e5 if D == 0 else max(Rmax / k, D) for D in d], dtype='f4')
        subList = None
        try:
            subList = rebin(x, loc, r, k, mem=.25 * mem)
            if subList.size * subList.itemsize > .25 * mem:
                subList = None  # treat like memory error
        except MemoryError:
            pass

        if subList is None and k == 1:
#                     raise MemoryError('List of atoms is too large to be stored on device')
            pass  # Memory error at smallest k
        elif subList is None:
            k -= 1; continue  # No extra information
        else:
            break

        return None, r, mem

    return subList, r, mem


def getDiscretisation(loc, Z, x, pointwise=False, FT=False):
    dim = loc.shape[-1]
    if Z.dtype.char not in 'US':
        Z = Z.astype(int)
    z = unique(Z)
    if z.size > 1:
        return sum(getDiscretisation(loc[Z == zz], zz, x, pointwise, FT) for zz in z)

    GPU = bool(_CUDA)

    Z = z[0]
    a, b = getA(Z, False)
    loc = require(loc.reshape(-1, dim), requirements='AC')
    x = x if len(x) == dim else list(x) + [array([0])]
    d = array([abs(X.item(1) - X.item(0)) if X.size > 1 else 0 for X in x])
    x = [X if X.size > 1 else array([0]) for X in x]

    if FT:
        out = empty([X.size for X in x], dtype=CTYPE)
        if out.size == 0:
            return 0 * out

        precomp, pms, mem = _precomp_atom(x, a, b, d, pointwise)

        notComputed = True
        if GPU:
            try:
                func = __FT3D_pw_gpu if pointwise else __FT3D_av_gpu
                if out.size * out.itemsize < .8 * mem:
                    grid, tpb = getGrid(out.shape[:dim])
                    func[grid, tpb](x[0], x[1], x[2], loc, a, b,
                                    d, sqrt(b), precomp, pms, out)
                else:
                    grid, tpb = getGrid(out.shape[:dim - 1])
                    lloc = cuda.to_device(loc, stream=cuda.stream())
                    n, buf = len(x[2]), ascontiguousarray(out[..., 0])
                    for i in range(n):
                        # TODO: check this one also works
                        func[grid, tpb](x[0], x[1], array([x[2][i]]), lloc, a, b,
                                        d, sqrt(b), precomp, pms, buf)
                        out[..., i] = buf
                notComputed = False
            except Exception:
                pass

        if notComputed:
            func = __FT3D_pw_cpu if pointwise else __FT3D_av_cpu
            for i in range(x[0].size):
                func(x[0][i], x[1], x[2], loc, a, b, d, sqrt(b),
                     precomp, pms, out[i])
    else:
        B = (1 / (4 * b)).astype(FTYPE)
        A = (a * (B / pi) ** (dim / 2)).astype(FTYPE)

        out = empty([X.size for X in x], dtype=FTYPE)
        if out.size == 0:
            return 0 * out

        # Precompute extra variables
        xmin = array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=FTYPE)

        precomp, pms, Rmax = _precomp_atom(x, A, B, d, pointwise)
        subList, r, mem = doBinning(x, loc, Rmax, d, GPU)

        if subList is None:
            # Volume is too large for memory, halve it in each dimension
            # TODO: assumption of 3D
            Slice = [None] * 3
            for i in range(2):
                Slice[0] = slice(out.shape[0] // 2, None) if i else slice(out.shape[0] // 2)
                for j in range(2):
                    Slice[1] = slice(out.shape[1] // 2, None) if j else slice(out.shape[1] // 2)
                    for k in range(2):
                        Slice[2] = slice(None) if k else slice(0)
                        out[tuple(Slice)] = getDiscretisation(loc, Z,
                                                              [x[t][Slice[t]] for t in range(3)],
                                                              pointwise, FT)
            return out

        elif subList.size == 0:
            # No atoms in this part of the volume
            out.fill(0)
            return out

        notComputed = True
        if GPU:
            try:
                n = max(1, int(ceil(min(x[0].size, mem / (2 * out[0].size * out.itemsize)) - 1e-5)))
                grid, tpb = getGrid((n,) + out.shape[-2:], 8)
                func = __density3D_pw_gpu if pointwise else __density3D_av_gpu

                ssubList, lloc = cuda.to_device(subList, stream=cuda.stream()), cuda.to_device(loc, stream=cuda.stream())
                i = 0
                for j in random.permutation(int(ceil(x[0].size / n))):
                    bins = j * n, (j + 1) * n
                    func[grid, tpb](
                        x[0][bins[0]:bins[1] + 1], x[1], x[2], xmin, lloc, ssubList,
                        r, A, d, sqrt(B), precomp, pms, out[bins[0]:bins[1] + 1])
                    i += 1
                notComputed = False
            except Exception:
                pass

        if notComputed:
            func = __density3D_pw_cpu if pointwise else __density3D_av_cpu
            for i in range(x[0].size):
                func(x[0][i], x[1], x[2], xmin, loc, subList,
                     r, A, d, sqrt(B), precomp, pms, out[i])

    return out

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ################