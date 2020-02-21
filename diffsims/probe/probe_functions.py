# -*- coding: utf-8 -*-
# Copyright 2017-2020 The diffsims developers
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

'''
Created on 5 Nov 2019

@author: Rob Tovey
'''
import numba
from math import sqrt as c_sqrt
from numpy import empty, maximum, sqrt, arange, pi, linspace, ones
from scipy.special import jv
import numpy as np
from scipy import constants as pc
from numpy.fft import fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt
import h5py

from diffsims.utils.atomic_diffraction_generator_support.fourier_transform import get_DFT, from_recip
from diffsims.utils.atomic_diffraction_generator_support.generic_utils import to_mesh
from diffsims.probe.probe_utils import evaluate_aberration_function


class ProbeFunction:
    '''
    Object representing a probe function.

    Parameters
    ----------
    func : function
        Function which takes in an array, `r`, of shape `[nx, ny, nz, 3]` and
        returns an array of shape `[nx, ny, nz]`. `r[...,0]` corresponds to the
        `x` coordinate, `r[..., 1]` to `y` etc. If not provided (or `None`) then the
        `__call__` and `FT` methods must be overrided.

    Attributes
    ----------
    __call__ : method(x, out=None, scale=None)
        Returns `func(x)*scale`. If `out` is provided then it is used as
        preallocated storage. If `scale` is not provided then it is assumed
        to be 1. If `x` is a list of arrays it is converted into a mesh first.

    FT : method(y, out=None)
        Returns the Fourier transform of func on the mesh `y`. Again, if `out` is
        provided then computation is `inplace`. If `y` is a list of arrays then
        it is converted into a mesh first. If this function is not overridden
        then an approximation is made using `func` and the `fft`.

    '''

    def __init__(self, func=None):
        self._func = func

    def __call__(self, x, out=None, scale=None):
        '''
        Parameters
        ----------
        x : `numpy.ndarray`, (nx, ny, nz, 3) or list of arrays of shape [(nx,), (ny,), (nz,)]
            Mesh points at which to evaluate the probe density
        out : `numpy.ndarray`, (nx, ny, nz), optional
            If provided then computation is performed inplace
        scale : `numpy.ndarray`, (nx, ny, nz), optional
            If provided then the probe density is scaled by this before being
            returned.

        Returns
        -------
        out : `numpy.ndarray`, (nx, ny, nz)
            An array equal to `probe(x)*scale`

        '''
        if self._func is None:
            raise NotImplementedError

        if not(hasattr(x, 'shape')):
            x = to_mesh(x)

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
        y : `numpy.ndarray`, (nx, ny, nz, 3) or list of arrays of shape [(nx,), (ny,), (nz,)]
            Mesh of Fourier coordinates at which to evaluate the probe density
        out : `numpy.ndarray`, (nx, ny, nz), optional
            If provided then computation is performed inplace

        Returns
        -------
        out : `numpy.ndarray`, (nx, ny, nz)
            An array equal to `FourierTransform(probe)(y)`

        '''
        if hasattr(y, 'shape'):
            y_start = y[(0,) * (y.ndim - 1)]
            y_end = y[(-1,) * (y.ndim - 1)]
            y = [linspace(y_start[i], y_end[i], y.shape[i], endpoint=True)
                  for i in range(3)]
        x = from_recip(y)
        ft = get_DFT(x, y)[0]
        tmp = ft(self(x, out=out))
        if out is None:
            out = tmp
        else:
            out[...] = tmp
        return out


class BesselProbe(ProbeFunction):
    '''
    Probe function given by a radially scaled Bessel function of the first kind.

    Parameters
    ----------
    r : `float`
        Width of probe at the surface of the sample. More specifically, the smallest
        0 of the probe.

    Attributes
    ----------
    __call__ : method(x, out=None, scale=None)
        If `X = sqrt(x[...,0]**2+x[...,1]**2)/r` then returns `J_1(X)/X*scale`.
        If `out` is provided then this is computed inplace. If `scale` is not
        provided then it is assumed to be 1. If `x` is a list of arrays it is
        converted into a mesh first.

    FT : method(y, out=None)
        If `Y = sqrt(y[...,0]**2 + y[...,1]**2)*r` then returns an indicator
        function on the disc `Y < 1, y[2]==0`. Again, if `out` is provided then
        computation is inplace. If `y` is a list of arrays then it is converted
        into a mesh first.
    '''

    def __init__(self, r):
        ProbeFunction.__init__(self)
        self.r = r
        self._r = r / 3.83170597020751

    def __call__(self, x, out=None, scale=None):
        '''
        Parameters
        ----------
        x : `numpy.ndarray`, (nx, ny, nz, 3) or list of arrays of shape [(nx,), (ny,), (nz,)]
            Mesh points at which to evaluate the probe density.
            As a plotting utility, if a lower dimensional mesh is provided then
            the remaining coordinates are assumed to be 0 and so only the
            respective 1D/2D slice of the probe is returned.
        out : `numpy.ndarray`, (nx, ny, nz), optional
            If provided then computation is performed inplace
        scale : `numpy.ndarray`, (nx, ny, nz), optional
            If provided then the probe density is scaled by this before being
            returned.

        Returns
        -------
        out : `numpy.ndarray`, (nx, ny, nz)
            An array equal to `probe(x)*scale`. If `ny=0` or `nz=0` then array is of
            smaller dimension.

        '''
        if not hasattr(x, 'shape'):
            x = to_mesh(x)
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
        y : `numpy.ndarray`, (nx, ny, nz, 3) or list of arrays of shape [(nx,), (ny,), (nz,)]
            Mesh of Fourier coordinates at which to evaluate the probe density.
            As a plotting utility, if a lower dimensional mesh is provided then
            the remaining coordinates are assumed to be 0 and so only the
            respective 1D/2D slice of the probe is returned.
        out : `numpy.ndarray`, (nx, ny, nz), optional
            If provided then computation is performed inplace

        Returns
        -------
        out : `numpy.ndarray`, (nx, ny, nz)
            An array equal to `FourierTransform(probe)(y)`. If `ny=0` or `nz=0` then
            array is of smaller dimension.

        '''
        if not hasattr(y, 'shape'):
            y = to_mesh(y)
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
                    dy2.append(abs(y[tmp] - y[..., 2].item(0)) if y.shape[-1] == 3 else 1)
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


# Coverage: Numba code does not register when code is run
@numba.jit(nopython=True, parallel=True, fastmath=True)
def _bess(X, R, H, J, scale, out):  # pragma: no cover
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


# Coverage: Numba code does not register when code is run
@numba.jit(nopython=True, parallel=True, fastmath=True)
def _bessFT(X, R, s, eps, out):  # pragma: no cover
    for i in numba.prange(X.shape[0]):
        rad = X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]
        if rad > R or abs(X[i, 2]) > eps:
            out[i] = 0
        else:
            out[i] = s


def define_probe_function(V,alpha, px_cal,array_px, aberr_input, dose = 1, plot_me = False):
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##% this function needs comments
    #
    #% ptycho.varfunctions.define_probe_function: Flag to indicate the function
    #% has been executed
    #
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    #% Set up the lens aberrations:
    #% Lens Aberrations (sorted by radial order):
    #    aberr_input(1) = ptycho.aberr.C1;
    #    aberr_input(2) = ptycho.aberr.C12a;
    #    aberr_input(3) = ptycho.aberr.C12b;
    #    aberr_input(4) = ptycho.aberr.C23a;
    #    aberr_input(5) = ptycho.aberr.C23b;
    #    aberr_input(6) = ptycho.aberr.C21a;
    #    aberr_input(7) = ptycho.aberr.C21b;
    #    aberr_input(8) = ptycho.aberr.C30a;
    #    aberr_input(9) = ptycho.aberr.C34a;
    #    aberr_input(10) = ptycho.aberr.C34b;
    #    aberr_input(11) = ptycho.aberr.C32a;
    #    aberr_input(12) = ptycho.aberr.C32b;
    #
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    l =(pc.h * pc.c) / np.sqrt((pc.e * V)**2  + 2 * pc.e * V * pc.m_e * pc.c**2)  # relativistic wavelength
    #% probe function
    #% chi function
    cen = array_px / 2 # center of array
    #K_max = ((cen * px_cal) - (px_cal/2)) / l # max scattering vector
    K_px =  l / (px_cal * array_px)
    K_max = K_px * array_px
    #print(K_px, K_max)
    Kx = np.linspace(-K_max , K_max, array_px)
    Kx = np.repeat(Kx,array_px,  axis = 0)
    Kx = np.reshape(Kx, (array_px, array_px))
    Ky = np.copy(Kx)
    Kx = Kx.T
    func_aber = FuncAberrUV(Kx,Ky,aberr_input)

    #% transfer function
    func_transfer=np.exp((-1j*2*np.pi/ (l)) * func_aber)

    #% aperture function
    #func_ObjApt = ones(size(ptycho.Kwp));
    #func_ObjApt( ptycho.Kwp > ptycho.ObjApt_angle) = 0;
    #array_px = Kx.shape
    func_ObjApt = np.zeros((array_px, array_px), dtype = int)
    xx, yy = np.mgrid[:array_px, :array_px]

    circle =np.sqrt((xx - cen) ** 2 + (yy - cen) ** 2)
    alpha_px = alpha / K_px # alpha in px
    func_ObjApt[np.where(circle< alpha_px)] = 1
    #% dose equals to the summed intensity of the average ronchigram.
    #dose = sum(ptycho.pacbed(:)) *1.0;
    #% for resampled ronchigram
    #% pacbed_rs = interp2(tx_wp,ty_wp,mean_m_wp,Kx,Ky,'cubic',0);
    #% dose = sum(pacbed_rs(:)) *1.0;

    #% normalize the Objective Aperture to the desired number of electrons
    scaling = np.sqrt(dose/func_ObjApt.sum())
    func_ObjApt = func_ObjApt* scaling

    #% convergence aperture function; to filter out the updated probe func
    #% during ptychography iterations.
    #% ObjAptMask = func_ObjApt./sum(func_ObjApt(:));

    #% probe function - reciprocal space
    A = func_ObjApt*func_transfer
    #% probe function - real space
    func_probe=fftshift(ifft2(ifftshift(A)));

    if plot_me:
        fig_mul = 0.0625
        #max_fig = px_cal * array_px
        im_lim = [cen - (array_px * fig_mul), cen + (array_px * fig_mul)]
        fig_lim =[-px_cal *array_px *fig_mul ,px_cal *array_px *fig_mul]
        plt.figure;
        fig, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2,2)
        ax11.imshow(np.angle(A))#axis square;colorbar; axis off
        ax11.set_title('Aperture Phase Surface')
        ax11.set_xlim(im_lim)
        ax11.set_ylim(im_lim)
        ax12.imshow(abs(func_probe))#;axis image; colorbar; axis off
        ax12.set_title('Probe Function')
        ax12.set_xlim(im_lim)
        ax12.set_ylim(im_lim)
        ax21.imshow(np.real(func_probe))
        ax21.set_title('real')
        ax21.set_xlim(im_lim)
        ax21.set_ylim(im_lim)
        ax22.imshow(np.imag(func_probe))
        ax22.set_title('imag')
        ax22.set_xlim(im_lim)
        ax22.set_ylim(im_lim)
        #plt.title(str(aberr_input[0]) + str(aberr_input[7]))
        x_list = np.arange(-cen * px_cal, cen*px_cal, px_cal)
        plt.figure()
        plt.plot(x_list, abs(func_probe[int(array_px/2), :]))
        plt.plot(x_list, np.real(func_probe[int(array_px/2), :]))
        plt.plot(x_list, np.imag(func_probe[int(array_px/2), :]))
        plt.xlim(fig_lim)
        plt.title('df = ' + str(aberr_input[0]) + ' Cs = ' + str(aberr_input[7]))
    return func_probe
#    end
#
#    ptycho.func_aber = func_aber;
#    ptycho.func_transfer = func_transfer;
#    ptycho.func_ObjApt = func_ObjApt;
#    ptycho.A = A;
#    ptycho.func_probe = func_probe;
#    ptycho.aberr_input = aberr_input;
#    ptycho.scaling = scaling;
#
#    ptycho.varfunctions.define_probe_function = 1;
#
#    end
V = 15000 #V
alpha = 100e-3 # rad
px_cal = 0.45e-10 # in m
array_px = 4096
output_size = 256
save_hdf5 = False
save_path = r'Y:\2019\cm22979-8\processing\Merlin\20191114_15kVptycho_graphene\probe_sims'
save_file = r'\15kV_10um_Cs987um'

aberrcoeff = np.zeros((12))
aberrcoeff[0] = 0 # defocus
aberrcoeff[1] = 0# 2 stig
aberrcoeff[2] = 0  # 2 stig
aberrcoeff[3] = 0 # 3 stig
aberrcoeff[4] = 0 # 3 stig
aberrcoeff[5] = 0 # coma
aberrcoeff[6] = 0 # coma
aberrcoeff[7] = 987e-6 # Spherical abb
aberrcoeff[8] = 0# 4 stig
aberrcoeff[9]  = 0# 4 stig
aberrcoeff[10] = 0 # star
aberrcoeff[11]  = 0 # star
aberr_input = aberrcoeff
probe = define_probe_function(V,alpha, px_cal,array_px, aberrcoeff, dose = 1, plot_me = False)
px_from, px_to = int(array_px/2 - output_size / 2) , int(array_px/2 + output_size / 2)
output_probe = probe[px_from:px_to, px_from:px_to]
plt.figure(); plt.imshow(np.real(output_probe))

if save_hdf5 == True:
    output_probe = output_probe[np.newaxis, np.newaxis, np.newaxis, np.newaxis,np.newaxis, :,:]
    fn = save_path + save_file
    d5 = h5py.File(fn +'.hdf5' , 'w')
    d5.create_dataset('entry_1/process_1/output_1/probe', data = output_probe)
    d5.create_dataset('entry_1/process_1/PIE_1/detector/binning', data = [1,1])
    d5.create_dataset('entry_1/process_1/PIE_1/detector/upsample', data = [1,1])
    d5.create_dataset('entry_1/process_1/PIE_1/detector/crop',data = [output_size, output_size])
    d5.create_dataset('entry_1/process_1/common_1/dx', data = [4.52391605e-11 , 4.52391605e-11 ]) # px size
    d5.close()
