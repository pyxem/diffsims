'''
Created on 31 Oct 2019

Module provides optimised fft and Fourier transform approximation.

@author: Rob Tovey
'''

from numpy import array, pi, inf, ceil, arange, exp, isscalar, prod, require, \
    empty
from numpy.fft import fftfreq
import numba

# This section produces 6 utilities:
#    fftn, ifftn, ifftshift, fftshift, plan_fft, plan_ifft
# By default, uses the pyfftw implementations. If a lot of identical ffts are
# needed then use the planning functions.
try:
    import pyfftw
    next_fast_len = pyfftw.next_fast_len
    from pyfftw.interfaces.numpy_fft import fftn, ifftn, ifftshift, fftshift
    pyfftw.config.NUM_THREADS = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
    pyfftw.interfaces.cache.enable()  # This can increase memory hugely sometimes
#     pyfftw.interfaces.cache.set_keepalive_time(10)

    def plan_fft(A, n=None, axis=None, overwrite=False, planner=1, threads=None,
            auto_align_input=True, auto_contiguous=True,
            avoid_copy=False, norm=None):
        '''
        Plans an fft for repeated use.

        Parameters
        ----------
        A : ndarray
            Array of same shape to be input for the fft
        n : iterable
            The output shape of fft (default=None is same as A.shape)
        axis : int or iterable
            The axis (or axes) to transform (default=None is all axes)
        overwrite : bool
            Whether the input array can be overwritten during computation
            (default=False)
        planner : (0, 1, 2 or 3)
            Amount of effort put into optimising Fourier transform. 0 is very low
            and 3 is very high (default=1).
        threads : int
            Number of threads to use (default=None is all threads)
        auto_align_input : bool
            If True then may re-align input (default=True)
        auto_contiguous : bool
            If True then may re-order input (default=True)
        avoid_copy : bool
            If True then may over-write initial input (default=False)
        norm : (None, 'ortho')
            Indicate whether fft is normalised (default=None)

        Returns
        -------
        plan : function
            `plan()` returns the fft of B
        B : ndarray
            Array which should be modified inplace for fft to be computed

        '''

        if threads is None:
            threads = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
        planner_effort = 'FFTW_' + ['ESTIMATE', 'MEASURE', 'PATIENT', 'EXHAUSTIVE'][planner]

        plan = pyfftw.builders.fftn(A, n, axis, overwrite,
                                   planner_effort, threads,
                                   auto_align_input, auto_contiguous,
                                   avoid_copy, norm)

        return plan, plan.input_array

    def plan_ifft(A, n=None, axis=None, overwrite=False, planner=1, threads=None,
            auto_align_input=True, auto_contiguous=True,
            avoid_copy=False, norm=None):
        '''
        Plans an ifft for repeated use.

        Parameters
        ----------
        A : ndarray
            Array of same shape to be input for the fft
        n : iterable
            The output shape of ifft (default=None is same as A.shape)
        axis : int or iterable
            The axis (or axes) to transform (default=None is all axes)
        overwrite : bool
            Whether the input array can be overwritten during computation
            (default=False)
        planner : (0, 1, 2 or 3)
            Amount of effort put into optimising Fourier transform. 0 is very low
            and 3 is very high (default=1).
        threads : int
            Number of threads to use (default=None is all threads)
        auto_align_input : bool
            If True then may re-align input (default=True)
        auto_contiguous : bool
            If True then may re-order input (default=True)
        avoid_copy : bool
            If True then may over-write initial input (default=False)
        norm : (None, 'ortho')
            Indicate whether ifft is normalised (default=None)

        Returns
        -------
        plan : function
            `plan()` returns the ifft of B
        B : ndarray
            Array which should be modified inplace for ifft to be computed
        '''

        if threads is None:
            threads = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
        planner_effort = 'FFTW_' + ['ESTIMATE', 'MEASURE', 'PATIENT', 'EXHAUSTIVE'][planner]

        plan = pyfftw.builders.ifftn(A, n, axis, overwrite,
                                   planner_effort, threads,
                                   auto_align_input, auto_contiguous,
                                   avoid_copy, norm)
        return plan, plan.input_array

except ImportError:
    # Only scipy has a next_fast_len, usually numpy is a little faster
    # (note they are not identical)
    from scipy.fftpack import fftn, ifftn, ifftshift, fftshift, next_fast_len
    from numpy.fft import fftn, ifftn, ifftshift, fftshift

    def plan_fft(A, n=None, axis=None, norm=None, **_):
        '''
        Plans an fft for repeated use.

        Parameters
        ----------
        A : ndarray
            Array of same shape to be input for the fft
        n : iterable
            The output shape of fft (default=None is same as A.shape)
        axis : int or iterable
            The axis (or axes) to transform (default=None is all axes)
        overwrite : bool
            Whether the input array can be overwritten during computation
            (default=False)
        planner : (0, 1, 2 or 3)
            Amount of effort put into optimising Fourier transform. 0 is very low
            and 3 is very high (default=1).
        threads : int
            Number of threads to use (default=None is all threads)
        auto_align_input : bool
            If True then may re-align input (default=True)
        auto_contiguous : bool
            If True then may re-order input (default=True)
        avoid_copy : bool
            If True then may over-write initial input (default=False)
        norm : (None, 'ortho')
            Indicate whether fft is normalised (default=None)

        Returns
        -------
        plan : function
            `plan()` returns the fft of B
        B : ndarray
            Array which should be modified inplace for fft to be computed

        '''
        return lambda : fftn(A, n, axis, norm), A

    def plan_ifft(A, n=None, axis=None, norm=None, **_):
        '''
        Plans an ifft for repeated use.

        Parameters
        ----------
        A : ndarray
            Array of same shape to be input for the fft
        n : iterable
            The output shape of ifft (default=None is same as A.shape)
        axis : int or iterable
            The axis (or axes) to transform (default=None is all axes)
        overwrite : bool
            Whether the input array can be overwritten during computation
            (default=False)
        planner : (0, 1, 2 or 3)
            Amount of effort put into optimising Fourier transform. 0 is very low
            and 3 is very high (default=1).
        threads : int
            Number of threads to use (default=None is all threads)
        auto_align_input : bool
            If True then may re-align input (default=True)
        auto_contiguous : bool
            If True then may re-order input (default=True)
        avoid_copy : bool
            If True then may over-write initial input (default=False)
        norm : (None, 'ortho')
            Indicate whether ifft is normalised (default=None)

        Returns
        -------
        plan : function
            `plan()` returns the ifft of B
        B : ndarray
            Array which should be modified inplace for ifft to be computed
        '''
        return lambda : ifftn(A, n, axis, norm), A


def fast_fft_len(n):
    '''
    Returns the smallest integer greater than input such that the fft can
    be computed efficiently at this size

    Parameters
    ----------
    n : int
        minimum size

    Returns
    -------
    N : int
        smallest integer greater than n which permits efficient ffts.
    '''
    N = next_fast_len(n)
    return N if N % 2 == 0 else fast_fft_len(N + 1)


def fftshift_phase(x):
    '''
    Fast implementation of fft_shift:
        fft(fftshift_phase(x)) = fft_shift(fft(x))

    Note two things:
    - this is an in-place manipulation of the (3D) input array
    - the input array must have even side lengths. This is softly guarranteed by
        fast_fft_len but will raise error if not true.
    '''
    assert all((s % 2 == 0) or (s == 1) for s in x.shape)
    sz = x.shape
    shrink = [s for s in sz if s > 1]
    x.reshape(shrink)
    if len(shrink) == 1:
        __fftshift_phase1(x)
    elif len(shrink) == 2:
        __fftshift_phase2(x)
    else:
        __fftshift_phase3(x)
    return x.reshape(sz)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def __fftshift_phase1(x):
    sz = x.shape[0] // 2
    for i in numba.prange(sz):
        x[2 * i + 1] = -x[2 * i + 1]


@numba.jit(nopython=True, parallel=True, fastmath=True)
def __fftshift_phase2(x):
    for i in numba.prange(x.shape[0]):
        start = (i + 1) % 2
        for j in range(start, x.shape[1], 2):
            x[i, j] = -x[i, j]


@numba.jit(nopython=True, parallel=True, fastmath=True)
def __fftshift_phase3(x):
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[2]):
            start = (i + j + 1) % 2
            for k in range(start, x.shape[1], 2):
                x[i, j, k] = -x[i, j, k]


def fast_abs(x, y=None):
    '''
    Fast computation of abs of an array

    Parameters
    ----------
    x : ndarray
        Input
    y : ndarray or None (default)
        If <y> is not None, used as preallocated output

    Returns
    -------
    y : ndarray
        Array equal to abs(<x>)
    '''
    if y is None:
        y = empty(x.shape, dtype=abs(x[(slice(1),) * x.ndim]).dtype)
    __fast_abs(x.reshape(-1), y.reshape(-1))
    return y


@numba.jit(nopython=True, parallel=True, fastmath=True)
def __fast_abs(x, y):
    for i in numba.prange(x.size):
        y[i] = abs(x[i])


# TODO: implement a switch for 2pi convention
# TODO: should x[0] be of shape (-1,1,1,...)?
def toFreq(x):
    '''
    Converts spatial coordinates to Fourier frequencies.

    Parameters
    ----------
    x : iterable collection of 1D ndarrays
        List (or equivalent) of vectors which define a mesh in the dimension
        equal to the length of x

    Returns
    -------
    y : list of 1D ndarrays
        List of vectors defining a mesh such that for a function, f, defined on
        the mesh given by x, fft(f) is defined on the mesh given by y
    '''
    y = []
    for X in x:
        if X.size > 1:
            y.append(fftfreq(X.size, X.item(1) - X.item(0)) * (2 * pi))
        else:
            y.append(array([0]))
    return [fftshift(Y) for Y in y]


def fromFreq(y):
    '''
    Converts Fourier frequencies to spatial coordinates.

    Parameters
    ----------
    y : iterable collection of 1D ndarrays
        List (or equivalent) of vectors which define a mesh in the dimension
        equal to the length of x

    Returns
    -------
    x : list of 1D ndarrays
        List of vectors defining a mesh such that for a function, f, defined on
        the mesh given by y, ifft(f) is defined on the mesh given by x. 0 will be
        in the middle of x.
    '''
    x = []
    for Y in y:
        if Y.size > 1:
            x.append(fftfreq(Y.size, Y.item(1) - Y.item(0)) * (2 * pi))
        else:
            x.append(array([0]))
    return [fftshift(X) for X in x]


def getFTpoints(ndim, n=None, dX=inf, rX=0, dY=inf, rY=1e-16):
    '''
    Returns a minimal pair of real and Fourier grids which satisfy each given
    requirement.

    Parameters
    ----------
    ndim : int
        Dimension of domain
    n : int (or list of length <ndim>)
        Sugested number of pixels (per dimension). default=None infers this from
        other parameters. If enough other constraints are given to define a
        discretisation then this will be shrunk if possible.
    dX : float > 0 (or list of length <ndim>)
        Maximum grid spacing (per dimension). default=None infers this from other
        parameters
    rX : float > 0 (or list of length <ndim>)
        Minimum grid range (per dimension). default=None infers this from other
        parameters. In this case, range is maximal span, i.e. diameter.
    dY : float > 0 (or list of length <ndim>)
        Maximum grid spacing (per dimension) in Fourier domain. default=None infers
        this from other parameters
    rY : float > 0 (or list of length <ndim>)
        Minimum grid range (per dimension) in Fourier domain. default=None infers
        this from other parameters. In this case, range is maximal span, i.e.
        diameter.

    Returns
    -------
    x : list of 1D ndarrays
        Real mesh of points, centred at 0 with at least <n> pixels, resolution
        higher than <dX>, and range greater than <rX>.
    y : list of 1D ndarrays
        Fourier mesh of points, centred at 0 with at least <n> pixels, resolution
        higher than <dY>, and range greater than <rY>.
    '''
    pad = lambda t: list(t) if hasattr(t, '__len__') else [t] * ndim
    n, dX, rX, dY, rY = (pad(t) for t in (n, dX, rX, dY, rY))

    X, Y = [], []
    for i in range(ndim):
        dX[i] = inf if dX[i] is None else dX[i]
        rX[i] = 0 if rX[i] is None else rX[i]
        dY[i] = inf if dY[i] is None else dY[i]
        rY[i] = 1e-16 if rY[i] is None else rY[i]

        r, d = max(rX[i], 2 * pi / dY[i]), min(dX[i], 2 * pi / rY[i])
        if n[i] is None:  # n not specified
            n[i] = r / d
        elif d > 1e15 or n[i] * d < r:  # n and d specified
            # Real range/ Fourier resolution is more important
            d = r / max(n[i], 1)
        elif r > 1e-10:  # all specified
            # If n can be reduced then do
            n[i] = min(n[i], r / d)
        n[i] = fast_fft_len(max(int(ceil(n[i])), 1))
        r = n[i] * d

        X.append(fftshift(fftfreq(n[i], 1 / r)))
        Y.append(2 * pi * fftshift(fftfreq(n[i], d)))

    return X, Y


def getDFT(X=None, Y=None):
    '''
    Returns discrete analogues for the Fourier/inverse Fourier transform pair
    defined from grid X to grid Y and back again.

    Parameters
    ----------
    X : list-like of 1D ndarrays
        Mesh on real space
    Y : list-like of 1D ndarrays
        Corresponding mesh on Fourier space

    If either X or Y is None then it is inferred from the other

    Returns
    -------
    DFT : function(f, axes=None)
        If <f> is a function on <X> then DFT(f) is the Fourier transform of <f> on
        <Y>. axes parameter can be used to specify which axes to transform.
    iDFT : function(f, axes=None)
        If <f> is a function on <Y> then iDFT(f) is the inverse Fourier transform
        of <f> on <X>. axes parameter can be used to specify which axes to transform.

    '''
    if X is None and Y is None:
        raise ValueError('Either X or Y must be provided')
    elif X is None:
        X = fromFreq(Y)
    elif Y is None:
        Y = toFreq(X)

    ndim = len(X)
    dx = [x.item(min(1, x.size - 1)) - x.item(0) for x in X]
    xmin = [x.item(0) for x in X]

    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def apply_phase_3D(x, f0, f1, f2):
        for i0 in numba.prange(x.shape[0]):
            F0 = f0[i0]
            for i1 in range(x.shape[1]):
                F01 = F0 * f1[i1]
                for i2 in range(x.shape[2]):
                    x[i0, i1, i2] *= F01 * f2[i2]

    def DFT(fx, axes=None):
        '''
        Discrete Fourier transform

        Parameters
        ----------
        fx : ndarray
            Array defining a function evaluated on a mesh.
        axes : (int, list of int)
            Specification of which axes to transform. default=None transforms all.

        Returns
        -------
        fy : ndarray
            The Fourier transform of <fx> evaluated on a mesh
        '''
        NDIM = fx.ndim
        if axes is None:
            axes = [NDIM + i for i in range(-ndim, 0)]
        elif not hasattr(axes, '__iter__'):
            axes = (axes,)
        axes = array(axes)
        axes.sort()

        FT = fftshift(fftn(fx, axes=axes), axes=axes)

        if NDIM != 3:
            # This is not typically a bottle-neck in <3D
            for i in axes:
                sz = [1] * NDIM
                sz[axes[i]] = -1
                FT *= exp(-xmin[i] * Y[i].reshape(sz) * 1j) * (dx[i] if dx[i] != 0 else 1)
        else:
            F = [exp(-xmin[i] * Y[i] * 1j) * (dx[i] if dx[i] != 0 else 1) for i in range(NDIM)]
            apply_phase_3D(FT, *F)

        return FT

    def iDFT(fy, axes=None):
        '''
        Discrete inverse Fourier transform

        Parameters
        ----------
        fy : ndarray
            Array defining a function evaluated on a mesh.
        axes : (int, list of int)
            Specification of which axes to transform. default=None transforms all.

        Returns
        -------
        fy : ndarray
            The Fourier transform of <fx> evaluated on a mesh
        '''
        NDIM = fy.ndim
        if axes is None:
            axes = [NDIM + i for i in range(-ndim, 0)]
        elif not hasattr(axes, '__iter__'):
            axes = (axes,)
        axes = array(axes)
        axes.sort()

        FT = fy.astype(
            'complex' + ('128' if fy.real.dtype.itemsize == 8 else '64'),
            copy=True)

        if NDIM != 3:
            # This is not typically a bottle-neck in <3D
            for i in axes:
                sz = [1] * NDIM
                sz[axes[i]] = -1
                FT *= exp(+xmin[i] * Y[i].reshape(sz) * 1j) / (dx[i] if dx[i] != 0 else 1)
        else:
            F = [exp(+xmin[i] * Y[i] * 1j) / (dx[i] if dx[i] != 0 else 1) for i in range(FT.ndim)]
            apply_phase_3D(FT, *F)

#       Equivalently: FT = ifftshift(FT, axes=axes)
        FT = ifftn(FT, axes=axes, overwrite_input=True)
        fftshift_phase(FT)  # removes need for ifftshift

        return FT

    return DFT, iDFT


def convolve(arr1, arr2, dx=None, axes=None):
    if arr2.ndim > arr1.ndim:
        arr1, arr2 = arr2, arr1
        if axes is None:
            axes = range(arr2.ndim)
    arr2 = arr2.reshape(arr2.shape + (1,) * (arr1.ndim - arr2.ndim))

    if dx is None:
        dx = 1
    elif isscalar(dx):
        dx = dx ** (len(axes) if axes is not None else arr1.ndim)
    else:
        dx = prod(dx)

    arr1 = fftn(arr1, axes=axes)
    arr2 = fftn(ifftshift(arr2), axes=axes)
    out = ifftn(arr1 * arr2, axes=axes) * dx
    return require(out, requirements='CA')
