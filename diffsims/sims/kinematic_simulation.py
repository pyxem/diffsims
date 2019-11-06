'''
Created on 1 Nov 2019

Back end for computing diffraction patterns with a kinematic model.

@author: Rob Tovey
'''
from diffsims.utils.discretise_utils import getDiscretisation
from numpy import array, pi, sin, cos, empty
from scipy.interpolate import interpn
from diffsims.utils.fourier_transform import (getDFT, toFreq, fftshift_phase,
    plan_fft, fast_abs)
from diffsims.utils.generic_utils import toMesh


def normalise(arr): return arr / arr.max()


def get_diffraction_image(coordinates, species, probe, x, wavelength,
                          precession, pointwise=True):
    '''
    Return kinematically simulated diffraction pattern

    Parameters
    ----------
    coordinates : ndarray of shape [n_atoms, 3] and floating point type
        List of atomic coordinates
    species : ndarray of shape [n_atoms] and integer type
        List of atomic numbers
    probe : instance of probeFunction
        Function representing 3D shape of beam
    x : list of length 3 of 1D ndarrays
        Mesh on which to compute the volume density
    wavelength : float
        Wavelength of electron beam
    precession : a pair (float, int)
        The float dictates the angle of precession and the int how many points are
        used to discretise the integration.
    pointwise : bool
        Optional parameter whether atomic intensities are computed point-wise at
        the centre of a voxel or an integral over the voxel. default=True

    Returns
    -------
    DP : ndarray
        The two-dimensional diffraction pattern evaluated on the reciprocal grid
        corresponding to the first two vectors of <x>.
    '''
    y = toFreq(x)
    if wavelength == 0:
        p = probe(x).mean(-1)
        # TODO: shouldn't have to compute the full volume
        vol = getDiscretisation(coordinates, species, x, pointwise).mean(-1)
#         print(vol.max())
#         vol = getDiscretisation(coordinates, species, x[:2], pointwise).sum(-1)
#         print(vol.max())
        ft = getDFT(x[:-1], y[:-1])[0]
    else:
        p = probe(x)
        vol = getDiscretisation(coordinates, species, x, pointwise)
        ft = getDFT(x, y)[0]

    if precession[0] == 0:
        arr = ft(vol * p)
        arr = fast_abs(arr, arr).real ** 2
        if wavelength == 0:
            return normalise(arr)
        else:
            return normalise(grid2sphere(arr, y, None, 1 / wavelength))

    R = [precess_mat(precession[0], i * 360 / precession[1]) for i in range(precession[1])]

    if wavelength == 0:
        return normalise(sum(get_diffraction_image(coordinates.dot(r),
                                         species, probe, x, wavelength,
                                         (0, 1), pointwise)
                                         for r in R))

    # TODO: cast vol to dtype?
    fftshift_phase(vol)  # removes need for fftshift after fft
    buf = empty(vol.shape, dtype=vol.dtype)
    ft, buf = plan_fft(buf, overwrite=True, planner=1)
    DP = None
    for r in R:
        probe(toMesh(x, r.T), out=buf, scale=vol)  # buf = bess*vol

        # Do convolution
        newFT = ft()
        newFT = fast_abs(newFT, buf).real
        newFT *= newFT  # newFT = abs(newFT) ** 2
        newFT = grid2sphere(newFT.real, y, list(r), 1 / wavelength)

        if DP is None:
            DP = newFT
        else:
            DP += newFT

    return normalise(DP)


def precess_mat(alpha, theta):
    '''
    Generates rotation matrices for precession curves.

    Parameters
    ----------
    alpha : float
        Angle (in degrees) of precession tilt
    theta : float
        Angle (in degrees) along precession curve

    Returns
    -------
    R : ndarray of shape [3,3]
        Rotation matrix associated to the tilt of <alpha> away from the vertical
        axis and a rotation of <theta> about the vertical axis.
    '''
    if alpha == 0:
        return array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    alpha, theta = alpha * pi / 180, theta * pi / 180
    R_a = array([[1, 0, 0], [0, cos(alpha), -sin(alpha)],
                 [0, sin(alpha), cos(alpha)]])
    R_t = array([[cos(theta), -sin(theta), 0],
                 [sin(theta), cos(theta), 0], [0, 0, 1]])
    R = (R_t.T.dot(R_a.dot(R_t)))

    return R


def grid2sphere(arr, x, dx, C):
    '''
    Projects 3d array onto a sphere

    Parameters
    ----------
    arr : ndarray of dimension 3
        Input function to be projected
    x : list of 1D ndarrays
        Vectors defining mesh of <arr>
    dx : list of 1D ndarrays
        Basis in which to orient sphere. Centre of sphere will be at <C>*<dx>[2]
        and mesh of output array will be defined by the first two vectors
    C : float
        Radius of sphere

    Returns
    -------
    out : ndarray of dimension 2
        If y is the point on the line between i*dx[0]+j*dx[1] and C*dx[2]
        which also lies on the sphere of radius C from C*dx[2] then:
            out[i,j] = arr(y)
        Interpolation on arr is linear.
    '''
    if C in (None, 0) or x[2].size == 1:
        if arr.ndim == 2:
            return arr
        elif arr.shape[2] == 1:
            return arr[:, :, 0]

    y = toMesh((x[0], x[1], array([0])), dx).reshape(-1, 3)
#     if C is not None: # project straight up
#         w = C - sqrt(maximum(0, C ** 2 - (y ** 2).sum(-1)))
#         if dx is None:
#             y[:, 2] = w.reshape(-1)
#         else:
#             y += w.reshape(y.shape[0], 1) * dx[2].reshape(1, 3)

    if C is not None:  # project on line to centre
        w = 1 / (1 + (y ** 2).sum(-1) / C ** 2)
        y *= w[:, None]
        if dx is None:
            y[:, 2] = C * (1 - w)
        else:
            y += C * (1 - w)[:, None] * dx[2]

    out = interpn(x, arr, y, method='linear', bounds_error=False, fill_value=0)

    return out.reshape(x[0].size, x[1].size)

