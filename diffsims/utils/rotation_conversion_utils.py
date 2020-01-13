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

"""

These utils provide vectorised implementations of conversions as descrbibed in the package transforms3d.

Currently avaliable are:

euler2quat
quat2axangle
euler2axangle (via chaining of the above)

"""

import numpy as np


def vectorised_euler2quat(ai, aj, ak, axes='rzxz'):
    """ Applies the transformation that takes eulers to quaternions

    Parameters
    ----------

    Returns
    -------

    """

    _NEXT_AXIS = [1,2,0,1]

    if axes != 'rzxz':
        raise ValueError()
    elif axes = 'rzxz':
        firstaxis,parity,repetition,frame = 2,0,1,1

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = np.divide(ai,2.0)
    aj = np.divide(aj,2.0)
    ak = np.divide(ak,2.0)
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((ai.shape[0],4))
    if repetition:
        q[:,0] = cj*(cc - ss)
        q[:,i] = cj*(cs + sc)
        q[:,j] = sj*(cc + ss)
        q[:,k] = sj*(cs - sc)
    else:
        q[:,0] = cj*cc + sj*ss
        q[:,i] = cj*sc - sj*cs
        q[:,j] = cj*ss + sj*cc
        q[:,k] = cj*cs - sj*sc

    if parity:
        q[:,j] *= -1.0

    return q

def vectorised_quat2axangle(q):
    """ Applies the transformation that takes quaternions to axis angles

    Parameters
    ----------

    Returns
    -------

    """

    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    Nq = w * w + x * x + y * y + z * z
    if not np.any(np.isfinite(Nq)):
        raise ValueError("Infinite elements are nightmare, check your entry")
    if np.any(Nq < 1e-6):
        raise ValueError("Very small numbers are at risk when we normalise")
    #normalize
    s = np.sqrt(Nq)
    w,x,y,z = np.divide(w,s),np.divide(x,s),np.divide(y,s),np.divide(z,s)
    len_img = np.sqrt((x*x)+(y*y)+(z*z))
    len_img_bool = (len_img < 1e-6)
    xr,yr,zr = np.divide(x,len_img),np.divide(y,len_img),np.divide(z,len_img)
    w[w > 1] = 1
    w[w < -1] = 1
    theta = 2 * np.arccos(w)
    output = np.asarray((xr,yr,zr,theta)).T
    return output#.T.reshape(xr.shape[0],4)
