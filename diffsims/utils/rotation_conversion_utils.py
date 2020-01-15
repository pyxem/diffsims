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

axangle2mat
mat2euler
axangle2euler (via chaining of the above)

"""

import numpy as np

def vectorised_euler2quat(ai, aj, ak, axes='rzxz'):
    """ Applies the transformation that takes eulers to quaternions

    Parameters #TODO: change this to eulers
    ----------
    ai : (N) numpy array
        First euler angle (in radians)
    aj : (N) numpy array
        Second euler angle (in radians)
    ak : (N) numpy array
        Third euler angle (in radians)
    axes :
        Euler angles conventions, as detailed in transforms3d. Only 'rzxz' is
        currently supported

    Returns
    -------
    q : (N,4) numpy array
            Contains elements w,x,y,z

    Notes
    -----
    This function is a port from transforms3d
    """

    _NEXT_AXIS = [1,2,0,1]

    if axes != 'rzxz':
        raise ValueError()
    elif axes == 'rzxz':
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
    q : (N,4) numpy array
        Contains elements w,x,y,z

    Returns
    -------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the normalised vector and
        theta is the angle in radians

    Notes
    -----
    This function is a port of the associated function in transforms3d. As there
    for the identity rotation [1,0,0,0] is returned

    """

    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    Nq = w * w + x * x + y * y + z * z
    if not np.any(np.isfinite(Nq)):
        raise ValueError("You have infinte elements, please check your entry")
    if np.any(Nq < 1e-6):
        raise ValueError("Very small numbers are at risk when we normalise, try normalising your quaternions")

    if np.any(Nq != 1):#normalize
        s = np.sqrt(Nq)
        w,x,y,z = np.divide(w,s),np.divide(x,s),np.divide(y,s),np.divide(z,s)

    len_img = np.sqrt((x*x)+(y*y)+(z*z))
    # case where the vector is nearly [0,0,0], return [1,0,0,0]
    x = np.where(len_img==0,1,x)
    y = np.where(len_img==0,0,y)
    z = np.where(len_img==0,0,z)
    w  = np.where(len_img==0,0,w)

    len_img = np.sqrt((x*x)+(y*y)+(z*z)) #recalculated so we avoid a divide by zero
    xr,yr,zr = np.divide(x,len_img),np.divide(y,len_img),np.divide(z,len_img)

    w[w > 1] = 1
    w[w < -1] = -1
    theta = 2 * np.arccos(w)
    axangles = np.asarray((xr,yr,zr,theta)).T
    return axangles

def vectorised_euler2axangle(ai, aj, ak, axes='rzxz'):
    """ Applies the transformation that takes eulers to axis-angles

    Parameters
    ----------
    ai : (N) numpy array
        First euler angle (in radians)
    aj : (N) numpy array
        Second euler angle (in radians)
    ak : (N) numpy array
        Third euler angle (in radians)
    axes :
        Euler angles conventions, as detailed in transforms3d. Only 'rzxz' is
        currently supported

    Returns
    -------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the normalised vector and
        theta is the angle in radians

    Notes
    -----
    This function is a port of the associated function(s) in transforms3d. As there
    for the identity rotation [1,0,0,0] is returned

    """
    return vectorised_quat2axangle(vectorised_euler2quat(ai,aj,ak,axes))

def vectorised_axangle2mat(axangles):
    """ Applies the transformation that takes eulers to axis-angles

    Parameters
    ----------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the (ideally normalised) vector and
        theta is the angle in radians

    Returns
    -------
    M : (3,3,N) np.array
        For internal use only

    Notes
    -----
    This function is a port of the associated function in transforms3d

    """
    x, y, z, angle = axangles[:,0], axangles[:,1], axangles[:,2], axangles[:,3]

    # Normalise for safety, skipping divide by zeros
    n = np.sqrt(x*x + y*y + z*z)
    x = np.where(n!=0,x/n,0)
    y = np.where(n!=0,y/n,0)
    z = np.where(n!=0,z/n,0)

    c = np.cos(angle); s = np.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    M = np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])

    return M

def vectorised_mat2euler(M,axes='rzxz'):
    """
    Applies the transformation to take rotation matricies to euler angles

    Parameters
    ----------
    M : (3,3,N) np.array
        From internal use only, returned by vectorised_axangle2mat
    axes: string
        Compliant convention string, only 'rzxz' is currently accepted

    Returns
    -------
    eulers : (N,4) numpy array
        Contains elements ai,aj,ak where i,j,k are determined by axes

    Notes
    -----
    This function is a port from transforms3d
    """
    _NEXT_AXIS = [1,2,0,1]

    if axes != 'rzxz':
        raise ValueError()
    elif axes == 'rzxz':
        firstaxis,parity,repetition,frame = 2,0,1,1

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]


    if repetition:
        sy = np.sqrt(M[i, j,:]*M[i, j,:] + M[i, k,:]*M[i, k,:])

        ax = np.where(sy > 0,np.arctan2( M[i, j,:],  M[i, k,:]),np.arctan2(-M[j, k,:],  M[j, j,:]))
        ay = np.arctan2(sy,M[i, i,:])
        az = np.where(sy > 0,np.arctan2(M[j, i,:], -M[k, i,:]),0.0)

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax

    euler = np.vstack((ax,ay,az)).T
    return euler

def vectorised_axangle2euler(axangles,axes='rzxz'):
    """ Applies the transformation that takes eulers to axis-angles

    Parameters
    ----------
    axangle : (N,4) numpy array
        Elements are [x,y,z,theta] in which [x,y,z] is the normalised vector and
        theta is the angle in radians
    axes: string
        Compliant convention string, only 'rzxz' is currently accepted

    Returns
    -------
    eulers : (N,4) numpy array
        Contains elements ai,aj,ak where i,j,k are determined by axes

    Notes
    -----
    This function is a port of the associated function(s) in transforms3d.
    """
    return vectorised_mat2euler(vectorised_axangle2mat(axangles),axes)
