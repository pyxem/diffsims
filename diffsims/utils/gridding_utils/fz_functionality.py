# -*- coding: utf-8 -*-
# Copyright 2018-2019 The pyXem developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

def axangle2rodrigo_frank(z):
    # converts to [vx,vy,vz,RF]
    # RF = tan(omega/2)
    z[:,3] = np.tan(np.divide(z[:,3],2))
    return z

def rodrigo_frank_to_axangle():
    pass

def numpy_bounding_plane(data,vector,distance):
    """

    Raises
    -----
    ValueError : This function is unsafe if pi rotations are preset
    """
    if not np.all(np.is_finite(data)):
        raise ValueError("pi rotations, be aware")
        
    return data

def cyclic_group(data,order):
    """ By CONVENTION the rotation axis is the cartesian z axis
    Note: Special case, as pi rotations are present we avoid a call to numpy_bounding_plane"""
    z_distance = np.multiply(data[2],data[3])
    z_distance = np.abs(np.nan_to_num(z_distance)) #case pi rotation, 0 z component of vector
    return data[z_distance < np.tan(np.pi/order)]

def dihedral_group(data,order):
    pass

def octahedral_group(data):
    pass

def tetragonal_group(data):
    for direction in [(1,1,1),(1,1,-1)]: #etc
        data = numpy_bounding_plane()

def rf_fundemental_zone(axangledata,point_group_str):
    rf = axangle2rodrigo_frank(axangledata)
    if point_group_str in ['1','2','3','4','6']:
        rf = cyclic_group(rf,order = int(point_group_str))
    elif point_group_str in ['222','32','422','622']:
        rf = dihedral_group(rf,order=int(point_group_str[0]))
    elif: point_group_str == '23':
        rf = tetragonal_group(rf)
    elif point_group_str == '432':
        rf = octahedral_group(rf)
    return rodrigo_frank_to_axangle(rf)
