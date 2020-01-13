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
Helper functions for gridding
"""

import numpy as np
from itertools import product
from transforms3d.euler import axangle2euler, euler2axangle

def convert_axangle_to_correct_range(vector,angle):
    """
    This repo uses axis-angle pairs between (0,pi) - however often wider
    ranges are used, most common are (0,2pi) and (-pi,pi), this function corrects
    for these
    """
    if (angle >= 0) and (angle < np.pi): #input in the desired convention
        pass
    elif (angle >= -np.pi) and (angle < 0):
        vector = np.multiply(vector,-1)
        angle  = angle * -1
    elif (angle >= np.pi) and (angle < 2*np.pi):
        vector = np.multiply(vector,-1)
        angle = 2*np.pi - angle
    else:
        raise ValueError("You have an axis-angle angle outside of acceptable ranges")

    return vector,angle


class AxAngle():
    """
    Class storing rotations in the axis-angle convention. Each row reads
    as [vx,vy,vz,theta], where [vx,vy,vz] is the rotation axis (normalised)
    and theta is the rotation angle in radians in range (0,pi)
    """
    def __init__(self,data):
        self.data = data.astype('float')
        self._check_data()
        return None

    def _check_data(self):
        """ Checks the data within AxAngle is acceptable, having the correct dimensions,
        acceptable angles and normalised vectors """
        if self.data.shape[1] != 4:
            raise ValueError("Your data is not in the correct shape")
        if np.any(self.data[:,3] < 0) or np.any(self.data[:,3] > np.pi):
            raise ValueError("Some of your angles lie outside of the range (0,pi)")
        if not np.allclose(np.linalg.norm(self.data[:,:3][self.data[:,3] > 0],axis=1),1):
            raise ValueError("You no longer have normalised direction vectors")
        return None

    def remove_large_rotations(self,threshold_angle):
        """
        Removes rotations that above a threshold angle

        Parameters
        ----------
        thereshold_angle : float
            angle in radians, rotations larger than this are removed

        Returns
        -------
        None :
            This functions operates in place
        """
        self._check_data()
        self.data = self.data[self.data[:,3] < threshold_angle]
        return None

    def to_Euler(self,axis_convention):
        """
        Produces euler angles from the axis-angle pairs.

        Parameters
        ----------
        axis_convention : str
            transforms3d compliant euler string

        Returns
        -------
        out_eulers : orix.Euler
        """
        from orix.np_inherits.euler import Euler
        self._check_data()
        stored_euler = np.ones((self.data.shape[0],3))
        for i,row in enumerate(self.data):
            a_array = axangle2euler(row[:3],row[3],axis_convention)
            for j in [0,1,2]:
                stored_euler[i,j] = a_array[j]

        stored_euler = np.rad2deg(stored_euler)
        return Euler(stored_euler,axis_convention)

    def to_Quat(self):
        self._check_data()
        stored_quat = np.ones((self.data.shape[0],4))
        for i,row in enumerate(self.data):
            q_array = axangle2quat(row[:3],row[3])
            for j in [0,1,2,3]:
                stored_quat = q_array[j]
        return stored_quat


    @classmethod
    def from_Quat(cls):
        pass


class Euler():
    """
    Class storing rotations as euler angles.
    Each row reads as [alpha,beta,gamma], where alpha, beta and gamma are rotations
    in degrees around the axes specified by Euler.axis_conventions
    as defined in transforms3d. Please always remember that Euler angles are difficult.
    """
    def __init__(self,data,axis_convention='rzxz'):
        self.data = data.astype('float')
        self.axis_convention = axis_convention
        self._check_data()
        return None

    def _check_data(self):
        """ Checks data within Euler is acceptable, to be used at the start
        of methods """
        if self.data.shape[1] != 3:
            raise ValueError("Your data is not in the correct shape")
        if np.any(self.data[:] > 360):
            raise ValueError("Some of your angles are greater 360")

        return None

    def to_AxAngle(self):
        """ Converts an Euler object to an AxAngle object

        Returns
        -------
        axangle : orix.AxAngle object
        """
        from orix.np_inherits.axangle import AxAngle,convert_axangle_to_correct_range
        self._check_data()
        stored_axangle = np.ones((self.data.shape[0],4))
        self.data = np.deg2rad(self.data) #for the transform operation
        for i,row in enumerate(self.data):
            temp_vect, temp_angle = euler2axangle(row[0],row[1],row[2],self.axis_convention)
            temp_vect,temp_angle  = convert_axangle_to_correct_range(temp_vect,temp_angle)
            for j in [0,1,2]:
                stored_axangle[i,j] = temp_vect[j]
            stored_axangle[i,3] = temp_angle #in radians!

        self.data = np.rad2deg(self.data) #leaves our eulers safe and sound
        return AxAngle(stored_axangle)

def create_linearly_spaced_array_in_rzxz(resolution):
    """
    Notes
    -----
    We use angular ranges alpha [0,360], beta [0,180] and gamma [0,360] in
    line with Convention 4 described in Reference [1]

    References
    ----------
    [1]  D Rowenhorst et al 2015 Modelling Simul. Mater. Sci. Eng.23 083501
         https://iopscience.iop.org/article/10.1088/0965-0393/23/8/083501/meta
    """
    num_steps = int(360/resolution + 0.5)
    alpha = np.linspace(0,360,num=num_steps,endpoint=False)
    beta  = np.linspace(0,180,num=int(num_steps/2),endpoint=False)
    gamma = np.linspace(0,360,num=num_steps,endpoint=False)
    z = np.asarray(list(product(alpha, beta, gamma)))
    return Euler(z,axis_convention='rzxz')

def rotate_axangle(Axangles,new_center):
    """
    Rotates a series of orientation described by axangle to a new center

    Parameters
    ----------
    Axangles :
        Axangles in the correct class
    new_center : (alpha,beta,gamma)
        The location of the (0,0,0) rotation as an rzxz euler angle
    """

    #convert axangle array to to_quat
    #find the relevant transformation quaternion
    #apply the relevant transformation quat
    # return new array of AxAngles
    pass


""" Fundemental Zone Functionality """
def select_fundemental_zone(space_group_number):
    """
    Parameters
    ----------
    space_group_number : int

    Returns
    -------
    point_group_str : str
        The proper point group string in --- convention

    Notes
    -----
    This function enumerates the list on https://en.wikipedia.org/wiki/List_of_space_groups
    Point groups (32) are converted to proper point groups (11) using the Schoenflies
    representations given in that table.
    """
    if space_group_number in [1,2]:
        return '1'   #triclinic
    if 2 < space_group_number < 16:
        return '2'   #monoclinic
    if 15 < space_group_number < 75:
        return '222' #orthorhomic
    if 74 < space_group_number < 143: #tetragonal
        if (74 < space_group_number < 89) or (99 < space_group_number < 110):
            return '4'  #cyclic
        else:
            return '422' #dihedral
    if 142 < space_group_number < 168: #trigonal
        if 142 < space_group_number < 148 or 156 < space_group_number < 161:
            return '3' #cyclic
        else:
            return '32' #dihedral
    if 167 < space_group_number < 194: #hexagonal
        if 167 < space_group_number <176 or space_group_number in [183,184,185,186]:
            return '6' #cyclic
        else:
            return '622'#dihedral
    if 193 < space_group_number < 231: #cubic
        if 193 < space_group_number < 207 or space_group_number in [215,216,217,218,219,220]:
            return '432' #oct
        else:
            return '23' #tet

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


def reduce_to_fundemental_zone(data,fundemental_zone):
    """
    Parameters
    ----------
    data :

    fundemental_zone : str
        A proper point group, allowed values are:
            '1','2','222','4','422','3','32','6','622','432','23'

    Returns
    -------
    reduced_data : orix.AxAngle

    """

    # we know what are max angles are, so save some time by cutting out chunks
    # see Figure 5 of "On 3 dimensional misorientation spaces"
    if fundemental_zone == '432':
        self.data = self.data[self.data[:,3] < np.deg2rad(66)]
    elif fundemental_zone == '222':
        self.data = self.data[self.data[:,3] < np.deg2rad(121)]
    elif fundemental_zone in ['23','622','32','422']:
        self.data = self.data[self.data[:,3] < np.deg2rad(106)]

    # convert to rodrigo-frank
    # call FZ functionality
