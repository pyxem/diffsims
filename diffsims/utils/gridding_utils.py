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
from transforms3d.euler import axangle2euler

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
