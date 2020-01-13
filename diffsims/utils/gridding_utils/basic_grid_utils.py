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

import numpy as np
from itertools import product

from orix.np_inherits.euler import Euler

def create_linearly_spaced_array_in_szxz(resolution):
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
    #TODO: Potentially load some v. v. common grids as a speed up (1,0.5 etc)

    num_steps = int(360/resolution + 0.5)
    alpha = np.linspace(0,360,num=num_steps,endpoint=False)
    beta  = np.linspace(0,180,num=int(num_steps/2),endpoint=False)
    gamma = np.linspace(0,360,num=num_steps,endpoint=False)
    z = np.asarray(list(product(alpha, beta, gamma)))
    return Euler(z,axis_convention='szxz')

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
