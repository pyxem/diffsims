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

""" Fundamental Zone Functionality """

import numpy as np


def get_proper_point_group_string(space_group_number):
    """
    Maps a space-group-number to a point group

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
    if space_group_number in [1, 2]:
        return '1'  # triclinic
    if 2 < space_group_number < 16:
        return '2'  # monoclinic
    if 15 < space_group_number < 75:
        return '222'  # orthorhomic
    if 74 < space_group_number < 143:  # tetragonal
        if (74 < space_group_number < 89) or (99 < space_group_number < 110):
            return '4'  # cyclic
        else:
            return '422'  # dihedral
    if 142 < space_group_number < 168:  # trigonal
        if 142 < space_group_number < 148 or 156 < space_group_number < 161:
            return '3'  # cyclic
        else:
            return '32'  # dihedral
    if 167 < space_group_number < 194:  # hexagonal
        if 167 < space_group_number < 176 or space_group_number in [183, 184, 185, 186]:
            return '6'  # cyclic
        else:
            return '622'  # dihedral
    if 193 < space_group_number < 231:  # cubic
        if 193 < space_group_number < 207 or space_group_number in [215, 216, 217, 218, 219, 220]:
            return '432'  # oct
        else:
            return '23'  # tet


def axangle2rodrigues_frank(z):
    """ Returns a copy of the array converted from [vx,vy,vz,omega] to [vx,vy,vz,RF]
    with RF = tan(omega/2) """

    rf = z.copy()
    rf[:, 3] = np.tan(np.divide(z[:, 3], 2))
    return rf


def numpy_bounding_plane(data, vector, distance):
    """
    Creates a mask formed by two planes that lie at 'distance' from the origin
    and lie along the 'vector'

    Parameters
    ----------
    data :
        The candidate rotations in Rodrigues-Frank to be removed/kept

    vector :
        The direction perpendicular to the plane under consideration

    distance :
        The perpendicular distance from the center to the plane

    Returns
    -------
    inner_region :
        True when element lies within the bounds of the planes

    Raises
    -----
    ValueError : This function is unsafe if pi rotations are preset
    """
    if not np.all(np.isfinite(data)):
        raise ValueError("Your data contains rotations of pi")

    n_vector = np.divide(vector, np.linalg.norm(vector))
    inner_region = (np.abs(np.dot(data[:, :3], n_vector)) * data[:, 3]) < distance

    return inner_region


def cyclic_group(data, order):
    """
    Keeps only rotations within the domain of the cyclic group under consideration

    Parameters
    ----------
    data : np.array
        The candidate rotations in Rodrigues-Frank to be removed/kept

    order :
        The order of the cyclic group

    Returns
    -------
    mask :
        True when element lies within the bounds of the domain

    Notes
    -----
    This makes use of the convention that puts the cyclic axis along z
    """
    # As pi rotations are present in the input and output we avoid a call to numpy_bounding_plane
    z_distance = np.multiply(data[:, 2], data[:, 3])  # gets the z component of the distance, can be nan
    z_distance = np.abs(np.nan_to_num(z_distance))  # case pi rotation, 0 z component of vector
    mask = z_distance < np.tan(np.pi / (2 * order))
    return mask


def dihedral_group(data, order):
    """
    Keeps only rotations within the domain of the dihedral group under consideration

    Parameters
    ----------
    data : np.array
        The candidate rotations in Rodrigues-Frank

    order :
        The order of the dihedral group

    Returns
    -------
    mask :
        True when element lies within the bounds of the domain

    Notes
    -----
    This makes use of the convention that puts the cyclic axis along 'z', an puts a minor axis along 'x'
    """
    mask_cyclic_axis = cyclic_group(data, order)

    angle_between_perpendicular_axes = np.deg2rad(180 / order)
    angle_counter = 0
    normal_vector_list = []
    while angle_counter < (2 * np.pi):
        normal_vector = [np.cos(angle_counter), np.sin(angle_counter), 0]
        normal_vector_list.append(normal_vector)
        angle_counter += angle_between_perpendicular_axes

    mask_other_axes = np.ones_like(data[:, 3])
    for normal_vector in normal_vector_list:
        local_mask = numpy_bounding_plane(data, normal_vector, 1)
        mask_other_axes = np.logical_and(mask_other_axes, local_mask)

    mask = np.logical_and(mask_cyclic_axis, mask_other_axes)

    return mask


def tetragonal_group(data):
    """
    Keeps only rotations within the tetragonal domain

    Parameters
    ----------
    data : np.array
        The candidate rotations in Rodrigues-Frank

    Returns
    -------
    mask :
        True when element lies within the bounds of the domain
    """
    mask = np.ones_like(data[:, 3])
    for normal_vector in [[1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1]]:
        normal_vector = np.divide(normal_vector, np.sqrt(3))
        local_mask = numpy_bounding_plane(data, normal_vector, 1 / np.sqrt(3))
        mask = np.logical_and(local_mask, mask)

    return mask


def octahedral_group(data):
    """
    Keeps only rotations within the octahedral domain

    Parameters
    ----------
    data : np.array
        The candidate rotations in Rodrigues-Frank

    Returns
    -------
    mask :
        True when element lies within the bounds of the domain
    """

    sub_mask_threefold = tetragonal_group(data)
    sub_mask_fourfold = np.ones_like(data[:, 3])
    for normal_vector in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        local_mask = numpy_bounding_plane(data, normal_vector, np.sqrt(2) - 1)
        sub_mask_fourfold = np.logical_and(local_mask, sub_mask_fourfold)

    mask = np.logical_and(sub_mask_threefold, sub_mask_fourfold)
    return mask


def remove_out_of_domain_rotations(Axangles, point_group_str):
    """ see Figure 5 of "On 3 dimensional misorientation spaces" """
    if point_group_str == '432':
        Axangles.remove_large_rotations(np.deg2rad(66))
    elif point_group_str == '222':
        Axangles.remove_large_rotations(np.deg2rad(121))
    elif point_group_str in ['23', '622', '32', '422']:
        Axangles.remove_large_rotations(np.deg2rad(106))

    return Axangles


def generate_mask_from_rodrigues_frank(Axangles, point_group_str):
    rf_data = axangle2rodrigues_frank(Axangles.data)
    if point_group_str in ['1', '2', '3', '4', '6']:
        mask = cyclic_group(rf_data, order=int(point_group_str))
    elif point_group_str in ['222', '32', '422', '622']:
        mask = dihedral_group(rf_data, order=int(point_group_str[0]))
    elif point_group_str == '23':
        mask = tetragonal_group(rf_data)
    elif point_group_str == '432':
        mask = octahedral_group(rf_data)

    return mask


def reduce_to_fundamental_zone(Axangles, point_group_str):
    """
    Parameters
    ----------
    Axangles : diffsims.AxAngle

    point_group_str : str
        A proper point group, allowed values are:
            '1','2','222','4','422','3','32','6','622','432','23'

    Returns
    -------
    reduced_data : diffsims.AxAngle
    """

    # we know what are max angles are, so save some time by cutting out chunks
    Axangles = remove_out_of_domain_rotations(Axangles, point_group_str)
    mask = generate_mask_from_rodrigues_frank(Axangles, point_group_str)
    Axangles.remove_with_mask(mask)
    return Axangles
