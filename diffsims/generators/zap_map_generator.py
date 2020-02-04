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

import numpy as np
from transforms3d.euler import axangle2euler

def get_rotation_from_z(structure,direction):
    """
    Finds the rotation that takes [001] to a given zone axis.

    Parameters
    ----------
    structure : diffpy.structure

    direction : array like
        [UVW]

    Returns
    -------
    euler_angles : tuple
        'rzxz' in degrees

    See Also
    --------
    generate_zap_map
    get_grid_around_beam_direction

    Notes
    -----
    This implementation works with an initial implementation that has +x as left to right,
    +y as bottom to top and +z as out of the plane of a page. Rotatins are counter clockwise
    as you look from the tip of the axis towards the origin
    """

    # Case where we don't need a rotation
    if np.dot(direction,[0,0,1]) == np.linalg.norm(direction):
        return (0,0,0)

    # Normalize our directions
    cartesian_direction = structure.lattice.cartesian(direction)
    cartesian_direction = cartesian_direction / np.linalg.norm(cartesian_direction)

    rotation_axis = np.cross([0,0,1],cartesian_direction)
    rotation_angle = np.arccos(np.dot([0,0,1],cartesian_direction))
    euler = axangle2euler(rotation_axis,rotation_angle,axes='rzxz')
    return np.rad2deg(euler)

def get_sensible_reciprocal_radius(structure):
    """
    Returns a reasonable reciprocal_radius if user hasn't specified one

    Parameters
    ----------
    structure : diffpy.structure

    Returns
    -------
    reciprocal_radius : float
    """
    lattice = structure.lattice
    shortest_real_space = min((lattice.a,lattice.b,lattice.c))
    d_star_max = 1/shortest_real_space
    reciprocal_radius = 3 * d_star_max
    return reciprocal_radius

def generate_directional_simulations(structure,simulator,direction_list):
    """
    Produces simualtion of a structure aligned with certain axes

    Parameters
    ----------
    structure : diffpy.structure

    simulator :

    direction_list : list of lists
        A list of [UVW] indicies, eg) [[1,0,0],[1,1,0]]

    Returns
    -------
    direction_dictionary : dict
        Keys are zone axes, values are simulations
    """

    if reciprocal_radius not in kwargs.keys():
        reciprocal_radius = get_sensible_reciprocal_radius(structure)
    else:
        reciprocal_radius = kwargs['reciprocal_radius']

    direction_dictionary = {}
    for direction in direction_list:
        rotation_rzxz = get_rotation_from_z(structure,direction)
        simulation = simulator.calculate_ed_data(structure,reciprocal_radius,rotation=rotation_rzxz,**kwargs)
        # adds the direction and simulation to dictionary
    return direction_dictionary

def generate_zap_map(structure,simulator,density):
    """
    Produces a number of zone axis patterns for a structure

    Parameters
    ----------
    structure : diffpy.structure

    simulator : diffsims.diffraction_generator

    density : str
        '3' for the corners or '7' (corners + 3 midpoints + 1 centroid)

    **kwargs :
        keyword arguments to be passed to simulator.calculate_ed_data()

    Returns
    -------
    zap_dictionary : dict
        Keys are zone axes, values are simulations

    Example
    -------

    """

    # generate list of zone axes directions
    #direction_list =

    zap_dictionary = generate_directional_simulations(structure,simulator,direction_list,**kwargs)

    pass
