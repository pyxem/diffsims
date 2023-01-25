# -*- coding: utf-8 -*-
# Copyright 2017-2023 The diffsims developers
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


def get_rotation_from_z_to_direction(structure, direction):
    """
    Finds the rotation that takes [001] to a given zone axis.

    Parameters
    ----------
    structure : diffpy.structure.structure.Structure
        The structure for which a rotation needs to be found.
    direction : array like
        [UVW] direction that the 'z' axis should end up point down.

    Returns
    -------
    euler_angles : tuple
        'rzxz' in degrees.

    See Also
    --------
    generate_zap_map
    :meth:`~diffsims.generators.rotation_list_generators.get_grid_around_beam_direction`

    Notes
    -----
    This implementation works with an axis arrangement that has +x as
    left to right, +y as bottom to top and +z as out of the plane of a
    page. Rotations are counter clockwise as you look from the tip of the
    axis towards the origin
    """
    # Case where we don't need a rotation, As axis is [0,0,z] or [0,0,0]
    if np.dot(direction, [0, 0, 1]) == np.linalg.norm(direction):
        return (0, 0, 0)

    # Normalize our directions
    cartesian_direction = structure.lattice.cartesian(direction)
    cartesian_direction = cartesian_direction / np.linalg.norm(cartesian_direction)

    # Find the rotation using cartesian vector geometry
    rotation_axis = np.cross([0, 0, 1], cartesian_direction)
    rotation_angle = np.arccos(np.dot([0, 0, 1], cartesian_direction))
    euler = axangle2euler(rotation_axis, rotation_angle, axes="rzxz")
    return np.rad2deg(euler)


def generate_directional_simulations(
    structure, simulator, direction_list, reciprocal_radius=1, **kwargs
):
    """
    Produces simulation of a structure aligned with certain axes

    Parameters
    ----------
    structure : diffpy.structure.structure.Structure
        The structure from which simulations need to be produced.
    simulator : DiffractionGenerator
        The diffraction generator object used to produce the simulations
    direction_list : list of lists
        A list of [UVW] indices, eg. [[1,0,0],[1,1,0]]
    reciprocal_radius : float
        Default to 1

    Returns
    -------
    direction_dictionary : dict
        Keys are zone axes, values are simulations
    """

    direction_dictionary = {}
    for direction in direction_list:
        if np.allclose(direction, 0):
            break
        rotation_rzxz = get_rotation_from_z_to_direction(structure, direction)
        simulation = simulator.calculate_ed_data(
            structure, reciprocal_radius, rotation=rotation_rzxz, **kwargs
        )
        direction_dictionary[direction] = simulation

    return direction_dictionary


def corners_to_centroid_and_edge_centers(corners):
    """
    Produces the midpoints and center of a trio of corners

    Parameters
    ----------
    corners : list of lists
        Three corners of a streographic triangle

    Returns
    -------
    list_of_corners : list
        Length 7, elements ca, cb, cc, mean, cab, cbc, cac where naming is such that
        ca is the first corner of the input, and cab is the midpoint between
        corner a and corner b.
    """
    ca, cb, cc = corners[0], corners[1], corners[2]
    mean = tuple(np.add(np.add(ca, cb), cc))
    cab = tuple(np.add(ca, cb))
    cbc = tuple(np.add(cb, cc))
    cac = tuple(np.add(ca, cc))
    return [ca, cb, cc, mean, cab, cbc, cac]


def generate_zap_map(
    structure, simulator, system="cubic", reciprocal_radius=1, density="7", **kwargs
):
    """
    Produces a number of zone axis patterns for a structure

    Parameters
    ----------
    structure : diffpy.structure.structure.Structure
        The structure to be simulated.
    simulator : DiffractionGenerator
        The simulator used to generate the simulations
    system : str
        'cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic',
        'monoclinic'. Defaults to 'cubic'.
    reciprocal_radius : float
        The range of reciprocal lattice spots to be included. Default to
        1.
    density : str
        '3' for the corners or '7' (corners + midpoints + centroids).
        Defaults to 7.
    kwargs :
        Keyword arguments to be passed to simulator.calculate_ed_data().

    Returns
    -------
    zap_dictionary : dict
        Keys are zone axes, values are simulations

    Examples
    --------
    Plot all of the patterns that you have generated

    >>> zap_map = generate_zap_map(structure,simulator,'hexagonal',density='3')
    >>> for k in zap_map.keys():
    >>>     pattern = zap_map[k]
    >>>     pattern.calibration = 4e-3
    >>>     plt.figure()
    >>>     plt.imshow(pattern.get_diffraction_pattern(),vmax=0.02)
    """

    corners_dict = {
        "cubic": [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
        "hexagonal": [(0, 0, 1), (2, 1, 0), (1, 1, 0)],
        "orthorhombic": [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
        "tetragonal": [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
        "trigonal": [(0, 0, 1), (-1, -2, 0), (1, -1, 0)],
        "monoclinic": [(0, 0, 1), (0, 1, 0), (0, -1, 0)],
    }

    if density == "3":
        direction_list = corners_dict[system]
    elif density == "7":
        direction_list = corners_to_centroid_and_edge_centers(corners_dict[system])

    zap_dictionary = generate_directional_simulations(
        structure, simulator, direction_list, **kwargs
    )

    return zap_dictionary
