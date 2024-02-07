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

from typing import Union, Sequence, TYPE_CHECKING, Any
import copy

import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.quaternion import Rotation
from orix.vector import Vector3d

from diffsims.crystallography.reciprocal_lattice_vector import ReciprocalLatticeVector
from diffsims.pattern.detector_functions import add_shot_and_point_spread

# to avoid circular imports
if TYPE_CHECKING:  # pragma: no cover
    from diffsims.generators.simulation_generator import SimulationGenerator


class PhaseGetter:
    """A class for getting the phases of a simulation library.

    Parameters
    ----------
    phases : Sequence[Phase]
        The phases in the library.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def __getitem__(self, item):
        all_phases = self.simulation.phases
        if isinstance(all_phases, Phase):
            raise ValueError("Only one phase in the simulation")
        elif isinstance(item, str):
            ind = [phase.name for phase in all_phases].index(item)
        elif isinstance(item, (int, slice)):
            ind = item
        else:
            raise ValueError("Item must be a string or integer")
        new_coords = self.simulation.coordinates[ind]
        new_rotations = self.simulation.rotations[ind]
        new_phases = all_phases[ind]
        return Simulation2D(
            phases=new_phases,
            coordinates=new_coords,
            rotations=new_rotations,
            simulation_generator=self.simulation.simulation_generator,
        )


class RotationGetter:
    """A class for getting a Rotation of a simulation library.

    Parameters
    ----------
    phases : Sequence[Phase]
        The phases in the library.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def __getitem__(self, item):
        all_phases = self.simulation.phases
        if self.simulation.current_size == 1:
            raise ValueError("Only one rotation in the simulation")
        elif isinstance(all_phases, Phase):  # only one phase in the simulation
            coords = self.simulation.coordinates[item]
            phases = self.simulation.phases
            rotations = self.simulation.rotations[item]
        else:  # multiple phases in the simulation
            coords = [c[item] for c in self.simulation.coordinates]
            phases = self.simulation.phases
            rotations = [rot[item] for rot in self.simulation.rotations]
        return Simulation2D(
            phases=phases,
            coordinates=coords,
            rotations=rotations,
            simulation_generator=self.simulation.simulation_generator,
        )


class Simulation2D:
    """Holds the result of a kinematic diffraction simulation for some phase
    and rotation. This class is iterable and can be used to iterate through
    simulations of different phases and rotations.
    """

    def __init__(
        self,
        phases: Sequence[Phase],
        coordinates: Union[
            Sequence[ReciprocalLatticeVector],
            Sequence[Sequence[ReciprocalLatticeVector]],
        ],
        rotations: Union[Rotation, Sequence[Rotation]],
        simulation_generator: "SimulationGenerator",
        reciprocal_radius: float = 1.0,
    ):
        """Initializes the DiffractionSimulation object with data values for
        the coordinates, indices, intensities, calibration and offset.

        Parameters
        ----------
        coordinates
            The list of ReciprocalLatticeVector objects for each phase and rotation. If there
            are multiple phases, then this should be a list of lists of ReciprocalLatticeVector objects.
            If there is only one phase, then this should be a list of ReciprocalLatticeVector objects.
        rotations
            The list of Rotation objects for each phase. If there are multiple phases, then this should
            be a list of Rotation objects. If there is only one phase, then this should be a single
            Rotation object.
        phases
            The list of Phase objects for each phase. If there is only one phase, then this should be
            a single Phase object.
        simulation_generator
            The SimulationGenerator object used to generate the diffraction patterns.

        """
        # Basic data
        if isinstance(rotations, Rotation) and rotations.size == 1:
            if not isinstance(coordinates, ReciprocalLatticeVector):
                raise ValueError(
                    "If there is only one rotation, then the coordinates must be a ReciprocalLatticeVector object"
                )
        elif isinstance(rotations, Rotation):
            coordinates = np.array(coordinates, dtype=object)
            if coordinates.size != rotations.size:
                raise ValueError(
                    f"The number of rotations: {rotations.size} must match the number of "
                    f"coordinates {coordinates.size}"
                )
        else:  # iterable of Rotation
            rotations = np.array(rotations, dtype=object)
            coordinates = np.array(coordinates, dtype=object)
            if len(coordinates.shape) != 2:
                coordinates = coordinates[:, np.newaxis]
            phases = np.array(phases)
            if rotations.size != phases.size:
                raise ValueError(
                    f"The number of rotations: {rotations.size} must match the number of "
                    f"phases {phases.size}"
                )

            for r, c in zip(rotations, coordinates):
                if r.size != c.size:
                    raise ValueError(
                        f"The number of rotations: {r.size} must match the number of "
                        f"coordinates {c.shape[0]}"
                    )
        self.phases = phases
        self.rotations = rotations
        self.coordinates = coordinates
        self.simulation_generator = simulation_generator

        # for interactive plotting and iterating through the Simulations
        self.phase_index = 0
        self.rotation_index = 0
        self._rot_plot = None
        self._diff_plot = None
        self.reciporical_radius = reciprocal_radius

        # for slicing a simulation
        self.iphase = PhaseGetter(self)
        self.irot = RotationGetter(self)

    def __iter__(self):
        return self

    def __next__(self):
        if self.phase_index == self.num_phases:
            self.phase_index = 0
            raise StopIteration
        else:
            if self.has_multiple_phases:
                coords = self.coordinates[self.phase_index]
            else:
                coords = self.coordinates
            if self.has_multiple_rotations:
                coords = coords[self.rotation_index]
            else:
                coords = coords
            if self.rotation_index + 1 == self.current_size:
                self.rotation_index = 0
                self.phase_index += 1
            else:
                self.rotation_index += 1
            return coords

    @property
    def current_size(self):
        """Returns the number of rotations in the current phase"""
        if self.has_multiple_phases:
            return self.rotations[self.phase_index].size
        else:
            return self.rotations.size

    def deepcopy(self):
        return copy.deepcopy(self)

    def _get_transformed_coordinates(
        self,
        angle: float,
        center: Sequence = (0, 0),
        mirrored: bool = False,
        units: str = "real",
        calibration: float = None,
    ):
        """Translate, rotate or mirror the pattern spot coordinates"""

        coords = self.get_current_coordinates()

        if units != "real":
            center = np.array(center)
            coords.data = coords.data / calibration
        transformed_coords = coords
        cx, cy = center
        x = transformed_coords.data[:, 0]
        y = transformed_coords.data[:, 1]
        mirrored_factor = -1 if mirrored else 1
        theta = mirrored_factor * np.arctan2(y, x) + np.deg2rad(angle)
        rd = np.sqrt(x**2 + y**2)
        transformed_coords[:, 0] = rd * np.cos(theta) + cx
        transformed_coords[:, 1] = rd * np.sin(theta) + cy
        return transformed_coords

    @property
    def current_phase(self):
        if self.has_multiple_phases:
            return self.phases[self.phase_index]
        else:
            return self.phases

    def rotate_shift_coordinates(
        self, angle: float, center: Sequence = (0, 0), mirrored: bool = False
    ):
        """Rotate, flip or shift patterns in-plane

        Parameters
        ----------
        angle
            In plane rotation angle in degrees
        center
            Center coordinate of the patterns
        mirrored
            Mirror across the x-axis
        """
        coords_new = self._get_transformed_coordinates(
            angle, center, mirrored, units="real"
        )
        return coords_new

    def polar_flatten_simulations(self):
        """Flattens the simulations into polar coordinates for use in template matching.
        The resulting arrays are of shape (n_simulations, n_spots) where n_spots is the
        maximum number of spots in any simulation.


        Returns
        -------
        r_templates, theta_templates, intensities_templates
        """

        flattened_vectors = [sim for sim in self]
        max_num_spots = max([v.size for v in flattened_vectors])

        r_templates = np.zeros((len(flattened_vectors), max_num_spots))
        theta_templates = np.zeros((len(flattened_vectors), max_num_spots))
        intensities_templates = np.zeros((len(flattened_vectors), max_num_spots))

        for i, v in enumerate(flattened_vectors):
            r, t, _ = v.to_polar()
            r_templates[i, : len(r)] = r
            theta_templates[i, : len(t)] = t
            intensities_templates[i, : len(v.intensity)] = v.intensity

        return r_templates, theta_templates, intensities_templates

    def get_diffraction_pattern(
        self,
        shape=None,
        sigma=10,
        direct_beam_position=None,
        in_plane_angle=0,
        calibration=0.01,
        mirrored=False,
    ):
        """Returns the diffraction data as a numpy array with
        two-dimensional Gaussians representing each diffracted peak. Should only
        be used for qualitative work.

        Parameters
        ----------
        shape  : tuple of ints
            The size of a side length (in pixels)
        sigma : float
            Standard deviation of the Gaussian function to be plotted (in pixels).
        direct_beam_position: 2-tuple of ints, optional
            The (x,y) coordinate in pixels of the direct beam. Defaults to
            the center of the image.
        in_plane_angle: float, optional
            In plane rotation of the pattern in degrees
        mirrored: bool, optional
            Whether the pattern should be flipped over the x-axis,
            corresponding to the inverted orientation

        Returns
        -------
        diffraction-pattern : numpy.array
            The simulated electron diffraction pattern, normalised.

        Notes
        -----
        If don't know the exact calibration of your diffraction signal using 1e-2
        produces reasonably good patterns when the lattice parameters are on
        the order of 0.5nm and the default size and sigma are used.
        """
        if direct_beam_position is None:
            direct_beam_position = (shape[1] // 2, shape[0] // 2)
        transformed = self._get_transformed_coordinates(
            in_plane_angle,
            direct_beam_position,
            mirrored,
            units="pixel",
            calibration=calibration,
        )
        in_frame = (
            (transformed.data[:, 0] >= 0)
            & (transformed.data[:, 0] < shape[1])
            & (transformed.data[:, 1] >= 0)
            & (transformed.data[:, 1] < shape[0])
        )
        spot_coords = transformed.data[in_frame].astype(int)

        spot_intens = transformed.intensity[in_frame]
        pattern = np.zeros(shape)
        # checks that we have some spots
        if spot_intens.shape[0] == 0:
            return pattern
        else:
            pattern[spot_coords[:, 0], spot_coords[:, 1]] = spot_intens
            pattern = add_shot_and_point_spread(pattern.T, sigma, shot_noise=False)
        return np.divide(pattern, np.max(pattern))

    @property
    def num_phases(self):
        """Returns the number of phases in the simulation"""
        if hasattr(self.phases, "__len__"):
            return len(self.phases)
        else:
            return 1

    @property
    def has_multiple_phases(self):
        """Returns True if the simulation has multiple phases"""
        return self.num_phases > 1

    @property
    def has_multiple_rotations(self):
        """Returns True if the simulation has multiple rotations"""
        return self.rotations.size > 1

    def get_current_coordinates(self):
        """Returns the coordinates of the current phase and rotation"""
        if self.has_multiple_phases:
            return copy.deepcopy(
                self.coordinates[self.phase_index][self.rotation_index]
            )
        elif not self.has_multiple_phases and self.has_multiple_rotations:
            return copy.deepcopy(self.coordinates[self.rotation_index])
        else:
            return copy.deepcopy(self.coordinates)

    def get_current_rotation(self):
        """Returns the matrix for the current matrix"""
        if self.has_multiple_phases:
            return copy.deepcopy(
                self.rotations[self.phase_index].to_matrix()[self.rotation_index]
            )
        else:
            return copy.deepcopy(self.rotations.to_matrix()[self.rotation_index])

    def plot_rotations(self, beam_direction: Vector3d = Vector3d.zvector()):
        """Plots the rotations of the current phase in stereographic projection"""
        if self.has_multiple_phases:
            rots = self.rotations[self.phase_index]
        else:
            rots = self.rotations
        vect_rot = rots * beam_direction
        facecolor = ["k"] * rots.size
        facecolor[self.rotation_index] = "r"  # highlight the current rotation
        fig = vect_rot.scatter(
            grid=True,
            facecolor=facecolor,
            return_figure=True,
        )
        pointer = vect_rot[self.rotation_index]
        _plot = fig.axes[0]
        _plot.scatter(pointer.data[0][0], pointer.data[0][1], color="r")
        _plot = fig.axes[0]
        _plot.set_title("Rotations" + self.current_phase.name)

    def plot(
        self,
        size_factor=1,
        direct_beam_position=None,
        in_plane_angle=0,
        mirrored=False,
        units="real",
        show_labels=False,
        label_offset=(0, 0),
        label_formatting=None,
        min_label_intensity=0.1,
        include_direct_beam=True,
        calibration=0.1,
        ax=None,
        **kwargs,
    ):
        """A quick-plot function for a simulation of spots

        Parameters
        ----------
        size_factor : float, optional
            linear spot size scaling, default to 1
        direct_beam_position: 2-tuple of ints, optional
            The (x,y) coordinate in pixels of the direct beam. Defaults to
            the center of the image.
        in_plane_angle: float, optional
            In plane rotation of the pattern in degrees
        mirrored: bool, optional
            Whether the pattern should be flipped over the x-axis,
            corresponding to the inverted orientation
        units : str, optional
            'real' or 'pixel', only changes scalebars, falls back on 'real', the default
        show_labels : bool, optional
            draw the miller indices near the spots
        label_offset : 2-tuple, optional
            the relative location of the spot labels. Does nothing if `show_labels`
            is False.
        label_formatting : dict, optional
            keyword arguments passed to `ax.text` for drawing the labels. Does
            nothing if `show_labels` is False.
        min_label_intensity : float, optional
            minimum intensity for a spot to be labelled
        include_direct_beam : bool, optional
            whether to include the direct beam in the plot
        ax : matplotlib Axes, optional
            axes on which to draw the pattern. If `None`, a new axis is created
        **kwargs :
            passed to ax.scatter() method

        Returns
        -------
        ax,sp

        Notes
        -----
        spot size scales with the square root of the intensity.
        """
        if label_formatting is None:
            label_formatting = {}
        if direct_beam_position is None:
            direct_beam_position = (0, 0)
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal")
        coords = self._get_transformed_coordinates(
            in_plane_angle,
            direct_beam_position,
            mirrored,
            units=units,
            calibration=calibration,
        )
        if include_direct_beam:
            spots = coords.data[:, :2]
            spots = np.concatenate((spots, np.array([direct_beam_position])))
            intensity = np.concatenate((coords.intensity, np.array([1])))
        else:
            spots = coords.data[:, :2]
            intensity = coords.intensity
        sp = ax.scatter(
            spots[:, 0],
            spots[:, 1],
            s=size_factor * np.sqrt(intensity),
            **kwargs,
        )
        ax.set_xlim(-self.reciporical_radius, self.reciporical_radius)
        ax.set_ylim(-self.reciporical_radius, self.reciporical_radius)

        if show_labels:
            millers = np.round(
                np.matmul(
                    np.matmul(coords.data, self.get_current_rotation().T),
                    coords.phase.structure.lattice.base.T,
                )
            ).astype(np.int16)
            # only label the points inside the axes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            condition = (
                (coords.data[:, 0] > min(xlim))
                & (coords.data[:, 0] < max(xlim))
                & (coords.data[:, 1] > min(ylim))
                & (coords.data[:, 1] < max(ylim))
            )
            millers = millers[condition]
            coords = coords.data[condition]
            # default alignment options
            if (
                "ha" not in label_offset
                and "horizontalalignment" not in label_formatting
            ):
                label_formatting["ha"] = "center"
            if "va" not in label_offset and "verticalalignment" not in label_formatting:
                label_formatting["va"] = "center"
            for miller, coordinate, inten in zip(millers, coords, intensity):
                if np.isnan(inten) or inten > min_label_intensity:
                    label = "("
                    for index in miller:
                        if index < 0:
                            label += r"$\bar{" + str(abs(index)) + r"}$"
                        else:
                            label += str(abs(index))
                        label += " "
                    label = label[:-1] + ")"
                    ax.text(
                        coordinate[0] + label_offset[0],
                        coordinate[1] + label_offset[1],
                        label,
                        **label_formatting,
                    )
            if units == "real":
                ax.set_xlabel(r"$\AA^{-1}$")
                ax.set_ylabel(r"$\AA^{-1}$")
            else:
                ax.set_xlabel("pixels")
                ax.set_ylabel("pixels")
        return ax, sp
