# -*- coding: utf-8 -*-
# Copyright 2017-2021 The diffsims developers
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

import matplotlib.pyplot as plt
import numpy as np
from diffsims.pattern.detector_functions import add_shot_and_point_spread
from diffsims.utils import mask_utils


class DiffractionSimulation:
    """Holds the result of a kinematic diffraction pattern simulation.

    Parameters
    ----------
    coordinates : array-like, shape [n_points, 2]
        The x-y coordinates of points in reciprocal space.
    indices : array-like, shape [n_points, 3]
        The indices of the reciprocal lattice points that intersect the
        Ewald sphere.
    intensities : array-like, shape [n_points, ]
        The intensity of the reciprocal lattice points.
    calibration : float or tuple of float, optional
        The x- and y-scales of the pattern, with respect to the original
        reciprocal angstrom coordinates.
    offset : tuple of float, optional
        The x-y offset of the pattern in reciprocal angstroms. Defaults to
        zero in each direction.
    """

    def __init__(
        self,
        coordinates,
        indices=None,
        intensities=None,
        calibration=1.0,
        offset=(0.0, 0.0),
        with_direct_beam=False,
    ):
        """Initializes the DiffractionSimulation object with data values for
        the coordinates, indices, intensities, calibration and offset.
        """
        if indices is None:
            indices = np.zeros((coordinates.shape[0], 3))
        if intensities is None:
            intensities = np.zeros((coordinates.shape[0]))
        # check here whether shapes are all the same
        if coordinates.shape[0] == indices.shape[0] == intensities.shape[0]:
            self._coordinates = coordinates
            self._indices = indices
            self._intensities = intensities
        else:
            raise ValueError("Coordinate, intensity, and indices lists must be same size.")
        self.calibration = calibration
        self.offset = offset
        self.with_direct_beam = with_direct_beam

    @property
    def indices(self):
        return self._indices[self.direct_beam_mask]

    @indices.setter
    def indices(self, indices):
        self._indices[self.direct_beam_mask] = indices

    @property
    def calibrated_coordinates(self):
        """ndarray : Coordinates converted into pixel space."""
        return (self.coordinates[:, :2] + np.array(self.offset))/np.array(self.calibration)

    @property
    def calibration(self):
        """tuple of float : The x- and y-scales of the pattern, with respect to
        the original reciprocal angstrom coordinates."""
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        if np.all(np.equal(calibration, 0)):
            raise ValueError("`calibration` cannot be zero.")
        if isinstance(calibration, float) or isinstance(calibration, int):
            self._calibration = (calibration, calibration)
        elif len(calibration) == 2:
            self._calibration = calibration
        else:
            raise ValueError(
                "`calibration` must be a float or length-2" "tuple of floats."
            )

    @property
    def direct_beam_mask(self):
        """ndarray : If `with_direct_beam` is True, returns a True array for all
        points. If `with_direct_beam` is False, returns a True array with False
        in the position of the direct beam."""
        if self.with_direct_beam:
            return np.ones_like(self._intensities, dtype=bool)
        else:
            return np.any(self._coordinates, axis=1)

    @property
    def coordinates(self):
        """ndarray : The coordinates of all unmasked points."""
        return self._coordinates[self.direct_beam_mask]

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates[self.direct_beam_mask] = coordinates

    @property
    def intensities(self):
        """ndarray : The intensities of all unmasked points."""
        return self._intensities[self.direct_beam_mask]

    @intensities.setter
    def intensities(self, intensities):
        self._intensities[self.direct_beam_mask] = intensities

    def get_as_mask(self, shape, radius=6., negative=True,
                    radius_function=None, direct_beam_position=None,
                    in_plane_angle=0, mirrored=False,
                    *args, **kwargs):
        """
        Return the diffraction pattern as a binary mask of type
        bool

        Parameters
        ----------
        shape: 2-tuple of ints
            Shape of the output mask (width, height)
        radius: float or array, optional
            Radii of the spots in pixels. An array may be supplied
            of the same length as the number of spots.
        negative: bool, optional
            Whether the spots are masked (True) or everything
            else is masked (False)
        radius_function: Callable, optional
            Calculate the radius as a function of the spot intensity,
            for example np.sqrt. args and kwargs supplied to this method
            are passed to this function. Will override radius.
        direct_beam_position: 2-tuple of ints, optional
            The (x,y) coordinate in pixels of the direct beam. Defaults to
            the center of the image.
        in_plane_angle: float, optional
            In plane rotation of the pattern in degrees
        mirrored: bool, optional
            Whether the pattern should be flipped over the x-axis,
            corresponding to the inverted orientation
        """
        r = radius
        cx, cy = shape[0]//2, shape[1]//2
        if direct_beam_position is not None:
            cx, cy = direct_beam_position
        point_coordinates_shifted = self.calibrated_coordinates.copy()
        x = point_coordinates_shifted[:, 0]
        y = point_coordinates_shifted[:, 1]
        mirrored_factor = -1 if mirrored else 1
        theta = mirrored_factor * np.arctan2(y, x) + np.deg2rad(in_plane_angle)
        rd = np.sqrt(x**2 + y**2)
        point_coordinates_shifted[:, 0] = rd * np.cos(theta) + cx
        point_coordinates_shifted[:, 1] = rd * np.sin(theta) + cy
        if radius_function is not None:
            r = radius_function(self.intensities, *args, **kwargs)
        mask = mask_utils.create_mask(shape, fill=negative)
        mask_utils.add_circles_to_mask(mask, point_coordinates_shifted, r,
                                    fill=not negative)
        return mask

    def get_diffraction_pattern(self, size=512, sigma=10):
        """Returns the diffraction data as a numpy array with
        two-dimensional Gaussians representing each diffracted peak. Should only
        be used for qualitative work.

        Parameters
        ----------
        size  : int
            The size of a side length (in pixels)

        sigma : float
            Standard deviation of the Gaussian function to be plotted (in pixels).

        Returns
        -------
        diffraction-pattern : numpy.array
            The simulated electron diffraction pattern, normalised.

        Notes
        -----
        If don't know the exact calibration of your diffraction signal using 1e-2
        produces reasonably good patterns when the lattice parameters are on
        the order of 0.5nm and a the default size and sigma are used.
        """
        side_length = np.min(np.multiply((size / 2), self.calibration))
        mask_for_sides = np.all(
            (np.abs(self.coordinates[:, 0:2]) < side_length), axis=1
        )

        spot_coords = np.add(
            self.calibrated_coordinates[mask_for_sides], size / 2
        ).astype(int)
        spot_intens = self.intensities[mask_for_sides]
        pattern = np.zeros([size, size])
        # checks that we have some spots
        if spot_intens.shape[0] == 0:
            return pattern
        else:
            pattern[spot_coords[:, 0], spot_coords[:, 1]] = spot_intens
            pattern = add_shot_and_point_spread(pattern.T, sigma, shot_noise=False)

        return np.divide(pattern, np.max(pattern))

    def plot(self, size_factor=1, units="real", show_labels=False,
            label_offset=(0, 0),
            label_formatting={},
            ax=None,
            **kwargs):
        """A quick-plot function for a simulation of spots

        Parameters
        ----------
        size_factor : float, optional
            linear spot size scaling, default to 1
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
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal")
        if units == "pixel":
            coords = self.calibrated_coordinates
        else:
            coords = self.coordinates

        sp = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=size_factor * np.sqrt(self.intensities),
            **kwargs
        )

        if show_labels:
            millers = self.indices.astype(np.int16)
            # only label the points inside the axes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            condition = ((coords[:,0] > min(xlim)) &
                         (coords[:,0] < max(xlim)) &
                         (coords[:,1] > min(ylim)) &
                         (coords[:,1] < max(ylim)))
            millers = millers[condition]
            coords = coords[condition]
            # default alignment options
            if "ha" not in label_offset and "horizontalalignment" not in label_formatting:
                label_formatting["ha"]="center"
            if "va" not in label_offset and "verticalalignment" not in label_formatting:
                label_formatting["va"]="center"
            for miller, coordinate in zip(millers, coords):
                label = "("
                for index in miller:
                    if index<0:
                        label += r"$\bar{" + str(abs(index)) +r"}$"
                    else:
                        label += str(abs(index))
                    label += " "
                label = label[:-1] + ")"
                ax.text(coordinate[0] + label_offset[0],
                        coordinate[1] + label_offset[1],
                        label,
                        **label_formatting,
                        )
        return ax, sp


class ProfileSimulation:
    """Holds the result of a given kinematic simulation of a diffraction
    profile.

    Parameters
    ----------
    magnitudes : array-like, shape [n_peaks, 1]
        Magnitudes of scattering vectors.
    intensities : array-like, shape [n_peaks, 1]
        The kinematic intensity of the diffraction peaks.
    hkls : [{(h, k, l): mult}] {(h, k, l): mult} is a dict of Miller
        indices for all diffracted lattice facets contributing to each
        intensity.
    """

    def __init__(self, magnitudes, intensities, hkls):
        self.magnitudes = magnitudes
        self.intensities = intensities
        self.hkls = hkls

    def get_plot(self, annotate_peaks=True, with_labels=True, fontsize=12):
        """Plots the diffraction profile simulation for the
           calculate_profile_data method in DiffractionGenerator.

        Parameters
        ----------
        annotate_peaks : boolean
            If True, peaks are annotaed with hkl information.
        with_labels : boolean
            If True, xlabels and ylabels are added to the plot.
        fontsize : integer
            Fontsize for peak labels.
        """

        ax = plt.gca()
        for g, i, hkls in zip(self.magnitudes, self.intensities, self.hkls):
            label = hkls
            ax.plot([g, g], [0, i], color="k", linewidth=3, label=label)
            if annotate_peaks:
                ax.annotate(label, xy=[g, i], xytext=[g, i], fontsize=fontsize)

            if with_labels:
                ax.set_xlabel("A ($^{-1}$)")
                ax.set_ylabel("Intensities (scaled)")

        return plt
