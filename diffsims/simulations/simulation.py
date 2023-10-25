import copy

import numpy as np
import matplotlib.pyplot as plt

from diffsims.crystallography.reciprocal_lattice_vector import ReciprocalLatticeVector
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
        coordinates: ReciprocalLatticeVector,
        calibration=None,
        offset=(0.0, 0.0),
        with_direct_beam=False,
        shape=(512, 512),
    ):
        """Initializes the DiffractionSimulation object with data values for
        the coordinates, indices, intensities, calibration and offset.
        """
        self._coordinates = coordinates
        self.shape = shape
        self.calibration = calibration
        self.offset = np.array(offset)
        self.with_direct_beam = with_direct_beam

    def __len__(self):
        return self.coordinates.shape[0]

    @property
    def size(self):
        return self.__len__()

    def __getitem__(self, sliced):
        """Sliced is any valid numpy slice that does not change the number of
        dimensions or number of columns"""
        coords = self.coordinates[sliced]
        return DiffractionSimulation(
            coords,
            calibration=self.calibration,
            offset=self.offset,
            with_direct_beam=self.with_direct_beam,
        )

    def deepcopy(self):
        return copy.deepcopy(self)

    def append(self, vectors: ReciprocalLatticeVector):
        new_data = copy.deepcopy(self)
        new_coords = np.concatenate(
            (new_data._coordinates.data, vectors._coordinates.data), axis=0
        )
        new_data._coordinates = ReciprocalLatticeVector(
            phase=self._coordinates.phase, xyz=new_coords
        )
        new_data._coordinates.intensity = np.concatenate(
            (self._coordinates.intensity, vectors._coordinates.intensity)
        )
        return new_data

    @property
    def calibrated_coordinates(self):
        """ndarray : Coordinates converted into pixel space."""
        if self.calibration is not None:
            return (self.coordinates.data[:, :2] + self.offset) / self.calibration
        else:
            raise Exception("Pixel calibration is not set!")

    @property
    def pixel_coordinates(self):
        half_shape = np.array(self.shape) / 2
        pixel_coordinates = np.rint(
            self.calibrated_coordinates[:, :2] + half_shape
        ).astype(int)
        return pixel_coordinates

    @property
    def calibration(self):
        """tuple of float : The x- and y-scales of the pattern, with respect to
        the original reciprocal angstrom coordinates."""
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        if calibration is None:
            pass
        elif np.all(np.equal(calibration, 0)):
            raise ValueError("`calibration` cannot be zero.")
        elif isinstance(calibration, float) or isinstance(calibration, int):
            calibration = np.array((calibration, calibration))
        elif len(calibration) == 2:
            calibration = np.array(calibration)
        else:
            raise ValueError(
                "`calibration` must be a float or length-2" "tuple of floats."
            )
        self._calibration = calibration

    def get_polar_coordinates(self, real=True):
        """Returns the polar coordinates of the diffraction pattern
        """
        x = self.coordinates.data[:, 0]
        y = self.coordinates.data[:, 1]
        if not real:
            x = x / self.calibration[0]
            y = y / self.calibration[1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

    @property
    def direct_beam_mask(self):
        """ndarray : If `with_direct_beam` is True, returns a True array for all
        points. If `with_direct_beam` is False, returns a True array with False
        in the position of the direct beam."""
        if self.with_direct_beam:
            return np.ones_like(self._coordinates.intensity, dtype=bool)
        else:
            mask = np.any(self._coordinates.data != 0, axis=1)
            return mask

    @property
    def coordinates(self):
        """ndarray : The coordinates of all unmasked points."""
        return self._coordinates[self.direct_beam_mask]

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates

    @property
    def intensities(self):
        return self.coordinates.intensity

    @intensities.setter
    def intensities(self, intensities):
        self._coordinates.intensity = intensities
        print(self.coordinates.intensity)

    def _get_transformed_coordinates(
        self, angle, center=(0, 0), mirrored=False, units="real"
    ):
        """Translate, rotate or mirror the pattern spot coordinates"""
        if units != "real":
            center = np.array(center) / self.calibration
        new_sim = self.deepcopy()
        transformed_coords = new_sim.coordinates
        cx, cy = center
        x = transformed_coords.data[:, 0]
        y = transformed_coords.data[:, 1]
        mirrored_factor = -1 if mirrored else 1
        theta = mirrored_factor * np.arctan2(y, x) + np.deg2rad(angle)
        rd = np.sqrt(x**2 + y**2)
        transformed_coords[:, 0] = rd * np.cos(theta) + cx
        transformed_coords[:, 1] = rd * np.sin(theta) + cy
        new_sim._coordinates = transformed_coords
        return new_sim

    def rotate_shift_coordinates(self, angle, center=(0, 0), mirrored=False):
        """
        Rotate, flip or shift patterns in-plane

        Parameters
        ----------
        angle: float
            In plane rotation angle in degrees
        center: 2-tuple of floats
            Center coordinate of the patterns
        mirrored: bool
            Mirror across the x axis
        """
        coords_new = self._get_transformed_coordinates(
            angle, center, mirrored, units="real"
        )
        return coords_new

    def get_as_mask(
        self,
        shape,
        radius=6.0,
        negative=True,
        radius_function=None,
        direct_beam_position=None,
        in_plane_angle=0,
        mirrored=False,
        *args,
        **kwargs,
    ):
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

        Returns
        -------
        mask: numpy.ndarray
            Boolean mask of the diffraction pattern
        """
        r = radius
        if direct_beam_position is None:
            direct_beam_position = (shape[1] // 2, shape[0] // 2)
        point_coordinates_shifted = self._get_transformed_coordinates(
            in_plane_angle,
            center=direct_beam_position,
            mirrored=mirrored,
            units="pixels",
        )
        if radius_function is not None:
            r = radius_function(self.intensities, *args, **kwargs)
        mask = mask_utils.create_mask(shape, fill=negative)
        mask_utils.add_circles_to_mask(
            mask, point_coordinates_shifted.coordinates.data, r, fill=not negative
        )
        return mask

    def get_diffraction_pattern(
        self,
        shape=None,
        sigma=10,
        direct_beam_position=None,
        in_plane_angle=0,
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
        the order of 0.5nm and a the default size and sigma are used.
        """
        if shape is None:
            shape = self.shape
        if direct_beam_position is None:
            direct_beam_position = (shape[1] // 2, shape[0] // 2)
        tranformed = self._get_transformed_coordinates(
            in_plane_angle, direct_beam_position, mirrored, units="pixel"
        )
        in_frame = (
            (tranformed.coordinates.data[:, 0] >= 0)
            & (tranformed.coordinates.data[:, 0] < shape[1])
            & (tranformed.coordinates.data[:, 1] >= 0)
            & (tranformed.coordinates.data[:, 1] < shape[0])
        )
        spot_coords = tranformed.coordinates.data[in_frame].astype(int)
        spot_intens = self.intensities[in_frame]
        pattern = np.zeros(shape)
        # checks that we have some spots
        if spot_intens.shape[0] == 0:
            return pattern
        else:
            pattern[spot_coords[:, 0], spot_coords[:, 1]] = spot_intens
            pattern = add_shot_and_point_spread(pattern.T, sigma, shot_noise=False)
        return np.divide(pattern, np.max(pattern))

    def plot(
        self,
        size_factor=1,
        direct_beam_position=None,
        in_plane_angle=0,
        mirrored=False,
        units="real",
        show_labels=False,
        label_offset=(0, 0),
        label_formatting={},
        min_label_intensity=0.1,
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
        if direct_beam_position is None:
            direct_beam_position = (0, 0)
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal")
        coords = self._get_transformed_coordinates(
            in_plane_angle, direct_beam_position, mirrored, units=units
        )
        sp = ax.scatter(
            coords.coordinates.data[:, 0],
            coords.coordinates.data[:, 1],
            s=size_factor * np.sqrt(self.intensities),
            **kwargs,
        )

        if show_labels:
            millers = self.coordinates.hkl.astype(np.int16)
            # only label the points inside the axes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            condition = (
                (coords.coordinates.data[:, 0] > min(xlim))
                & (coords.coordinates.data[:, 0] < max(xlim))
                & (coords.coordinates.data[:, 1] > min(ylim))
                & (coords.coordinates.data[:, 1] < max(ylim))
            )
            millers = millers[condition]
            coords = coords.coordinates.data[condition]
            # default alignment options
            if (
                "ha" not in label_offset
                and "horizontalalignment" not in label_formatting
            ):
                label_formatting["ha"] = "center"
            if "va" not in label_offset and "verticalalignment" not in label_formatting:
                label_formatting["va"] = "center"
            for miller, coordinate, inten in zip(millers, coords, self.intensities):
                if inten > min_label_intensity:
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

    def plot(self, annotate_peaks=True, with_labels=True, fontsize=12):
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
