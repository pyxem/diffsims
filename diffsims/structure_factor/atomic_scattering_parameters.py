# -*- coding: utf-8 -*-
# Copyright 2017-2025 The diffsims developers
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

# List of elements Z = 1-118
# fmt: off
ELEMENTS = [
    "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al",
    "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn",
    "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb",
    "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in",
    "sn", "sb", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd", "pm",
    "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta",
    "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at",
    "rn", "fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk",
    "cf", "es", "fm", "md", "no", "lr", "rf", "db", "sg", "bh", "hs", "mt",
    "ds", "rg", "cn", "nh", "fl", "mc", "lv", "ts", "og"
]
# fmt: on
N_ELEMENTS = len(ELEMENTS)

# fmt: off
ATOMIC_SCATTERING_PARAMETERS_DOYLETURNER = np.array([
    # a1      a2     a3     a4      b1     b2     b3     b4
    0.202, 0.244, 0.082, 0.000, 30.868, 8.544, 1.273, 0.000,  # H
    0.091, 0.181, 0.110, 0.036, 18.183, 6.212, 1.803, 0.284,  # He
    1.611, 1.246, 0.326, 0.099, 107.638, 30.480, 4.533, 0.495,  # etc.
    1.250, 1.334, 0.360, 0.106, 60.804, 18.591, 3.653, 0.416,
    0.945, 1.312, 0.419, 0.116, 46.444, 14.178, 3.223, 0.377,
    0.731, 1.195, 0.456, 0.125, 36.995, 11.297, 2.814, 0.346,
    0.572, 1.043, 0.465, 0.131, 28.847, 9.054, 2.421, 0.317,
    0.455, 0.917, 0.472, 0.138, 23.780, 7.622, 2.144, 0.296,
    0.387, 0.811, 0.475, 0.146, 20.239, 6.609, 1.931, 0.279,
    0.303, 0.720, 0.475, 0.153, 17.640, 5.860, 1.762, 0.266,
    2.241, 1.333, 0.907, 0.286, 108.004, 24.505, 3.391, 0.435,
    2.268, 1.803, 0.839, 0.289, 73.670, 20.175, 3.013, 0.405,
    2.276, 2.428, 0.858, 0.317, 72.322, 19.773, 3.080, 0.408,
    2.129, 2.533, 0.835, 0.322, 57.775, 16.476, 2.880, 0.386,
    1.888, 2.469, 0.805, 0.320, 44.876, 13.538, 2.642, 0.361,
    1.659, 2.386, 0.790, 0.321, 36.650, 11.488, 2.469, 0.340,
    1.452, 2.292, 0.787, 0.322, 30.935, 9.980, 2.234, 0.323,
    1.274, 2.190, 0.793, 0.326, 26.682, 8.813, 2.219, 0.307,
    3.951, 2.545, 1.980, 0.482, 137.075, 22.402, 4.532, 0.434,
    4.470, 2.971, 1.970, 0.482, 99.523, 22.696, 4.195, 0.417,
    3.966, 2.917, 1.925, 0.480, 88.960, 20.606, 3.856, 0.399,
    3.565, 2.818, 1.893, 0.483, 81.982, 19.049, 3.590, 0.386,
    3.245, 2.698, 1.860, 0.486, 76.379, 17.726, 3.363, 0.374,
    2.307, 2.334, 1.823, 0.490, 78.405, 15.785, 3.157, 0.364,
    2.747, 2.456, 1.792, 0.498, 67.786, 15.674, 3.000, 0.357,
    2.544, 2.343, 1.759, 0.506, 64.424, 14.880, 2.854, 0.350,
    2.367, 2.236, 1.724, 0.515, 61.431, 14.180, 2.725, 0.344,
    2.210, 2.134, 1.689, 0.524, 58.727, 13.553, 2.609, 0.339,
    1.579, 1.820, 1.658, 0.532, 62.940, 12.453, 2.504, 0.333,
    1.942, 1.950, 1.619, 0.543, 54.162, 12.518, 2.416, 0.330,
    2.321, 2.486, 1.688, 0.599, 65.602, 15.458, 2.581, 0.351,
    2.447, 2.702, 1.616, 0.601, 55.893, 14.393, 2.446, 0.342,
    2.399, 2.790, 1.529, 0.594, 45.718, 12.817, 2.280, 0.328,
    2.298, 2.854, 1.456, 0.590, 38.830, 11.536, 2.146, 0.316,
    2.166, 2.904, 1.395, 0.589, 33.899, 10.497, 2.041, 0.307,
    2.034, 2.927, 1.342, 0.589, 29.999, 9.598, 1.952, 0.299,
    4.776, 3.859, 2.234, 0.868, 140.782, 18.991, 3.701, 0.419,
    5.848, 4.003, 2.342, 0.880, 104.972, 19.367, 3.737, 0.414,
    4.129, 3.012, 1.179, 0.000, 27.548, 5.088, 0.591, 0.0,
    4.105, 3.144, 1.229, 0.000, 28.492, 5.277, 0.601, 0.0,
    4.237, 3.105, 1.234, 0.000, 27.415, 5.074, 0.593, 0.0,
    3.120, 3.906, 2.361, 0.850, 72.464, 14.642, 3.237, 0.366,
    4.318, 3.270, 1.287, 0.000, 28.246, 5.148, 0.590, 0.0,
    4.358, 3.298, 1.323, 0.000, 27.881, 5.179, 0.594, 0.0,
    4.431, 3.343, 1.345, 0.000, 27.911, 5.153, 0.592, 0.0,
    4.436, 3.454, 1.383, 0.000, 28.670, 5.269, 0.595, 0.0,
    2.036, 3.272, 2.511, 0.837, 61.497, 11.824, 2.846, 0.327,
    2.574, 3.259, 2.547, 0.838, 55.675, 11.838, 2.784, 0.322,
    3.153, 3.557, 2.818, 0.884, 66.649, 14.449, 2.976, 0.335,
    3.450, 3.735, 2.118, 0.877, 59.104, 14.179, 2.855, 0.327,
    3.564, 3.844, 2.687, 0.864, 50.487, 13.316, 2.691, 0.316,
    4.785, 3.688, 1.500, 0.000, 27.999, 5.083, 0.581, 0.0,
    3.473, 4.060, 2.522, 0.840, 39.441, 11.816, 2.415, 0.298,
    3.366, 4.147, 2.443, 0.829, 35.509, 11.117, 2.294, 0.289,
    6.062, 5.986, 3.303, 1.096, 155.837, 19.695, 3.335, 0.379,
    7.821, 6.004, 3.280, 1.103, 117.657, 18.778, 3.263, 0.376,
    4.940, 3.968, 1.663, 0.000, 28.716, 5.245, 0.594, 0.0,
    5.007, 3.980, 1.678, 0.000, 28.283, 5.183, 0.589, 0.0,
    5.085, 4.043, 1.684, 0.000, 28.588, 5.143, 0.581, 0.0,
    5.151, 4.075, 1.683, 0.000, 28.304, 5.073, 0.571, 0.0,
    5.201, 4.094, 1.719, 0.000, 28.079, 5.081, 0.576, 0.0,
    5.255, 4.113, 1.743, 0.000, 28.016, 5.037, 0.577, 0.0,
    6.267, 4.844, 3.202, 1.200, 100.298, 16.066, 2.980, 0.367,
    5.225, 4.314, 1.827, 0.000, 29.158, 5.259, 0.586, 0.0,
    5.272, 4.347, 1.844, 0.000, 29.046, 5.226, 0.585, 0.0,
    5.332, 4.370, 1.863, 0.000, 28.888, 5.198, 0.581, 0.0,
    5.376, 4.403, 1.884, 0.000, 28.773, 5.174, 0.582, 0.0,
    5.436, 4.437, 1.891, 0.000, 28.655, 5.117, 0.577, 0.0,
    5.441, 4.510, 1.956, 0.000, 29.149, 5.264, 0.590, 0.0,
    5.529, 4.533, 1.945, 0.000, 28.927, 5.144, 0.578, 0.0,
    5.553, 4.580, 1.969, 0.000, 28.907, 5.160, 0.577, 0.0,
    5.588, 4.619, 1.997, 0.000, 29.001, 5.164, 0.579, 0.0,
    5.659, 4.630, 2.014, 0.000, 28.807, 5.114, 0.578, 0.0,
    5.709, 4.677, 2.019, 0.000, 28.782, 5.084, 0.572, 0.0,
    5.695, 4.740, 2.064, 0.000, 28.968, 5.156, 0.575, 0.0,
    5.750, 4.773, 2.079, 0.000, 28.933, 5.139, 0.573, 0.0,
    5.754, 4.851, 2.096, 0.000, 29.159, 5.152, 0.570, 0.0,
    5.803, 4.870, 2.127, 0.000, 29.016, 5.150, 0.572, 0.0,
    2.388, 4.226, 2.689, 1.255, 42.866, 9.743, 2.264, 0.307,
    2.682, 4.241, 2.755, 1.270, 42.822, 9.856, 2.295, 0.307,
    5.932, 4.972, 2.195, 0.000, 29.086, 5.126, 0.572, 0.0,
    3.510, 4.552, 3.154, 1.359, 52.914, 11.884, 2.571, 0.321,
    3.841, 4.679, 3.192, 1.363, 50.261, 11.999, 2.560, 0.318,
    6.070, 4.997, 2.232, 0.000, 28.075, 4.999, 0.563, 0.0,
    6.133, 5.031, 2.239, 0.000, 28.047, 4.957, 0.558, 0.0,
    4.078, 4.978, 3.096, 1.326, 38.406, 11.020, 2.355, 0.299,
    6.201, 5.121, 2.275, 0.000, 28.200, 4.954, 0.556, 0.0,
    6.215, 5.170, 2.316, 0.000, 28.382, 5.002, 0.562, 0.0,
    6.278, 5.195, 2.321, 0.000, 28.323, 4.949, 0.557, 0.0,
    6.264, 5.263, 2.367, 0.000, 28.651, 5.030, 0.563, 0.0,
    6.306, 5.303, 2.386, 0.000, 28.688, 5.026, 0.561, 0.0,
    6.767, 6.729, 4.014, 1.561, 85.951, 15.642, 2.936, 0.335,
    6.323, 5.414, 2.453, 0.000, 29.142, 5.096, 0.568, 0.0,
    6.415, 5.419, 2.449, 0.000, 28.836, 5.022, 0.561, 0.0,
    6.378, 5.495, 2.495, 0.000, 29.156, 5.102, 0.565, 0.0,
    6.460, 5.469, 2.471, 0.000, 28.396, 4.970, 0.554, 0.0,
    6.502, 5.478, 2.510, 0.000, 28.375, 4.975, 0.561, 0.0,
    6.548, 5.526, 2.520, 0.000, 28.461, 4.965, 0.557, 0.0,
]).reshape(98, 8)  # 1 / Å^2
# fmt: on


def get_atomic_scattering_parameters(element, unit=None):
    """Return the eight atomic scattering parameters a_1-4, b_1-4 for
    elements with atomic numbers Z = 1-98 from Table 12.1 in
    [DeGraef2007]_, which are themselves from [Doyle1968]_ and
    [Smith1962]_.

    Parameters
    ----------
    element : int or str
        Element to return scattering parameters for. Either one-two
        letter string or integer atomic number.
    unit : str, optional
        Either "nm" or "Å"/"A". Whether to return parameters in terms
        of Å^-2 or nm^-2. If None (default), Å^-2 is used.

    Returns
    -------
    a : numpy.ndarray
        The four atomic scattering parameters a_1-4.
    b : numpy.ndarray
        The four atomic scattering parameters b_1-4.

    References
    ----------
    .. [DeGraef2007] M. De Graef, M. E. McHenry, "Structure of\
        Materials," Cambridge University Press (2007).
    .. [Doyle1968] P. A. Doyle, P. S. Turner, "Relativistic Hartree-Fock
        X-ray and electron scattering factors," *Acta Cryst.* **24**
        (1968), doi: https://doi.org/10.1107/S0567739468000756.
    .. [Smith1962] G. Smith, R. Burge, "The analytical representation
        of atomic scattering amplitudes for electrons," *Acta Cryst.*
        **A15** (1962), doi: https://doi.org/10.1107/S0365110X62000481.
    """
    if isinstance(element, str):
        element_id = get_element_id_from_string(element)
    else:
        element_id = int(element)

    factor = 1  # Ångstrøm
    if unit is not None:
        if unit.lower() == "nm":
            factor = 1e-2

    a = ATOMIC_SCATTERING_PARAMETERS_DOYLETURNER[element_id - 1, :4] * factor
    b = ATOMIC_SCATTERING_PARAMETERS_DOYLETURNER[element_id - 1, 4:] * factor

    return a, b


def get_element_id_from_string(element_str):
    r"""Get periodic element ID for elements :math:`Z` = 1-98 from
    one-two letter string.

    Parameters
    ----------
    element_str : str
        One-two letter string.

    Returns
    -------
    element_id : int
        Integer ID in the periodic table of elements.
    """
    element2periodic = dict(zip(ELEMENTS[:N_ELEMENTS], np.arange(1, N_ELEMENTS)))
    element_id = element2periodic[element_str.lower()]
    return element_id


def _get_string_from_element_id(element_id):
    r"""Get one-two letter string for a periodic element ID :math:`Z`
    within 1-98.

    Parameters
    ----------
    element_id : int
        Integer ID in the periodic table of elements.

    Returns
    -------
    element_str : str
        One-two letter string :math:`Z`.
    """
    periodic2element = dict(zip(np.arange(1, N_ELEMENTS), ELEMENTS[:N_ELEMENTS]))
    element_str = periodic2element[element_id].capitalize()
    return element_str
