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

"""Calculation of scattering factors and structure factors."""

from diffsims.structure_factor.atomic_scattering_factor import (
    get_doyleturner_atomic_scattering_factor,
    get_kinematical_atomic_scattering_factor,
)
from diffsims.structure_factor.atomic_scattering_parameters import (
    get_atomic_scattering_parameters,
    get_element_id_from_string,
)
from diffsims.structure_factor.structure_factor import (
    find_asymmetric_positions,
    get_doyleturner_structure_factor,
    get_kinematical_structure_factor,
    get_refraction_corrected_wavelength,
)

__all__ = [
    "get_doyleturner_atomic_scattering_factor",
    "get_kinematical_atomic_scattering_factor",
    "get_atomic_scattering_parameters",
    "get_element_id_from_string",
    "find_asymmetric_positions",
    "get_doyleturner_structure_factor",
    "get_kinematical_structure_factor",
    "get_refraction_corrected_wavelength",
]
