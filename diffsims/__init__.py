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

import logging
import os
import warnings

import numpy as np

from .generators.diffraction_generator import DiffractionGenerator, AtomicDiffractionGenerator
from .generators.library_generator import DiffractionLibraryGenerator
from .generators.library_generator import VectorLibraryGenerator

from .sims.diffraction_simulation import DiffractionSimulation

from .utils.atomic_diffraction_generator_support.probe_utils import ProbeFunction, BesselProbe
from .utils.atomic_diffraction_generator_support.fourier_transform import to_recip, from_recip, get_recip_points, get_DFT
from .utils.atomic_diffraction_generator_support.discretise_utils import get_discretisation

from . import release_info

__version__ = release_info.version
__author__ = release_info.author
__copyright__ = release_info.copyright
__credits__ = release_info.credits
__license__ = release_info.license
__maintainer__ = release_info.maintainer
__email__ = release_info.email
__status__ = release_info.status

_logger = logging.getLogger(__name__)
