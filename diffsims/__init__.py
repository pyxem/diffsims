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

import glob
import logging
import os
import warnings

import numpy as np

#from .sims.diffraction_simulation import DiffractionSimulation

from .generators.diffraction_generator import DiffractionGenerator
from .generators.library_generator import DiffractionLibraryGenerator

_logger = logging.getLogger(__name__)


def load(filename):
    """
    An extremely thin wrapper around hyperspy's load function

    Parameters
    ----------
    filename : str
        A single filename of a previously saved diffsims object. Other arguments may
        succeed, but will have fallen back on hyperspy load and warn accordingly
    *args :
        args to be passed to hyperspy's load function
    **kwargs :
        kwargs to be passed to hyperspy's load function
    """
    pass
