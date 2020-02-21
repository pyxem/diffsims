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


def evaluate_aberration_function(kx, ky,
                                 C1, C12a, C12b, C23a, C23b, C21a, C21b,
                                 C3, C34a, C34b, C32a, C32b):
    """
    Evaluates lens aberration function for particular k and aberration
    coefficients.

    Parameters
    ----------
    kx : float
        x-coordinate in reciprocal angstroms.
    ky : float
        y-coordinate in reciprocal angstroms.
    C1 : float
        defocus.
    C12a : float
        2 stig
    C12b : float
        2 stig
    C23a : float
        3 stig
    C23b : float
        3 stig
    C21a : float
        coma
    C21b : float
        coma
    C3 : float
        Spherical abb
    C34a : float
        4 stig
    C34b : float
        4 stig
    C32a : float
        star
    C32b : float
        star

    Returns
    -------
    func_aberr : float
        in unit of meter*radian.  multiply by 2pi/lambda to get dimensionless

    """
    u2 = u*u
    u3 = u2*u
    u4 = u3*u

    v2 = v*v
    v3 = v2*v
    v4 = v3*v

    func_aberr =  1/2*C1*(u2+v2)\
            + 1/2*(C12a*(u2-v2) + 2*C12b*u*v)\
            + 1/3*(C23a*(u3-3*u*v2) + C23b*(3*u2*v - v3))\
            + 1/3*(C21a*(u3+u*v2) + C21b*(v3+u2*v))\
            + 1/4* C3*(u4+v4+2*u2*v2)\
            + 1/4* C34a*(u4-6*u2*v2+v4)\
            + 1/4* C34b*(4*u3*v-4*u*v3)\
            + 1/4* C32a*(u4-v4)\
            + 1/4* C32b*(2*u3*v + 2*u*v3)\

    return func_aberr
