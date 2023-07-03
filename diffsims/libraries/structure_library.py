# -*- coding: utf-8 -*-
# Copyright 2017-2022 The diffsims developers
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

from __future__ import annotations
from typing import Sequence, NamedTuple, Mapping

from diffpy.structure import Structure
from orix.quaternion import Rotation


class StructureData(NamedTuple):
    structure: Structure
    orientations: Rotation


class StructureLibrary:
    """
    Storage container for the structures and their associated rotations
    """

    def __init__(
        self,
        names: Sequence[str] = None,
        structures: Sequence[Structure] = None,
        orientations: Sequence[Rotation] = None,
    ):
        if names is None:
            names = []
        if structures is None:
            structures = []
        if orientations is None:
            orientations = []
        if (len(names) != len(structures)) or (len(names) != len(orientations)):
            raise ValueError("All sequences must be of the same length")

        self.__data = dict()
        for name, structure, orientation in zip(names, structures, orientations):
            phase_info = StructureData(structure, orientation)
            self.__data[name] = phase_info

    def __getitem__(self, name: str) -> StructureData:
        return self.__data[name]

    def add_structure(
        self,
        name: str,
        structure: Structure,
        orientations: Rotation,
    ) -> None:
        self.__data[name] = StructureData(structure, orientations)

    @property
    def phases(self) -> Sequence[str]:
        return list(self.__data.keys())

    @property
    def orientations(self) -> Mapping[str, Rotation]:
        return {
            phase: self.__data[phase].orientations
            for phase in self.phases
        }

    @property
    def structures(self) -> Mapping[str, Structure]:
        return {
            phase: self.__data[phase].structure
            for phase in self.phases
        }

    def get_structure(self, name: str) -> Structure:
        return self.__data[name].structure

    def get_orientations(self, name: str) -> Rotation:
        return self.__data[name].orientations

    @property
    def n_structures(self) -> int:
        return len(self.__data)

    @property
    def n_orientations(self) -> Mapping[str, int]:
        return {
            phase: self.__data[phase].orientations.size
            for phase in self.phases
        }
