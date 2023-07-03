from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
from typing import Union, NamedTuple, Callable, Type, Any, Optional, Mapping
from functools import partial
from enum import Enum

import diffpy
from diffpy.structure import Structure as DiffpyStructure

try:
    from ase import Atoms as ASEStructure
    from ase.io import read as ase_read
except ImportError:
    ASEStructure = None
    ase_read = None


AtomStructure = Union[DiffpyStructure, ASEStructure]


class StructureBackend(ABC):
    @abstractclassmethod
    def structure_type(cls) -> Type[AtomStructure]:
        pass

    @abstractclassmethod
    def load_from_file(cls, filename: str, **kwargs: Any) -> AtomStructure:
        pass


class DiffpyBackend(StructureBackend):

    @classmethod
    def structure_type(cls) -> Type[DiffpyStructure]:
        return DiffpyStructure

    @classmethod
    def load_from_file(cls, filename: str, **kwargs: Any) -> DiffpyStructure:
        return diffpy.structure.loadStructure(filename, **kwargs)


class ASEBackend(StructureBackend):

    @classmethod
    def structure_type(cls) -> Type[ASEStructure]:
        if ASEStructure:
            return ASEStructure
        else:
            return cls._handle_missing_ase()

    @classmethod
    def load_from_file(
        cls,
        filename: str,
        **kwargs: Any,
    ) -> Optional[DiffpyStructure]:
        if ase_read:
            return ase_read(filename, **kwargs)
        else:
            return cls._handle_missing_ase()

    @classmethod
    def _handle_missing_ase(cls):
        raise ImportError("ASE was not found, please install with pip")


class Structure:

    _backends: Mapping[str, Type[StructureBackend]] = {
        "diffpy": DiffpyBackend,
        "ase": ASEBackend,
    }

    def __str__(self):
        return self.backend.__str__()

    def __init__(self, structure: AtomStructure):
        self.__structure = structure

    @property
    def backend(self) -> AtomStructure:
        return self.__structure

    @classmethod
    def from_file(
        cls,
        file_path: str,
        backend: str = "diffpy",
        **kwargs: Any,
    ) -> Structure:
        structure_backend = cls._backends[backend]
        atomic_structure = structure_backend.load_from_file(
            filename=file_path,
            **kwargs,
        )
        return cls(atomic_structure)
