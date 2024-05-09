"""
0.5.x --> 0.6.x Migration Guide
===============================
This is a migration guide for version 0.5.x to 0.6.x. This guide helps to show the changes
that were made to the API and how to update your code to use the new API.

Here you can see how to make an equivalent to a diffraction library

Old
---
"""

import numpy as np
import matplotlib.pyplot as plt
from diffpy.structure import Atom, Lattice, Structure

from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator


latt = Lattice(4, 4, 4, 90, 90, 90)
atoms = [
    Atom(atype="Al", xyz=[0.0, 0.0, 0.0], lattice=latt),
    Atom(atype="Al", xyz=[0.5, 0.5, 0.0], lattice=latt),
    Atom(atype="Al", xyz=[0.5, 0.0, 0.5], lattice=latt),
    Atom(atype="Al", xyz=[0.0, 0.5, 0.5], lattice=latt),
]
structure_matrix = Structure(atoms=atoms, lattice=latt)
euler_angles = np.array([[0, 0, 0], [10.0, 0.0, 0.0]])
struct_library = StructureLibrary(["Al"], [structure_matrix], [euler_angles])
diff_gen = DiffractionGenerator(accelerating_voltage=200)
lib_gen = DiffractionLibraryGenerator(diff_gen)
diff_lib = lib_gen.get_diffraction_library(
    struct_library,
    calibration=0.0262,
    reciprocal_radius=1.6768,
    half_shape=64,
    with_direct_beam=True,
    max_excitation_error=0.02,
)

# %%
# New
# ---

from orix.crystal_map import Phase
from orix.quaternion import Rotation
from diffsims.generators.simulation_generator import SimulationGenerator

latt = Lattice(4, 4, 4, 90, 90, 90)
atoms = [
    Atom(atype="Al", xyz=[0.0, 0.0, 0.0], lattice=latt),
    Atom(atype="Al", xyz=[0.5, 0.5, 0.0], lattice=latt),
    Atom(atype="Al", xyz=[0.5, 0.0, 0.5], lattice=latt),
    Atom(atype="Al", xyz=[0.0, 0.5, 0.5], lattice=latt),
]
structure_matrix = Structure(atoms=atoms, lattice=latt)
p = Phase("Al", point_group="m-3m", structure=structure_matrix)
gen = SimulationGenerator(accelerating_voltage=200)
rot = Rotation.from_euler([[0, 0, 0], [10.0, 0.0, 0.0]], degrees=True)
sim = gen.calculate_diffraction2d(
    phase=p,
    rotation=rot,
    reciprocal_radius=1.6768,
    max_excitation_error=0.02,
    with_direct_beam=True,
)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(2):
    diff_lib["Al"]["simulations"][i].plot(
        size_factor=15, show_labels=True, ax=axs[i, 0]
    )
    sim.irot[i].plot(ax=axs[i, 1], size_factor=15, show_labels=True)
    axs[i, 0].set_xlim(-1.5, 1.5)
    axs[i, 0].set_ylim(-1.5, 1.5)
    axs[i, 1].set_xlim(-1.5, 1.5)
    axs[i, 1].set_ylim(-1.5, 1.5)

_ = axs[0, 0].set_title("Old")
_ = axs[0, 1].set_title("New")

# %%
