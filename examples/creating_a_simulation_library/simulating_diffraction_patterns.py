"""
==============================================
Simple Diffraction Pattern Simulation Examples
==============================================

This example demonstrates how to simulate diffraction patterns using the
:class:`diffsims.generators.simulation_generator.SimulationGenerator` class. A
single diffraction pattern can be simulated for a single phase or multiple
diffraction patterns can be simulated for a single/multiple phases given
a rotation.

One Pattern for One Phase
--------------------------
"""

from orix.crystal_map import Phase
from orix.quaternion import Rotation
from diffpy.structure import Atom, Lattice, Structure
import matplotlib.pyplot as plt

from diffsims.generators.simulation_generator import SimulationGenerator

a = 5.431
latt = Lattice(a, a, a, 90, 90, 90)
atom_list = []
for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
    x, y, z = coords[0], coords[1], coords[2]
    atom_list.append(Atom(atype="Si", xyz=[x, y, z], lattice=latt))  # Motif part A
    atom_list.append(
        Atom(atype="Si", xyz=[x + 0.25, y + 0.25, z + 0.25], lattice=latt)
    )  # Motif part B
struct = Structure(atoms=atom_list, lattice=latt)
p = Phase(structure=struct, space_group=227)

gen = SimulationGenerator(
    accelerating_voltage=200,
)
rot = Rotation.from_axes_angles(
    [1, 0, 0], 45, degrees=True
)  # 45 degree rotation around x-axis
sim = gen.calculate_diffraction2d(phase=p, rotation=rot)

_ = sim.plot(show_labels=True)  # plot the first (and only) diffraction pattern

# %%

sim.coordinates  # coordinates of the first (and only) diffraction pattern

# %%
# Simulating Multiple Patterns for a Single Phase
# -----------------------------------------------

rot = Rotation.from_axes_angles(
    [1, 0, 0], (0, 15, 30, 45, 60, 75, 90), degrees=True
)  # 45 degree rotation around x-axis
sim = gen.calculate_diffraction2d(phase=p, rotation=rot)

_ = sim.plot(show_labels=True)  # plot the first diffraction pattern

# %%

_ = sim.irot[3].plot(
    show_labels=True
)  # plot the fourth(45 degrees) diffraction pattern
# %%

sim.coordinates  # coordinates of all the diffraction patterns

# %%
# Simulating Multiple Patterns for Multiple Phases
# ------------------------------------------------

p2 = p.deepcopy()  # copy the phase

p2.name = "al_2"

rot = Rotation.from_axes_angles(
    [1, 0, 0], (0, 15, 30, 45, 60, 75, 90), degrees=True
)  # 45 degree rotation around x-axis
sim = gen.calculate_diffraction2d(phase=[p, p2], rotation=[rot, rot])

_ = sim.plot(
    include_direct_beam=True, show_labels=True, min_label_intensity=0.1
)  # plot the first diffraction pattern

# %%

_ = (
    sim.iphase["al_2"].irot[3].plot(show_labels=True, min_label_intensity=0.1)
)  # plot the fourth(45 degrees) diffraction pattern

# %%
# Plotting a Pixelated Diffraction Pattern
# ----------------------------------------
dp = sim.get_diffraction_pattern(
    shape=(512, 512),
    calibration=0.01,
)
plt.figure()
_ = plt.imshow(dp)
# %%
