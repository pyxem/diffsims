####################################
# Simulating One Diffraction Pattern
# ==================================

from orix.crystal_map import Phase
from orix.quaternion import Rotation
from diffpy.structure import Atom, Lattice, Structure

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
sim = gen.calculate_ed_data(phase=p, rotation=rot)

sim.simulations[0].plot()  # plot the first (and only) diffraction pattern

# %%
