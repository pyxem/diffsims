###########################
# Creating Template Library
###########################

from diffsims.crystallography import CrystalPhase
from diffpy.structure import Lattice, Atom, Structure

crystal = CrystalPhase("al",
                  space_group=225,
                  structure=Structure(lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
                  atoms=[Atom("Al", [0, 0, 1])],),
                  )

crystal.zap_rotations(density="3")

