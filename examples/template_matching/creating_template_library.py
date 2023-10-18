###########################
# Create a Template Library
###########################

from diffsims.crystallography import CrystalPhase
from diffpy.structure import Lattice, Atom, Structure
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator

# Creating a crystal phase to which we will make the template library from
crystal = CrystalPhase("al",
                  space_group=225,
                  structure=Structure(lattice=Lattice(4.04, 4.04, 4.04, 90, 90, 90),
                  atoms=[Atom("Al", [0, 0, 1])],),
                  )

# Creating a structure library from the crystal phase
struct = StructureLibrary(names=["al_phase", ],
                          phases=[crystal, ],
                          orientations=[crystal.constrained_rotation(), ])
# Creating a DiffractionGenerator to generate the diffraction patterns
calc = DiffractionGenerator(300.0)  # 300 keV

# Creating a DiffractionLibraryGenerator to generate the library
dfl = DiffractionLibraryGenerator(electron_diffraction_calculator=calc,
                                  structure_library=struct,
                                  reciprocal_radius=1.0,
                                  calibration=0.1,
                                  half_shape=72)
# Computing the library
library = dfl.calculate_library()

# plot the first diffraction pattern in the library
library["al_phase"].simulations[0].plot()
# %%
