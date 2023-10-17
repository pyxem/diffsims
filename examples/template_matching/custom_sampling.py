#######################################
# Custom Sampling for Template Matching
#######################################

from diffsims.generators.rotation_list_generators import get_reduced_fundamental_zone_grid
from orix.quaternion import symmetry
import matplotlib.pyplot as plt

syms = [symmetry.C1, symmetry.C2, symmetry.C4, symmetry.C6, symmetry.Oh]
sym_names = ["C1", "C2", "C4", "C6", "Oh"]

rots = [get_reduced_fundamental_zone_grid(2, point_group=s) for s in syms]
fig, axs = plt.subplots(1, len(syms), figsize=(15, 3))
for r, ax, s in zip(rots, axs, sym_names):
    r.plot(ax=ax, s=2)
    ax.set_title(s)
plt.show()

# %%
