import numpy as np

#  TODO: Remove this section when minimum numpy version is 1.25.0
if np.__version__ >= "1.25.0":
    from numpy.exceptions import VisibleDeprecationWarning
else:
    VisibleDeprecationWarning = np.VisibleDeprecationWarning
