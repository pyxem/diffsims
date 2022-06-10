=============
API reference
=============

This reference manual details the public modules, classes, and functions in diffsims, as
generated from their docstrings. Many of the docstrings contain examples. See the user
guide for other ways to use diffsims.

.. caution::

    diffsims is in an alpha stage, so there may be breaking changes with each release.

.. currentmodule:: diffsims

The list of top modules:

.. autosummary::

    crystallography
    generators
    libraries
    sims
    structure_factor
    utils

....

crystallography
================

.. currentmodule:: diffsims.crystallography
.. automodule:: diffsims.crystallography

.. autosummary::
    ReciprocalLatticeVector
    ReciprocalLatticePoint
    get_equivalent_hkl
    get_highest_hkl
    get_hkl

ReciprocalLatticeVector
-----------------------

.. currentmodule:: diffsims.crystallography.ReciprocalLatticeVector

.. rubric:: Methods

.. autosummary::
    angle_with
    calculate_structure_factor
    calculate_theta
    cross
    deepcopy
    draw_circle
    dot
    dot_outer
    flatten
    from_highest_hkl
    from_min_dspacing
    from_miller
    get_circle
    get_hkl_sets
    print_table
    reshape
    sanitise_phase
    scatter
    squeeze
    stack
    symmetrise
    to_miller
    to_polar
    transpose
    unique

.. currentmodule:: diffsims.crystallography

.. autoclass:: ReciprocalLatticeVector
    :members: allowed, angle_with, azimuth, calculate_structure_factor, calculate_theta, coordinates, coordinate_format, cross, data, deepcopy, dim, dot, dot_outer, draw_circle, dspacing, flatten, from_highest_hkl, from_min_dspacing, from_miller, get_circle, get_hkl_sets, gspacing, has_hexagonal_lattice, hkl, hkil, h, k, i, l, multiplicity, ndim, polar, print_table, radial, reshape, sanitise_phase, scattering_parameter, scatter, shape, size, squeeze, stack, structure_factor, symmetrise, theta, to_miller, to_polar, transpose, unique, unit, x, xyz, y, z
    :undoc-members:
    :show-inheritance:

ReciprocalLatticePoint
----------------------

.. currentmodule:: diffsims.crystallography

.. autoclass:: ReciprocalLatticePoint
    :members:
    :undoc-members:
    :show-inheritance:

Functions
---------

.. currentmodule:: diffsims.crystallography

.. autofunction:: get_equivalent_hkl
.. autofunction:: get_highest_hkl
.. autofunction:: get_hkl

....

generators
==========

.. automodule:: diffsims.generators
    :members:
    :undoc-members:
    :show-inheritance:

diffraction_generator
---------------------

.. automodule:: diffsims.generators.diffraction_generator
    :members:
    :undoc-members:
    :show-inheritance:

library_generator
-----------------

.. automodule:: diffsims.generators.library_generator
    :members:
    :undoc-members:
    :show-inheritance:

rotation_list_generators
------------------------

.. automodule:: diffsims.generators.rotation_list_generators
    :members:
    :undoc-members:
    :show-inheritance:

sphere_mesh_generators
----------------------

.. automodule:: diffsims.generators.sphere_mesh_generators
    :members:
    :undoc-members:
    :show-inheritance:

zap_map_generator
-----------------

.. automodule:: diffsims.generators.zap_map_generator
    :members:
    :undoc-members:
    :show-inheritance:

....

libraries
=========

.. automodule:: diffsims.libraries
    :members:
    :undoc-members:
    :show-inheritance:

diffraction_library
-------------------

.. automodule:: diffsims.libraries.diffraction_library
    :members:
    :undoc-members:
    :show-inheritance:

structure_library
-----------------

.. automodule:: diffsims.libraries.structure_library
    :members:
    :undoc-members:
    :show-inheritance:

vector_library
--------------

.. automodule:: diffsims.libraries.vector_library
    :members:
    :undoc-members:
    :show-inheritance:

....

sims
====

.. automodule:: diffsims.sims
    :members:
    :undoc-members:
    :show-inheritance:

diffraction_simulation
----------------------

.. automodule:: diffsims.sims.diffraction_simulation
    :members:
    :undoc-members:
    :show-inheritance:

....

structure_factor
================

.. automodule:: diffsims.structure_factor
    :members:
    :undoc-members:
    :show-inheritance:

....

pattern
=======

detector_functions
------------------

.. automodule:: diffsims.pattern.detector_functions
    :members:
    :undoc-members:
    :show-inheritance:

....

utils
=====

.. automodule:: diffsims.utils
    :members:
    :undoc-members:
    :show-inheritance:

atomic_diffraction_generator_utils
----------------------------------

.. automodule:: diffsims.utils.atomic_diffraction_generator_utils
    :members:
    :undoc-members:
    :show-inheritance:

atomic_scattering_params
------------------------

.. automodule:: diffsims.utils.atomic_scattering_params
    :members:
    :undoc-members:
    :show-inheritance:

discretise_utils
----------------

.. automodule:: diffsims.utils.discretise_utils
    :members:
    :undoc-members:
    :show-inheritance:

fourier_transform
-----------------

.. automodule:: diffsims.utils.fourier_transform
    :members:
    :undoc-members:
    :show-inheritance:

generic_utils
-------------

.. automodule:: diffsims.utils.generic_utils
    :members:
    :undoc-members:
    :show-inheritance:

kinematic_simulation_utils
--------------------------

.. automodule:: diffsims.utils.kinematic_simulation_utils
    :members:
    :undoc-members:
    :show-inheritance:

lobato_scattering_params
------------------------

.. automodule:: diffsims.utils.lobato_scattering_params
    :members:
    :undoc-members:
    :show-inheritance:

probe_utils
-----------

.. automodule:: diffsims.utils.probe_utils
    :show-inheritance:

.. autoclass:: diffsims.utils.probe_utils.ProbeFunction
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __call__

.. autoclass:: diffsims.utils.probe_utils.BesselProbe
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __call__

scattering_params
-----------------

.. automodule:: diffsims.utils.scattering_params
    :members:
    :undoc-members:
    :show-inheritance:

shape_factor_models
-------------------

.. automodule:: diffsims.utils.shape_factor_models
    :members:
    :undoc-members:
    :show-inheritance:

sim_utils
---------

.. automodule:: diffsims.utils.sim_utils
    :members:
    :undoc-members:
    :show-inheritance:

vector_utils
------------

.. automodule:: diffsims.utils.vector_utils
    :members:
    :undoc-members:
    :show-inheritance:
