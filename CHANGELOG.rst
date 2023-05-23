=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0>`_,
and this project adheres to `Semantic Versioning
<https://semver.org/spec/v2.0.0.html>`_.

2023-05-22 - version 0.5.2
==========================

Fixed
-----
- Always use no-python mode to silence Numba deprecation warnings.

2023-01-25 - version 0.5.1
==========================

Fixed
-----
- ``ReciprocalLatticeVector.allowed`` rounds indices (hkl) internally to ensure correct
  selection of which vectors are allowed or not given a lattice centering. Integer
  indices are assumed.

Deprecated
----------
- Support for Python 3.6 is deprecated and will be removed in v0.6.

2022-06-10 - version 0.5.0
==========================

Added
-----
- Extra parameters in diffraction pattern's plot method for drawing miller index labels
  next to the diffraction spots.
- Option to use None for ``scattering_params`` which ignores atomic scattering.
- Python 3.10 support.
- Class ``ReciprocalLatticeVector`` for handling generation, handling and plotting of
  vectors. This class replaces ``ReciprocalLatticePoint``, which is deprecated.

Changed
-------
- Minimal version of dependencies orix >= 0.9, numpy >= 1.17 and tqdm >= 4.9.
- The Laue group representing the rotation list sampling of "hexagonal" from 6/m to
  6/mmm.
- Loosened the angle tolerance in ``DiffractionLibrary.get_library_entry()`` from
  ``1e-5`` to ``1e-2``.

Deprecated
----------
- Class ``ReciprocalLatticePoint`` is deprecated and will be removed in v0.6. Use
  ``ReciprocalLatticeVector`` instead.

2021-04-16 - version 0.4.2
==========================

Added
-----
- Simulations now have a .get_as_mask() method (#154, #158)
- Python 3.9 testing (#161)

Changed
-------
- Simulations now use a fractional (rather than absolute) min_intensity (#161)

Fixed
-----
- Precession simulations (#161)

2021-03-15 - version 0.4.1
==========================

Changed
-------
- `get_grid_beam_directions` default meshing changed to "spherified_cube_edge" from
  "spherified_cube_corner"

Fixed
-----
- `get_grid_beam_directions` now behaves correctly for the triclinic and monoclinic
  cases

2021-01-11 - version 0.4.0
==========================

Added
-----
- API reference documentation via Read The Docs: https://diffsims.readthedocs.io/en/latest/
- New module: `sphere_mesh_generators`
- New module: `detector_functions`
- New module: `ring_pattern_utils`
- beam precession is now supported in simulating electron diffraction patterns
- plot method for `DiffractionSimulation`
- more shape factor functions have been added
- This project now keeps a Changelog

Changed
-------
- `get_grid_beam_directions`, now works based off of meshes
- the arguments in the `DiffractionGenerator` constructor and the
  `DiffractionLibraryGenerator.get_diffraction_library` function have been shuffled so
  that the former captures arguments related to "the instrument/physics" while the
  latter captures arguments relevant to "the sample/material".
- CI is now provided by github actions

Removed
-------
- Python 3.6 testing

Fixed
-----
- ReciprocalLatticePoint handles having only one point/vector
