# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed
- Minimal version of dependencies: numpy >= 1.17, scipy >= 0.15, tqdm >= 4.9

## 2021-04-16 - version 0.4.2

### Added
- Simulations now have a .get_as_mask() method (#154, #158)
- Python 3.9 testing (#161)

### Fixed
- Precession simulations (#161)

### Changed
- Simulations now use a fractional (rather than absolute) min_intensity (#161)

## 2021-03-15 - version 0.4.1
### Changed
- `get_grid_beam_directions` default meshing changed to "spherified_cube_edge" from "spherified_cube_corner"

### Fixed
- `get_grid_beam_directions` now behaves correctly for the triclinic and monoclinic cases

## 2021-01-11 - version 0.4.0
### Changed
- `get_grid_beam_directions`, now works based off of meshes
- the arguments in the `DiffractionGenerator` constructor and the `DiffractionLibraryGenerator.get_diffraction_library` function have been shuffled so that the former captures arguments related to "the instrument/physics" while the latter captures arguments relevant to "the sample/material".
- CI is now provided by github actions

### Added
- API reference documentation via Read The Docs: https://diffsims.rtfd.io
- New module: `sphere_mesh_generators`
- New module: `detector_functions`
- New module: `ring_pattern_utils`
- beam precession is now supported in simulating electron diffraction patterns
- plot method for `DiffractionSimulation`
- more shape factor functions have been added
- This project now keeps a Changelog

### Removed
- Python 3.6 testing

### Fixed
- ReciprocalLatticePoint handles having only one point/vector
