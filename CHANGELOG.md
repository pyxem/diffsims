# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- `ReciprocalLatticePoint.unique()` takes a `reduce` boolean parameter to make
  e.g. (2, 0, 0) equivalent to (1, 0, 0)
- `get_grid_beam_directions`, now works based off of meshes
- the arguments in the `DiffractionGenerator` constructor and the `DiffractionLibraryGenerator.get_diffraction_library` function have been shuffled so that the former captures arguments related to "the instrument/physics" while the latter captures arguments relevant to "the sample/material".
- CI is now provided by github actions

### Added
- `ReciprocalLatticePoint` read-only properties `families`, `n_families` and `family`
- API reference documentation via Read The Docs: https://diffsims.rtfd.io
- New module: `sphere_mesh_generators`
- beam precession is now supported in simulating electron diffraction patterns
- more shape factor functions have been added
- This project now keeps a Changelog

### Removed
- Python 3.6 testing

### Fixed
- `ReciprocalLatticePoint` handles having only one point/vector
