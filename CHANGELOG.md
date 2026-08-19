# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Adopted the py-canon fleet standard: reusable CI, docs and release
  workflows, ruff-only linting, pyright in place of mypy, and `uv_build`
  as the build backend.
- `__version__` is now read from installed package metadata rather than
  duplicated in the source.

## 0.1.0 - 2026-04-04

### Added

- Initial release: greedy, Hungarian, and structure-aware matchers;
  configurable field scorers; hierarchy specification; soft (optimal
  transport) matching; raking calibration; match diagnostics and plots.
