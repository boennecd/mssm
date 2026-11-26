## New version submission

CRAN archived mssm on 2024-01-12 after check issues remained open.
Version 0.1.7 corrects C++ trace warnings, fixes C++20 compilation, and allows for the rounding that caused the symmetric band matrix test to fail on Apple Silicon.

## Test environments

The local check built the source tarball with `R CMD build` and used `R CMD check --as-cran`.
The external clock service was unavailable, so the check compared file timestamps with the local clock.

- Debian GNU/Linux 13 (x86_64), R 4.5.3, GCC and GNU Fortran 15.3.0.

The hosted checks used `--no-manual --as-cran` with the incoming checks disabled.

- Ubuntu 26.04.1 LTS (x86_64), R 4.5.3, R 4.6.2 and R-devel (2026-11-25 r90991), GCC and GNU Fortran 15.2.0.
- macOS 26.7.2 (arm64), R 4.6.2, Apple Clang 21.0.0 and GNU Fortran 14.2.0.
- Windows (x86_64, build 26100), R 4.6.2, GCC and GNU Fortran 14.3.0.

## R CMD check results

All checks reported 0 errors and 0 warnings. The Debian check reported 1 NOTE for the prior CRAN archival. The hosted checks reported 0 NOTEs.
