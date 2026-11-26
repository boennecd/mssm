## New version submission

CRAN archived mssm on 2024-01-12 after check issues remained open.
Version 0.1.7 corrects C++ trace format warnings and allows for the rounding differences that caused the symmetric band matrix test to fail on Apple Silicon.
The package now requires R 4.1.0 or later and uses the default C++ standard selected by R.

## Test environment

- Debian GNU/Linux 13 (x86_64), R 4.5.3, GCC and GNU Fortran 15.3.0.
- Built the source tarball with `R CMD build` and checked it with `R CMD check --as-cran`. The external clock service was unavailable, so the check compared file timestamps with the local clock.

## R CMD check results

0 errors, 0 warnings, and 1 NOTE. The incoming check recorded that mssm was previously archived.
