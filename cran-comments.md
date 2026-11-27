## New version submission

CRAN archived mssm on 2024-01-12 after check issues remained open.
Version 0.1.7 fixes C++ trace warnings, C++20 build errors, and the rounding issue in the Apple Silicon band matrix test.

## Test environments

The local check built the source tarball with `R CMD build` and used `R CMD check --as-cran`.
The external clock service was unavailable, so the check compared file timestamps with the local clock.

- Debian GNU/Linux 13 (x86_64), R 4.5.3, GCC and GNU Fortran 15.3.0.

The earlier standard hosted checks used `--no-manual --as-cran` with the incoming checks disabled.

- Ubuntu 26.04.1 LTS (x86_64), R 4.5.3, R 4.6.2 and R-devel (2026-11-25 r90991), GCC and GNU Fortran 15.2.0.
- macOS 26.7.2 (arm64), R 4.6.2, Apple Clang 21.0.0 and GNU Fortran 14.2.0.
- Windows (x86_64, build 26100), R 4.6.2, GCC and GNU Fortran 14.3.0.

### Memory checks on an earlier candidate

These checks used R-devel (2026-11-25 r90991) on x86_64 Linux.

- Fedora Linux 44, GCC and GNU Fortran 16.2.1. Valgrind 3.27.1 used origin tracking and full leak checking. R used `--no-manual --use-valgrind --no-stop-on-test-error`.
- Fedora Linux 44, GCC and GNU Fortran 16.2.1. The GCC build used address, undefined behavior, and strict bounds sanitizers with leak detection disabled. R used `--no-manual --no-build-vignettes --as-cran`.
- Ubuntu 26.04.1 LTS, Clang and Flang 23.1.6, with libc++ for C++. R used `--no-manual --no-stop-on-test-error` for both Clang checks. Clang ASan disabled leak, allocation mismatch, alignment, and floating-point division-by-zero checks. Clang UBSan disabled function checks.

## R CMD check results

The standard checks reported 0 errors and 0 warnings. The Debian check reported 1 NOTE for the prior CRAN archival. The standard hosted checks reported 0 NOTEs.

On the earlier candidate, Valgrind reported 0 memory errors and 0 bytes definitely or possibly lost. The examples and tests ran and the package check reported 0 errors, warnings, or NOTEs.

Both Clang checks stopped during installation because the source omitted a C++ thread header. After adding the header, all C++ translation units passed a local C++20 syntax check with Clang 23.1.6 and libc++.

The Fedora GCC sanitizer check stopped during installation because the `nloptr` binary required an unavailable `libRlapack.so` library. The three sanitizer checks await a rerun on the revised candidate.
