## Test environments
* Ubuntu 20.04 LTS with gcc 10.1.0
  R version 4.1.1
* Ubuntu 20.04 LTS with gcc 10.1.0
  R version 4.1.1 with valgrind
* Ubuntu 20.04 LTS with gcc 10.1.0
  R devel 2021-09-06 r80861 with LTO checks
* Ubuntu 20.04 LTS with gcc 10.1.0
  R devel 2021-10-02 r81000
* Github actions on windows-latest (release), macOS-latest (release), 
  ubuntu-20.04 (release), and ubuntu-20.04 (devel)
* win-builder (devel, oldrelease, and release)
* `rhub::check_for_cran()`
* `rhub::check(platform = c("fedora-clang-devel", "macos-highsierra-release-cran"))`
  
## R CMD check results
The errors on CRAN have been fixed.

There were no WARNINGs or ERRORs.

There is a NOTE about the package size in some cases.
