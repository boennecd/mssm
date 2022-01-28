## Test environments
* Ubuntu 20.04 LTS with gcc 10.1.0
  R version 4.1.2
* Ubuntu 20.04 LTS with gcc 10.1.0
  R version 4.1.2 with valgrind
* Ubuntu 20.04 LTS with gcc 10.1.0
  R devel 2022-01-27 r81578 with LTO checks
* Github actions on windows-latest (release), macOS-latest (release), 
  ubuntu-20.04 (release), and ubuntu-20.04 (devel)
* win-builder (devel, oldrelease, and release)
* `rhub::check_for_cran()`
* `rhub::check(platform = c("fedora-clang-devel", "macos-highsierra-release-cran"))`
* `rhub::check_on_solaris()`
  
## R CMD check results
The new version of this packages works with the new version of nloptr on CRAN.
I.e. the errors on CRAN for this package are solved.

There were no WARNINGs or ERRORs except on win-builder. The packages does not 
build with R-oldrelease and R-release. The reasons is that they seem to use the 
pre 2.0.0 version of nloptr.

There is a NOTE about the package size in some cases.

There are notes about (possibly) invalid URLs in some cases. The URLs are valid.
