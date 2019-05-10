## Test environments
* Ubuntu 18.04 LTS with gcc 8.2.0
  R version 3.5.3
* Ubuntu 14.04.5 LTS (on travis-ci with codename: trusty)
  R version 3.5.2
* win-builder (devel and release)
* Local Ubuntu 18.04 with R 3.5.2 and with clang 6.0.0 with ASAN and 
  UBSAN checks
* `rhub::check_for_cran()` and `rhub::check_with_sanitizers()`

## R CMD check results
There were no ERRORs, or WARNINGs. There is a note about the size of the 
package on all platforms.
