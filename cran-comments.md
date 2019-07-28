## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.0
* Ubuntu 16.04 LTS (on travis-ci)
  R version 3.6.0
* win-builder (devel and release)
* Local Ubuntu 18.04 with R 3.5.2 and with clang 6.0.0 with ASAN and 
  UBSAN checks
* `rhub::check_for_cran()`, `rhub::check_on_solaris`, and 
  `rhub::check_on_macos()`
 
## R CMD check results
There were no ERRORs or WARNINGs.

I got the following NOTE on some environments (not local)

> Found the following (possibly) invalid DOIs:
>  DOI: 10.1093/biomet/asq062
>    From: DESCRIPTION
>    Status: libcurl error code 56:
>    	SSL read: error:00000000:lib(0):func(0):reason(0), errno 104
>    Message: Error

The doi do seem valid so I am not sure why I an error code 56
from libcurl. I also got the following on some environments (not local)

> Found the following (possibly) invalid URLs:
>   URL: http://doi.acm.org/10.1145/1143844.1143905 (moved to https://dl.acm.org/citation.cfm?doid=1143844.1143905)
>    From: README.md
>    Status: 403
>    Message: Forbidden
>   URL: http://www.jstor.org/stable/29777165
>    From: README.md
>    Status: 403
>    Message: Forbidden

but the urls are valid. I do not know why.

There is a NOTE about the size of the package on some platforms.
