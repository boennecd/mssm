## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.0
* Ubuntu 14.04.5 LTS (on travis-ci with codename: trusty)
  R version 3.6.0
* win-builder (devel and release)
* Local Ubuntu 18.04 with R 3.5.2 and with clang 6.0.0 with ASAN and 
  UBSAN checks
* `rhub::check_for_cran()`
 
## R CMD check results
This submission is mainly to fix the compilation problem with clang on some 
platforms.

I got the following WARNING on some environments (not local)

> Found the following (possibly) invalid DOIs:
>  DOI: 10.1093/biomet/asq062
>    From: DESCRIPTION
>    Status: libcurl error code 56:
>    	SSL read: error:00000000:lib(0):func(0):reason(0), errno 104
>    Message: Error

The doi do seem valid so I am not sure why the servers get an error code 56
from libcurl. I also got the following on some environments (not local)

> Found the following (possibly) invalid URLs:
>   URL: http://doi.acm.org/10.1145/1143844.1143905 (moved to https://dl.acm.org/citation.cfm?doid=1143844.1143905)
>    From: README.md
>    Status: 403
>    Message: Forbidden
>  URL: http://www.jstor.org/stable/29777165
>    From: README.md
>    Status: 403
>    Message: Forbidden

but the urls are valid. I do not know why some servers would get a 403.

There is a NOTE about the size of the package on most platforms.
