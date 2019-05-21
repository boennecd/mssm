## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.0
* Ubuntu 14.04.5 LTS (on travis-ci with codename: trusty)
  R version 3.6.0
* win-builder (devel and release)
* Local Ubuntu 18.04 with R 3.5.2 and with clang 6.0.0 with ASAN and 
  UBSAN checks
* `rhub::check_for_cran()` and `rhub::check_with_sanitizers()`
 
## R CMD check results
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

## Resubmission
This is a resubmission. Below, I cover the changes due to the comments I got.

> please specify "Boost developers".

I have removed the author entry. Before it was as in my parglm package. 
It was included due to `src/thread_pool.h` which is a modified version from 
one of Anthony Williams' books and the first entries in the file are from the 
original file. 

I have removed the author entry as e.g., the BH package does not have any
author entry mentioning boost.

> Please provide small executable examples in all your exported functions'
> Rd files to illustrate the use of the exported function but also enable
> automatic testing.

I have added small executable examples to all manual pages.

> \dontrun{} is supposed to be used for examples which should not be 
> called by the user. Please consider replacing \dontrun{} with 
> \donttest{}, or unwrap the examples if they are executable in < 5 sec.

I have removed the `\dontrun{}`s from the manual pages. 

## Second Resubmission
I wrote to Anthony Williams. He is the only author of the original 
`src/thread_pool.h` file and has licensed it under Boost Software License 
1.0. Thus, the `Authors@R` is correct.
