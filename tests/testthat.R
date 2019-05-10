library(testthat)
library(mssm)

suppressWarnings(RNGversion("3.5.0"))

test_check("mssm", reporter = "summary")
