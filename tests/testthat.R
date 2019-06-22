library(testthat)
library(mssm)

options(testthat.summary.max_reports = 1000L)
test_check("mssm", reporter = "summary")
