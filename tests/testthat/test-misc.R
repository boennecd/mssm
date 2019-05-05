context("Testing miscellaneous functions")

test_that("'.get_Q0' gives correct value", {
  F. <- structure(c(0.72, 0.06, 0.51, 0.25, -0.52, 0.62, -0.72, 0.05,
                    -0.35), .Dim = c(3L, 3L))
  Q <- structure(c(14.12, 2.05, 0.04, 2.05, 8.07, -1.67, 0.04, -1.67,
                   5.41), .Dim = c(3L, 3L))

  expect <- Q
  for(i in 1:1000)
    expect <- Q + tcrossprod(F. %*% expect, F.)

  expect_equal(expect, .get_Q0(Q, F.))
})
