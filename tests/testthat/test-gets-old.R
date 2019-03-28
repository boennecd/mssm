context("Test versus old results")

test_that("'FSKA' gives the same", {
  set.seed(90638579)
  n <- 20000L
  p <- 4L
  X <- matrix(rnorm(n * p), nrow = p)
  ws <- exp(rnorm(n))
  ws <- ws / sum(ws)
  out <- FSKA:::FSKA(X = X, ws = ws, Y = X, N_min = 10L, eps = 1e-2,
                     n_threads = 2L)
  # exact <- FSKA:::naive(X = X, ws = ws, Y = X)
  # all.equal(out, exact)
  # out <- out / sum(out)
  # exact <- exact / sum(exact)
  # all.equal(out, exact)

  expect_known_value(out, "FSKA-old-res-small-eps.RDS")

  out <- FSKA:::FSKA(X = X, ws = ws, Y = X, N_min = 10L, eps = 1e-2,
                     n_threads = 1L)
  expect_known_value(out, "FSKA-old-res-small-eps.RDS")
})
