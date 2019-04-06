context("Testing sample methods")

test_that("'mv_norm' samples from correct distribution and results are reproducible", {
  set.seed(seed <- 41305703)
  Q <- matrix(c(4, 1, 2,
                1, 3, 2,
                2, 2, 6), ncol = 3)
  mu <- c(-4, 3, 1)
  N <- 1000L

  X1 <- sample_mv_normal(N = N, Q = Q, mu = mu)
  Z <- solve(t(chol(Q)), X1 - mu)
  pvs <- pnorm(c(Z))
  expect_true(ks.test(pvs, "punif")$p.value > 1e-4)

  # we can get the same
  set.seed(seed)
  X2 <- sample_mv_normal(N = N, Q = Q, mu = mu)
  expect_equal(X1, X2)
})

test_that("'mv_tdist' samples from correct distribution and results are reproducible", {
  set.seed(seed <- 41305703)
  Q <- matrix(c(4, 1, 2,
                1, 3, 2,
                2, 2, 6), ncol = 3)
  mu <- c(-4, 3, 1)
  N <- 1000L
  nu <- 6.7

  X1 <- sample_mv_tdist(N = N, Q = Q, mu = mu, nu = nu)
  Z <- solve(t(chol(Q)), X1 - mu)
  pvs <- pt(c(Z), df = nu)
  expect_true(ks.test(pvs, "punif")$p.value > 1e-4)

  # we can get the same
  set.seed(seed)
  X2 <- sample_mv_tdist(N = N, Q = Q, mu = mu, nu = nu)
  expect_equal(X1, X2)
})
