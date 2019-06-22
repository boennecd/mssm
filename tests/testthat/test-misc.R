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

test_that("'mv_tdist::sample_anti' works with univariate distribution", {
  #####
  # works with multiple of 4
  set.seed(1)
  n_sims <- 10000L
  nu <- 5.5
  Q <- as.matrix(.5^2)
  mu <- 2
  samp <- drop(t_dist_antithe_test(n_sims = n_sims, Q = Q, mu = mu, nu = nu))

  s <- cbind(
    s1 = samp[seq(0, n_sims / 4L - 1L) * 4 + 1L],
    s2 = samp[seq(0, n_sims / 4L - 1L) * 4 + 2L],
    s3 = samp[seq(0, n_sims / 4L - 1L) * 4 + 3L],
    s4 = samp[seq(0, n_sims / 4L - 1L) * 4 + 4L])

  . <- function(x){
    substitute({
      expect_gt(
        ks.test((x - mu) / sqrt(drop(Q)), pt, df = nu)$p.value,
        .001)
      expect_equal(var(x), drop(Q * nu / (nu - 2)), tolerance = .1)
      expect_equal(mean(x), mu, tolerance = .1)
    }, list(x = substitute(x)))
  }
  eval(.(s[, 1]))
  eval(.(s[, 2]))
  eval(.(s[, 3]))
  eval(.(s[, 4]))
  eval(.(as.vector(s)))

  #####
  # works with non-multiple of 4
  set.seed(1)
  n_sims <- 19L
  nu <- 4.5
  Q <- as.matrix(.7^2)
  mu <- 2

  o <- replicate(1000, {
    x <- drop(t_dist_antithe_test(n_sims = n_sims, Q = Q, mu = mu, nu = nu))
    list(x1 = x[1:3], x2 = x[4:n_sims])
  })
  x1 <- unlist(o["x1", ])
  eval(.(x1))
  x2 <- unlist(o["x2", ])
  eval(.(x2))
})

test_that("'mv_tdist::sample_anti' works with multivariate distribution", {
  #####
  # works with multiple of 4
  set.seed(1)
  n_sims <- 100000L
  nu <- 5.5
  Q <- matrix(c(2, 1, 0, 1, 2, 2, 0, 2, 3), 3)
  mu <- c(2, -4, 0)
  samp <- t_dist_antithe_test(n_sims = n_sims, Q = Q, mu = mu, nu = nu)

  # we just check the margins an covariance matrix
  s <- list(
    s1 = t(samp[, seq(0, n_sims / 4L - 1L) * 4 + 1L]),
    s2 = t(samp[, seq(0, n_sims / 4L - 1L) * 4 + 2L]),
    s3 = t(samp[, seq(0, n_sims / 4L - 1L) * 4 + 3L]),
    s4 = t(samp[, seq(0, n_sims / 4L - 1L) * 4 + 4L]))

  . <- function(x){
    substitute({
      expect_gt(
        ks.test((x[1] - mu[1]) / sqrt(drop(Q[1, 1])), pt, df = nu)$p.value,
        .001)
      expect_gt(
        ks.test((x[2] - mu[2]) / sqrt(drop(Q[2, 2])), pt, df = nu)$p.value,
        .001)
      expect_gt(
        ks.test((x[3] - mu[3]) / sqrt(drop(Q[3, 3])), pt, df = nu)$p.value,
        .001)

      expect_equal(cov(x), Q * nu / (nu - 2), tolerance = .1)
      expect_equal(colMeans(x), mu, tolerance = .1)
    }, list(x = substitute(x)))
  }

  eval(.(s[["s1"]]))
  eval(.(s[["s2"]]))
  eval(.(s[["s3"]]))
  eval(.(s[["s4"]]))

  #####
  # works with non-multiple of 4
  set.seed(1)
  n_sims <- 19L
  nu <- 4.5

  o <- replicate(1000, {
    x <- t_dist_antithe_test(n_sims = n_sims, Q = Q, mu = mu, nu = nu)
    list(x1 = x[, 1:3], x2 = x[, 4:n_sims])
  }, simplify = FALSE)

  x1 <- t(do.call("cbind", lapply(o, "[[", "x1")))
  x2 <- t(do.call("cbind", lapply(o, "[[", "x2")))
  eval(.(x1))
  eval(.(x2))
})
