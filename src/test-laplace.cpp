#include "laplace.h"
#include <array>
#include <testthat.h>
#include "utils-test.h"

context("Test Laplace approximation functions") {
  test_that("Testing get_concentration") {
    /* R code
     set.seed(1L)
     p <- 3L
     T. <- 10L

     F. <- NULL
     while(is.null(F.) || any(abs(eigen(F.)$values) >= 1)){
     F. <- matrix(round(runif(p * p, -1, 1), 2), p)
     }
     Q <- round(drop(rWishart(1, 10L, diag(p))), 2)
     Q0 <- mssm:::.get_Q0(Q, F.)

     pT <- p * T.
     Ft <- Qt <- Z <- matrix(0., pT, pT)
     diag(Ft) <- 1.
     for(i in 1:(T. - 1L) - 1L)
     Ft[1:p + (i + 1L) * p, 1:p + i * p] <- -F.
     for(i in 2:T.- 1L)
     Qt[1:p + i * p, 1:p + i * p] <- Q
     Qt[1:p, 1:p] <- Q0
     x <- round(round(rnorm(pT), 2), 2)

     dput(F.)
     dput(Q)
     dput(Q0)
     dput(x)

     dput(crossprod(Ft, solve(Qt) %*% Ft) %*% x)
     */
    constexpr unsigned n_periods = 10L, p = 3L;
    const auto F = create_mat<p, p>(
      { -0.47, -0.26, 0.15, 0.82, -0.6, 0.8, 0.89, 0.32, 0.26 });
    const auto Q = create_mat<p, p>(
      { 3.65, 1.41, -0.58, 1.41, 10.61, 4.57, -0.58, 4.57, 11.7 });
    const auto Q0 = create_mat<p, p>(
      { 28.7923318168611, -2.5276264523623, 14.2122310836353,
        -2.52762645236234, 21.8536527974113, -4.83652849023149, 14.2122310836353,
        -4.83652849023149, 26.6238664524269 });
    const auto x = create_vec<p * n_periods>(
      { 0.39, -0.62, -2.21, 1.12, -0.04, -0.02, 0.94, 0.82, 0.59, 0.92,
        0.78, 0.07, -1.99, 0.62, -0.06, -0.16, -1.47, -0.48, 0.42, 1.36,
        -0.1, 0.39, -0.05, -1.38, -0.41, -0.39, -0.06, 1.1, 0.76, -0.16 });

    auto b_mat = get_concentration(F, Q, Q0, n_periods);

    auto out = b_mat.mult(x);
    auto expect = create_vec<p * n_periods>({
      0.527061847869083, -1.30230108923904, -1.14825424902337,
      1.3469720571271, -0.574618310376901, -0.163900890569835, 0.461603078409831,
      0.318741498806335, 0.076569539827185, -0.304014530445169, 1.23380322722634,
      0.498907523734899, -0.987025686334222, 0.65155012741432, 0.154989794427021,
      -0.168723673658683, -0.774787710121386, -0.600827185338358, 0.597264904522682,
      0.549162419015232, 0.420151547469758, -0.150023386406172, -0.0799055941824427,
      -0.611623835120652, 0.476550522223197, -0.417484690105351, -0.270580107101796,
      0.369862885520763, -0.0290294685221643, 0.0492551405767806
    });

    expect_true(is_all_aprx_equal(out, expect));
  }
}
