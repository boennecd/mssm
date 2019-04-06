#include "dists.h"
#include <testthat.h>
#include <array>

context("Test mvariate") {
  test_that("Test mvs_norm gives correct results in 3D") {
    std::array<double, 3> x = {-3, 2, 1};
    std::array<double, 3> y = {-1, 0, 2};

    std::unique_ptr<trans_obj> kernel(new mvs_norm(3L));

    /* print(sum(dnorm(c(2, 2, 1), log = TRUE)), digits = 16) */
    constexpr double expect = -7.256815599614018;
    expect_true(std::abs(kernel->operator()(
        x.data(), y.data(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(kernel->operator()(
        x.data(), y.data(), 3L, 1) - (expect + 1.)) < 1e-8);

    arma::vec x1(x.data(), 3L, false), y2(y.data(), 3L, false);
    mvs_norm k2(y2);

    expect_true(std::abs(k2.log_density_state(
        x1, nullptr, nullptr, cdist::log_densty) - expect) < 1e-8);
  }

  test_that("Test mv_norm gives correct results in 3D") {
    /* R code
       x <- c(-3, 2, 3)
       y <- c(-1, 0, 2)
       Q <- matrix(c(2, 1, 1,
                     1, 1, 1,
                     1, 1, 3), 3L)

       library(mvtnorm)
       dput(dmvnorm(x, y, Q, log = TRUE))
    */

    std::array<double, 3> x     = {-3, 2, 3};
    std::array<double, 3> y     = {-1, 0, 2};
    std::array<double, 9> Q_dat = {2, 1, 1,
                                   1, 1, 1,
                                   1, 1, 3};

    arma::mat Q(Q_dat.data(), 3L, 3L, false);
    mv_norm di(Q);

    constexpr double expect = -13.353389189894;

    expect_true(std::abs(di(
        x.data(), y.data(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(di(
        x.data(), y.data(), 3L, 1) - (expect + 1.)) < 1e-8);

    arma::vec x1(x.data(), 3L, false), y2(y.data(), 3L, false);
    mv_norm di2(Q, y2);

    expect_true(std::abs(di2.log_density_state(
        x1, nullptr, nullptr, cdist::log_densty) - expect) < 1e-8);
    expect_true(std::abs(di2.log_prop_dens(
        x1                                     ) - expect) < 1e-8);
  }

  test_that("Test mv_tdist gives correct results in 3D") {
    /* R code
      x <- c(-3, 2, 3)
      y <- c(-1, 0, 1)
      Q <- matrix(c(2, 1, 1,
                    1, 1, 1,
                    1, 1, 3), 3L)
      nu <- 5

      library(mvtnorm)
      dput(dmvt(x, y, Q, df = nu, log = TRUE))
    */

    std::array<double, 3> x     = {-3, 2, 3};
    std::array<double, 3> y     = {-1, 0, 1};
    std::array<double, 9> Q_dat = {2, 1, 1,
                                   1, 1, 1,
                                   1, 1, 3};
    const double nu = 5;

    arma::mat Q(Q_dat.data(), 3L, 3L, false);
    mv_tdist di(Q, nu);

    constexpr double expect = -9.40850033868649;

    expect_true(std::abs(di(
        x.data(), y.data(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(di(
        x.data(), y.data(), 3L, 1) - (expect + 1.)) < 1e-8);

    arma::vec x1(x.data(), 3L, false), y2(y.data(), 3L, false);
    mv_tdist di2(Q, y2, nu);

    expect_true(std::abs(di2.log_density_state(
        x1, nullptr, nullptr, cdist::log_densty) - expect) < 1e-8);
    expect_true(std::abs(di2.log_prop_dens(
        x1                                     ) - expect) < 1e-8);
  }
}
