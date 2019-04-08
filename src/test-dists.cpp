#include "dists.h"
#include <testthat.h>
#include <array>
#include "utils-test.h"

context("Test mvariate") {
  test_that("Test mvs_norm gives correct results in 3D") {
    auto x = create_vec<3L>({ -3, 2, 1 });
    auto y = create_vec<3L>({ -1, 0, 2 });

    std::unique_ptr<trans_obj> kernel(new mvs_norm(3L));

    /* print(sum(dnorm(c(2, 2, 1), log = TRUE)), digits = 16) */
    constexpr double expect = -7.256815599614018;
    expect_true(std::abs(kernel->operator()(
        x.begin(), y.begin(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(kernel->operator()(
        x.begin(), y.begin(), 3L, 1) - (expect + 1.)) < 1e-8);

    mvs_norm k2(y);
    expect_true(std::abs(k2.log_density_state(
        x, nullptr, nullptr, log_densty) - expect) < 1e-8);
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

    auto x = create_vec<3L>({ -3, 2, 3 });
    auto y = create_vec<3L>({ -1, 0, 2 });
    auto Q = create_mat<3L, 3L>({ 2, 1, 1,
                                  1, 1, 1,
                                  1, 1, 3 });

    mv_norm di(Q);

    constexpr double expect = -13.353389189894;

    expect_true(std::abs(di(
        x.begin(), y.begin(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(di(
        x.begin(), y.begin(), 3L, 1) - (expect + 1.)) < 1e-8);

    mv_norm di2(Q, y);

    expect_true(std::abs(di2.log_density_state(
        x, nullptr, nullptr, log_densty) - expect) < 1e-8);
    expect_true(std::abs(di2.log_prop_dens(
        x                                     ) - expect) < 1e-8);
  }

  test_that("Test mv_norm_reg gives correct results in 3D") {
    /* R code
    x <- c(-3, 2)
    y <- c(-1, 0)
    F. <- matrix(c(.8, .2, .1, .3), 2L)
    Q <- matrix(c(2, 1, 1, 1), 2L)

    library(mvtnorm)
    dput(dmvnorm(y, F. %*% x, Q, log = TRUE))
    F. %*% x
    */

    auto x = create_vec<2L>({ -3, 2});
    auto y = create_vec<2L>({ -1, 0,});
    auto Q = create_mat<2L, 2L>({ 2, 1, 1, 1});
    auto F = create_mat<2L, 2L>({.8, .2, .1, .3});

    mv_norm_reg di(F, Q);

    constexpr double expect = -2.55787706640935;

    expect_true(std::abs(di(
        x.begin(), y.begin(), 2L, 0.) - expect) < 1e-8);
    expect_true(std::abs(di(
        x.begin(), y.begin(), 2L, 1) - (expect + 1.)) < 1e-8);

    auto mea = di.mean(x);
    auto expected = create_vec<2L>({-2.2, 0});
    expect_true(is_all_aprx_equal(mea, expected));
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

    auto x = create_vec<3L>({-3, 2, 3});
    auto y = create_vec<3L>({-1, 0, 1});
    auto Q = create_mat<3L, 3L>({2, 1, 1,
                                 1, 1, 1,
                                 1, 1, 3});
    const double nu = 5;

    mv_tdist di(Q, nu);

    constexpr double expect = -9.40850033868649;

    expect_true(std::abs(di(
        x.begin(), y.begin(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(di(
        x.begin(), y.begin(), 3L, 1) - (expect + 1.)) < 1e-8);

    mv_tdist di2(Q, y, nu);
    expect_true(std::abs(di2.log_density_state(
        x, nullptr, nullptr, log_densty) - expect) < 1e-8);
    expect_true(std::abs(di2.log_prop_dens(
        x                                     ) - expect) < 1e-8);
  }
}
