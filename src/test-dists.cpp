#include "dists.h"
#include <testthat.h>
#include <array>
#include "utils-test.h"

context("Test state distribution") {
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

    {
      arma::mat x1 = x, y1 = y;
      di.trans_X(x1);
      di.trans_Y(y1);

      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 3L, 0.) - expect) < 1e-8);
      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 3L, 1) - (expect + 1.)) < 1e-8);

      di.trans_inv_X(x1);
      expect_true(is_all_aprx_equal(x1, x));
      di.trans_inv_Y(y1);
      expect_true(is_all_aprx_equal(y1, y));
    }

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

    {
      arma::mat x1 = x, y1 = y;
      di.trans_X(x1);
      di.trans_Y(y1);

      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 0.) - expect) < 1e-8);
      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 1) - (expect + 1.)) < 1e-8);

      di.trans_inv_X(x1);
      expect_true(is_all_aprx_equal(x1, x));
      di.trans_inv_Y(y1);
      expect_true(is_all_aprx_equal(y1, y));
    }

    auto mea = di.mean(x);
    auto expected = create_vec<2L>({-2.2, 0});
    expect_true(is_all_aprx_equal(mea, expected));
  }

  test_that("Test mv_norm_reg::comp_stats_state_state gives correct results in 3D"){
    /* R code
       x <- c(-3, 2, 1)
       y <- c(-1, 0, 2)
       F. <- matrix(c(.8, .2, .3, .1, .6, .2, .1, .1, .5), 3L)
       Q <- matrix(c(3, 1, 1, 1, 2, 3, 1, 3, 5), 3L)
       library(mvtnorm)
       library(numDeriv)

      dput(c(jacobian(function(f.){
      F.[] <- f.
      dmvnorm(y, F. %*% x, Q, log = TRUE)
      }, c(F.))))

      cp_lower <- function(x){
      x[upper.tri(x)] <- t(x)[upper.tri(x)]
      x
      }
      o1 <- c(jacobian(function(q.){
      Q[lower.tri(Q, diag = TRUE)] <- q.
      Q <- cp_lower(Q)
      dmvnorm(y, F. %*% x, Q, log = TRUE)
      }, Q[lower.tri(Q, diag = TRUE)]))
      o2 <- Q
      o2[lower.tri(o2, diag = TRUE)] <- o1
      o2[lower.tri(o2)] <- o2[lower.tri(o2)] * .5
      dput(c(cp_lower(o2)))
     */

    auto x = create_vec<3L>({ -3, 2, 1 });
    auto y = create_vec<3L>({ -1, 0, 2});
    auto F = create_mat<3L, 3L>({ .8, .2, .3, .1, .6, .2, .1, .1, .5 });
    auto Q = create_mat<3L, 3L>({ 3, 1, 1, 1, 2, 3, 1, 3, 5 });

    mv_norm_reg di(F, Q);
    di.trans_X(x);
    di.trans_Y(y);

    double log_one = 0., log_one_half = std::log(.5);
    {
      std::array<double, 0L> stat;
      expect_true(di.obs_stat_dim(log_densty)   == 0L);
      expect_true(di.obs_stat_dim(gradient)     == 0L);
      expect_true(di.obs_stat_dim(Hessian)      == 0L);
      expect_true(di.state_stat_dim(log_densty) == 0L);
      /* run to check it does not throw or access memory that it should not */
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), log_one, stat.data(), log_densty);
    }
    {
      expect_true(di.state_stat_dim(gradient)   == 18L);

      std::array<double, 18L> stat;
      arma::mat d_F(stat.data(), 3L, 3L, false);
      d_F.zeros();
      arma::mat d_Q(stat.data() + 9L, 3L, 3L, false);
      d_Q.zeros();
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), log_one, stat.data(), gradient);

      auto d_F_expect = create_mat<3L, 3L>({
        -6.74999999979376, 41.9999999988044, -25.0500000003388, 4.5000000023869,
        -27.9999999998909, 16.6999999985687, 2.25000000409953, -13.9999999981379,
        8.34999999954181 });
      auto d_Q_expect = create_mat<3L, 3L>({
        2.28125000103852, -15.2499999991022, 9.14374999806836, -15.2499999991022,
        94.5000000004886, -56.4499999999718, 9.14374999806836, -56.4499999999718,
        33.6112500002287 });

      expect_true(is_all_aprx_equal(d_F, d_F_expect, 1e-4));
      expect_true(is_all_aprx_equal(d_Q, d_Q_expect, 1e-4));

      /* mult by .5 weight instead */
      d_F.zeros();
      d_Q.zeros();
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), log_one_half, stat.data(), gradient);

      arma::mat ep_F_half = d_F_expect * .5;
      arma::mat ep_Q_half = d_Q_expect * .5;

      expect_true(is_all_aprx_equal(d_F, ep_F_half, 1e-4));
      expect_true(is_all_aprx_equal(d_Q, ep_Q_half, 1e-4));

      /* add something already */
      d_F.fill(1.);
      d_Q.fill(1.);
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), log_one_half, stat.data(), gradient);

      arma::mat ep_F_p1 = .5 * d_F_expect + 1;
      arma::mat ep_Q_p1 = .5 * d_Q_expect + 1;

      expect_true(is_all_aprx_equal(d_F, ep_F_p1, 1e-4));
      expect_true(is_all_aprx_equal(d_Q, ep_Q_p1, 1e-4));
    }
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

    {
      arma::mat x1 = x, y1 = y;
      di.trans_X(x1);
      di.trans_Y(y1);

      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 0.) - expect) < 1e-8);
      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 1) - (expect + 1.)) < 1e-8);

      di.trans_inv_X(x1);
      expect_true(is_all_aprx_equal(x1, x));
      di.trans_inv_Y(y1);
      expect_true(is_all_aprx_equal(y1, y));
    }

    mv_tdist di2(Q, y, nu);
    expect_true(std::abs(di2.log_density_state(
        x, nullptr, nullptr, log_densty) - expect) < 1e-8);
    expect_true(std::abs(di2.log_prop_dens(
        x                                     ) - expect) < 1e-8);
  }
}
