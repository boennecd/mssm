#include "utils.h"
#include <array>
#include <testthat.h>
#include "utils-test.h"

context("Test utils functions") {
  test_that("Testing chol decomposition") {
    /* R code:
      Q <- matrix(c(2, 1, 1, 3), 2L)
      X <- matrix(c(0.7, -0.9, 0.071, 0.35, -0.74, -0.25), 2L)
      Q_c <- chol(Q)
      dput(drop(backsolve(Q_c, X[, 1], transpose = TRUE)))
      dput(drop(backsolve(Q_c, X     , transpose = TRUE)))
      dput(drop(crossprod(Q_c, X)))
      dput(solve(Q, X))
      dput(solve(Q, X[, 1]))
      dput(solve(Q))
    */

    auto Q = create_mat<2L, 2L>({ 2, 1, 1, 3 });
    const arma::mat X = create_mat<2L, 3L>({ 0.7, -0.9, 0.071, 0.35, -0.74, -0.25 });
    const arma::vec x = X.col(0);

    chol_decomp cl(Q);

    double val, sign;
    arma::log_det(val, sign, Q);
    expect_true(std::abs(cl.log_det() - val) < 1e-12);

    {
      arma::mat O = cl.solve_half(X);
      std::array<double, 6> expect =
        { 0.494974746830583, -0.790569415042095, 0.0502045814642449,
          0.198907264824591, -0.523259018078045, 0.0758946638440411};
      expect_true(is_all_aprx_equal(O, expect));

      arma::mat Xc = X;
      cl.solve_half(Xc);
      expect_true(is_all_aprx_equal(Xc, expect));
    }
    {
      arma::mat O = cl.solve_half(x);
      std::array<double, 2> expect =
        { 0.494974746830583, -0.790569415042095 };
      expect_true(is_all_aprx_equal(O, expect));

      arma::vec xc = x;
      cl.solve_half(xc);
      expect_true(is_all_aprx_equal(xc, expect));
    }
    {
      std::array<double, 6> expect =
        { 0.989949493661167, -0.928050200245188, 0.10040916292849,
          0.603603171993711, -1.04651803615609, -0.918543725599092 };
      arma::mat Xc = X;
      cl.mult(Xc);
      expect_true(is_all_aprx_equal(Xc, expect));
    }
    {
      std::array<double, 6> expect =
        {0.6, -0.5, -0.0274, 0.1258, -0.394, 0.048};
      arma::mat O = cl.solve(X);
      expect_true(is_all_aprx_equal(O, expect));

      arma::mat Xc = X;
      cl.solve(Xc);
      expect_true(is_all_aprx_equal(Xc, expect));
    }
    {
      std::array<double, 2> expect = {0.6, -0.5};
      arma::vec O = cl.solve(x);
      expect_true(is_all_aprx_equal(O, expect));
    }
    {
      std::array<double, 4> expect = {0.6, -0.2, -0.2, 0.4};
      const arma::mat &inv_mat = cl.get_inv();
      expect_true(is_all_aprx_equal(inv_mat, expect));
    }
  }

  test_that("Testing dsyr") {
    /* R code
       A <- matrix(c(3, 1, 4,
       1, 9, 2,
       4, 2, 11), 3L)
       x <- c(-1, 3, 2)
       alpha <- 3.2
       o <- A + alpha * tcrossprod(x)
       o[lower.tri(o)] <- A[lower.tri(A)]
       dput(o)
     */
    auto A = create_mat<3L, 3L>({ 3, 1, 4, 1, 9, 2, 4, 2, 11});
    auto x = create_vec<3L>({ -1, 3, 2 });
    double alpha = 3.2;

    arma_dsyr(A, x, alpha);
    std::array<double, 9> expected =
      { 6.2, 1, 4, -8.6, 37.8, 2, -2.4, 21.2, 23.8 };
    expect_true(is_all_aprx_equal(A, expected));
  }
}
