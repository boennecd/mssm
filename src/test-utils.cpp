#include "utils.h"
#include <array>
#include <testthat.h>

context("Test utils functions") {
  test_that("Testing chol decomposition") {
    /* R code:
      Q <- matrix(c(2, 1, 1, 3), 2L)
      X <- matrix(c(0.7, -0.9, 0.071, 0.35, -0.74, -0.25), 2L)
      Q_c <- chol(Q)
      dput(drop(backsolve(Q_c, X[, 1], transpose = TRUE)))
      dput(drop(backsolve(Q_c, X     , transpose = TRUE)))
      dput(drop(crossprod(Q_c, X)))
    */

    std::array<double, 4> Q_dat = { 2, 1, 1, 3 };
    std::array<double, 6> X_dat = {0.7, -0.9, 0.071, 0.35, -0.74, -0.25};
    const arma::mat Q(Q_dat.data(), 2L, 2L, false),
    X(X_dat.data(), 2L, 3L, false);
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

      for(double *e = expect.begin(), *o = O.begin(); o != O.end();
          ++e, ++o)
        expect_true(std::abs(*e - *o) < 1e-12);

      arma::mat Xc = X;
      cl.solve_half(Xc);
      for(double *e = expect.begin(), *o = Xc.begin(); o != Xc.end();
          ++e, ++o)
        expect_true(std::abs(*e - *o) < 1e-12);
    }
    {
      arma::mat O = cl.solve_half(x);
      std::array<double, 2> expect =
        { 0.494974746830583, -0.790569415042095 };
      for(double *e = expect.begin(), *o = O.begin(); o != O.end();
          ++e, ++o)
        expect_true(std::abs(*e - *o) < 1e-12);

      arma::vec xc = x;
      cl.solve_half(xc);
      for(double *e = expect.begin(), *o = xc.begin(); o != xc.end();
          ++e, ++o)
        expect_true(std::abs(*e - *o) < 1e-12);
    }
    {
      std::array<double, 6> expect =
        { 0.989949493661167, -0.928050200245188, 0.10040916292849,
          0.603603171993711, -1.04651803615609, -0.918543725599092 };
      arma::mat Xc = X;
      cl.mult(Xc);
      for(double *e = expect.begin(), *o = Xc.begin(); o != Xc.end();
          ++e, ++o)
        expect_true(std::abs(*e - *o) < 1e-12);
    }
  }
}
