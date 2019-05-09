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

      arma::mat X_back = cl.mult_half(static_cast<const arma::mat>(O));
      expect_true(is_all_aprx_equal(X, X_back));

      arma::mat Xc = X;
      cl.solve_half(Xc);
      expect_true(is_all_aprx_equal(Xc, expect));

      cl.mult_half(Xc);
      expect_true(is_all_aprx_equal(Xc, X));
    }
    {
      arma::vec O = cl.solve_half(x);
      std::array<double, 2> expect =
        { 0.494974746830583, -0.790569415042095 };
      expect_true(is_all_aprx_equal(O, expect));

      arma::vec x_back = cl.mult_half(static_cast<const arma::vec>(O));
      expect_true(is_all_aprx_equal(x, x_back));

      arma::vec xc = x;
      cl.solve_half(xc);
      expect_true(is_all_aprx_equal(xc, expect));

      cl.mult_half(xc);
      expect_true(is_all_aprx_equal(x, xc));
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
    {
      arma::mat O = cl.solve(X), O2 = cl.solve_half(X);
      cl.solve_half(O2, true);
      expect_true(is_all_aprx_equal(O, O2));
    }
  }

  test_that("Testing chol decomposition with 'tranpose = true'") {
    /* R code:
    Q <- matrix(c(2, 1, 1, 3), 2L)
    X <- matrix(c(0.7, -0.9, 0.071, 0.35, -0.74, -0.25), 2L)
    Q_c <- chol(Q)
    dput(drop(solve(Q_c, X[, 1])))
    dput(drop(solve(Q_c, X)))
    dput(drop(crossprod(Q_c, X)))
    */

    auto Q = create_mat<2L, 2L>({ 2, 1, 1, 3 });
    const arma::mat X = create_mat<2L, 3L>({ 0.7, -0.9, 0.071, 0.35, -0.74, -0.25 });
    const arma::vec x = X.col(0);

    chol_decomp cl(Q);

    {
      arma::mat O = cl.solve_half(X, true);
      std::array<double, 6> expect =
        { 0.779579736245737, -0.569209978830308, -0.0604751366416484,
          0.221359436211787, -0.444202076573836, -0.158113883008419 };
      expect_true(is_all_aprx_equal(O, expect));

      arma::mat X_back = cl.mult_half(static_cast<const arma::mat>(O), true);
      expect_true(is_all_aprx_equal(X, X_back));

      arma::mat Xc = X;
      cl.solve_half(Xc, true);
      expect_true(is_all_aprx_equal(Xc, expect));

      cl.mult_half(Xc, true);
      expect_true(is_all_aprx_equal(Xc, X));
    }
    {
      arma::vec O = cl.solve_half(x, true);
      std::array<double, 2> expect =
        { 0.779579736245737, -0.569209978830308 };
      expect_true(is_all_aprx_equal(O, expect));

      arma::vec x_back = cl.mult_half(static_cast<const arma::vec>(O), true);
      expect_true(is_all_aprx_equal(x, x_back));

      arma::vec xc = x;
      cl.solve_half(xc, true);
      expect_true(is_all_aprx_equal(xc, expect));

      cl.mult_half(xc, true);
      expect_true(is_all_aprx_equal(x, xc));
    }
  }

  test_that("Testing LU_fact") {
    /* R code:
      F. <- matrix(c(2, 1, 5, 3), 2L)
      X <- matrix(c(0.7, -0.9, 0.071, 0.35, -0.74, -0.25), 2L)
      dput(solve(F., X))
      dput(solve(F., X[, 2]))
    */

    auto Fmat = create_mat<2L, 2L>({ 2, 1, 5, 3 });
    const arma::mat X = create_mat<2L, 3L>({ 0.7, -0.9, 0.071, 0.35, -0.74, -0.25 });
    const arma::vec x = X.col(1);

    LU_fact lu(Fmat);
    expect_true(is_all_equal(lu.X, Fmat));

    {
      arma::mat Xc = X;
      lu.solve(Xc);
      std::array<double, 6> expect = { 6.6, -2.5, -1.537, 0.629, -0.97, 0.24 };

      expect_true(is_all_aprx_equal(Xc, expect));
    }

    {
      arma::vec xc = x;
      lu.solve(xc);
      std::array<double, 2> expect = { -1.537, 0.629 };
      expect_true(is_all_aprx_equal(xc, expect));
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

  test_that("Testing add_back") {
    arma::vec x(5), expect(5);
    x.fill(1.);
    {
      expect.zeros();
      add_back<arma::vec> ad(x);
      expect_true(is_all_equal(x, expect));
    }
    expect.fill(1.);
    expect_true(is_all_equal(x, expect));
    {
      expect.zeros();
      add_back<arma::vec> ad(x);
      x(2) = 3;
      expect(2) = 3;
      expect_true(is_all_equal(x, expect));
    }
    expect.fill(1.);
    expect(2) = 4;
    expect_true(is_all_equal(x, expect));
  }
}

class get_rngs {
  std::array<double, 1000L> rngs = {{
    -0.5, -0.18, 0.19, 0.5, 1.67, 2.55, 0.52, -0.89, 0.73, -1.16,
    1.49, -0.85, 0.31, 1.93, 0.7, 0.02, 0.64, 1.66, -0.86, 0.62,
    1.34, -0.53, 0.53, -0.07, 0.11, 1.1, -0.15, -0.16, -0.39, -0.85,
    -2.17, -0.72, 1.7, -0.07, -1.28, -0.58, -1.39, -1.5, 0.93, -0.29,
    0.68, -0.63, -1.44, -0.59, -0.79, -0.93, -0.02, 0.99, 0.04, -2.02,
    1.89, 1.18, 2.41, -0.24, 2.07, 0.23, -0.37, -0.83, -1.46, 1.04,
    1.29, 1.14, 0.71, 0.29, 0.26, 1.99, -0.45, -0.68, 0.41, -0.59,
    -1.27, -0.9, 0.5, 1.9, 1.56, -1.11, 0.18, 2.03, 1.41, 0.95, 0.21,
    1.22, 1.09, -0.19, -0.8, -0.29, -1.44, 0.88, -0.11, -2.33, -1.1,
    0.39, 1.02, 0.38, -1.14, -0.75, 3.06, 1.44, 0.32, -0.11, -0.05,
    0.79, -0.22, -0.54, -0.27, 0.03, 0.97, -2.2, 1.95, 0.18, 0.42,
    0.65, -0.65, -1.12, -1, 0.57, 0.63, -1.21, -0.09, -1.48, 0.39,
    -0.51, -0.8, 0.28, 1.14, -1.69, -0.5, 1.92, 0.57, 2.62, -1.4,
    0.15, -0.91, 1.19, 1.34, 0.46, 1.68, 0.97, -1.44, -0.26, 1.02,
    -2.3, 0.32, -0.44, -0.42, -1.11, 0.37, 0.27, 0.59, 0.37, 0.45,
    0.53, -0.65, 1.02, 1.09, 0.08, -0.27, 0.08, 0.97, -0.52, 0.83,
    0.71, -2.05, -0.68, -1.26, 1.16, -1.05, -0.43, -0.23, 1.19, 0.54,
    -0.09, -0.71, 0.5, -0.11, -2.58, 0.95, 0.47, -0.44, 0.16, -0.75,
    2.03, 0.11, 1.39, 1.36, -0.71, -0.69, 0.27, -0.38, 1.12, 0.17,
    0.51, -1.77, -0.67, 0.28, 0.43, -0.96, 0.03, -0.37, 0.81, 0.3,
    0.02, 1.1, -0.16, 0.09, 0.02, 0.15, 0.48, 0.88, 0.32, 0.23, -0.11,
    1.11, 0.59, -0.32, 0.37, -0.76, -0.39, -2.35, -0.77, 0.46, 2.17,
    -2.04, 0.16, -0.76, -0.21, 0.76, -1, -0.94, -0.03, -1.51, -1.62,
    -0.39, -0.38, -1.6, -1.11, 0.06, -1.37, 1.37, -2.06, -0.99, 0.25,
    -1.06, -0.22, -0.84, 0.66, 0.45, 0.8, 0.36, 2.09, -0.07, 0.2,
    -1.53, -0.34, 0.95, 0.08, 1.71, 2.01, -0.79, -2.01, 1.14, 1.87,
    1.02, -0.21, 0.21, 0.45, 0.52, 0.38, -0.62, -0.57, 0.31, -0.71,
    0.52, 0.45, -0.51, -0.21, 0.03, 2.05, -0.77, -0.57, 1.67, -1.54,
    1.15, -1.56, 0.47, 0.04, -0.35, -0.94, 0.37, -1.87, 0.11, -0.49,
    1.2, 1.51, -0.73, -0.25, 0.13, 1.09, 1.3, 1.29, -0.34, 0.6, 0.92,
    1.64, -0.89, 2.17, 0.71, 1.04, -1.08, -0.47, -0.07, -0.76, -0.57,
    -1.13, 0.12, 1.23, -0.59, -1.84, 0.5, 0.51, 0.52, 0.62, -2.98,
    1.89, 0.46, 0.39, -1.19, 0.44, -0.75, 0.21, -0.14, 0.01, 0.68,
    -0.06, -0.5, -0.83, -0.29, -0.49, 0.98, 0.1, -0.5, 0.42, -0.52,
    1.92, -1.24, -0.16, 1.02, -0.63, -0.2, 0.41, -1.01, 1, -1.63,
    -0.23, -1.66, -0.83, 2.31, -0.26, 2, -0.24, -0.55, -0.53, 0.7,
    -0.94, 0.88, -0.74, 1.73, -0.81, -1.69, 1.49, 0.7, -0.96, -1.2,
    0, 0.7, -0.25, -0.01, 0.14, 0.53, -1.18, 2.48, -1.33, -1.36,
    -0.24, -0.09, 0.2, -0.27, -0.11, -0.21, 0.76, -0.16, -0.25, -1.65,
    -1.76, -0.39, -0.94, -1.77, -1.76, 0.72, 0.11, 0.39, -1.64, -0.65,
    -0.09, -0.69, 0.14, -0.54, 1.42, 0.53, -0.36, 0.43, -0.41, 1.07,
    1.05, -0.36, -0.25, 2.63, 0.08, -0.99, 0.34, 0.05, -2.51, 0.33,
    1.12, -0.4, 1.26, 0.13, 1.08, -0.03, 0.1, -0.51, 0.91, -0.39,
    2.06, -0.43, -0.63, -0.78, 0.11, 0.87, 1.42, 0.69, -2.26, 0.71,
    1.54, 0.32, -3.21, 0.44, -0.61, -0.03, -0.91, -0.51, 0.15, -1.34,
    0.16, -0.54, 0.42, -1.23, 1.11, 0.05, -3.2, -0.95, -0.21, 0.35,
    0.59, 1.57, -0.63, 0.87, -1.48, -0.4, -0.64, 0.22, -1.26, 1.99,
    0.98, -0.31, -1.31, -0.91, -0.09, -2.5, -0.58, 1.12, 1.17, -0.22,
    0.27, -0.91, -0.74, -0.43, 0.75, 0.65, -0.45, 1.06, -2.7, -0.33,
    1.32, 1.56, 0.86, 0.14, 0.2, -0.5, -1.15, -0.61, -0.15, -1.44,
    -0.24, 2.03, -1.02, -1.31, -0.62, 1.02, -0.83, 1.44, -1.27, -0.31,
    -0.61, -0.56, -0.56, -3.07, -1.31, 0.68, 1.6, -0.98, -0.43, -1.62,
    -0.65, 0.36, -0.79, 0.26, -0.11, -1.63, -0.99, -1.78, -1.12,
    1.59, -0.78, -0.66, -0.97, 0.75, 0.68, 1.3, 1, -0.47, 0.57, 0.42,
    1.57, 0.24, 0.32, -0.54, 0.12, 0.95, 1.49, 0.78, 0.53, 0.57,
    0.16, -0.4, 0.08, -1.2, -0.37, 0.65, 0.62, -0.45, 0.16, -0.51,
    -1.12, -0.45, -2.11, -0.73, -0.25, -0.61, 0.1, -0.8, 1.32, 0.03,
    -0.82, -0.87, -1.25, -1.05, 0.83, -0.84, -2.17, 0.77, 0.6, 1.2,
    -0.37, 0.7, 0.08, -2.26, 1.05, -0.9, 2.4, 1.26, 1.29, 1.99, 0.21,
    -0.44, -1.66, 0.58, -0.71, 1.37, 0.06, -0.25, 1.01, -0.66, 2.63,
    -0.31, 0.1, -0.69, 0.31, 0.92, -1.51, -0.01, 0.61, -1.02, 1.12,
    -0.26, 1.04, 0.49, -1.33, 1.11, -0.36, 1.72, -0.8, 1.43, -1.09,
    0.57, 1.42, -0.41, 0.05, -1.03, 1.03, 0.8, -0.67, 0.23, 1.86,
    -0.06, -0.98, 0.95, 0.19, -1.02, 0.01, -1.52, -0.67, -0.75, -0.03,
    -2.97, 0.69, 0.73, -1.13, -0.06, -0.26, -1.25, -0.1, 1.67, -1.28,
    0.75, 0.49, -2, -1.41, 0.92, -0.75, -1.24, 1.61, -0.39, 0.26,
    -0.2, 1.06, 0.26, 0.11, 0.83, 0.86, 0.33, 1.09, 0.6, 0.5, -0.46,
    0.92, -0.41, -0.83, 0.05, 0.02, -0.9, -0.65, -0.55, -1.05, -0.1,
    -1.43, -1.38, -1.5, 0.02, 0.1, -0.12, -0.4, 2.34, -0.98, 0.9,
    -0.09, -0.32, -1.4, -0.15, 0.64, -1.41, -0.89, 0.12, 0.93, -0.68,
    -0.13, 0.13, -0.21, 0.87, 1.56, 0.22, 0.41, -0.26, 0.89, 0.56,
    0.18, 1.45, 0.37, -0.4, -0.48, -0.16, -0.19, 0.18, 0.83, 0.39,
    0.74, -0.27, -0.06, 0.85, -1.21, 1.06, 2.37, -0.4, 0.99, 0.96,
    0.1, -0.65, -1.36, 0.45, 0.01, -1.02, -1.62, -0.86, 0.87, -1.01,
    -0.14, 0.57, -2.4, 0.28, 0.04, -1.85, -0.79, -0.5, -0.69, -0.55,
    -0.1, 0.82, -0.13, -0.59, 0.67, 1.01, -0.14, 0.86, -0.02, -0.26,
    -0.91, -1.13, 0.76, 0.57, -1.35, -2.03, 0.59, -1.41, 1.61, 1.84,
    1.37, -1.26, -1.38, -0.02, 0.16, -0.13, -0.08, -0.57, 1.67, -0.58,
    0.3, -0.39, -0.81, -0.65, -1.83, -0.1, 0.13, 0.42, 1.31, -1.66,
    -0.56, -1.64, 0.29, 2.29, 0.87, -0.37, 1.3, -0.16, 0.32, -0.07,
    0.18, 0.78, -1.26, 1.99, -0.35, 0.4, 2.13, -0.07, -0.6, 2.05,
    -0.99, 0.14, -0.46, -0.05, -0.19, -1.02, -1.44, 0.76, 0.7, -1.05,
    -1.48, -0.03, 0.94, -0.63, -0.63, -1.02, 0.79, 1.2, 1.25, -0.69,
    -0.54, 0.09, 1.03, 0.52, -0.52, 0.64, 0.32, 0.52, -1.5, -0.06,
    0.18, -0.95, 0.46, 0.91, -0.81, 0.21, -1.12, 0.33, -0.08, 0.92,
    -0.75, 0.85, -0.41, -0.03, 0.35, 1.33, -0.4, 0.71, -1.5, -0.99,
    -0.02, 0.41, 0.43, -1.1, -0.33, 0.23, -1.17, 0.59, 0.15, 0.35,
    0.49, -0.85, 0.63, 0.48, 1.46, -2.74, 0.77, -2.36, -1.27, -0.43,
    -0.44, -0.46, -1.13, -1.32, 0.15, -1.6, 0.43, -0.23, 0.55, 0.01,
    -1.75, -0.45, 0.52, 0.42, 1.01, 0.69, -0.95, -0.47, 1.25, 0.33,
    0.2, 1.42, 0.49, -0.64, 1.5, 2.38, 0.01, 1.15, -0.88, -1.47,
    -0.2, -1.5, 1.97, 0.59, -0.9, 2.07, -0.41, 0.05, -0.15, 0.03,
    0.48, -1.1, 0.33, -1.27, -0.5, -0.07, -1.66, -0.01, 0.92, 0.38,
    -0.73, 0.7, -0.42, 1.47, 0.08, -0.15, 1.02, 0.33, -0.58, 1.31,
    -1.14, -0.88, -0.49, 0.23, -0.07, 0.35, 0.69, -0.3, 2.09, -2.21,
    -1.24, 0.76, 0.15, 0.24, 0.47, -0.5, -0.62, 0.02, -0.41, 0.41,
    -1.34, -1.48, 1.03, -2.22, -1.64, 0.36, 0.03, -0.15, 0.81, -0.57,
    0.58, -0.97, 1.84, 1.2, 1.07, 0.64, 1.29, 0.62, 0.67, -2.62,
    -0.6, 1.01, -0.14, 0.03, 0.52, -0.08, -0.7, -1.1, -2.23, -0.18,
    0.5, -0.59, 0.56, 0.78, 0.4 }};

  int counter = 0L;

public:
  void reset(){
    counter = 0L;
  }

  double operator()(){
    return rngs.at(counter++);
  }
};

inline void do_tests
  (const unsigned dim_dia, const unsigned dim_off, const unsigned n_bands){
  unsigned n_cols = dim_dia * n_bands;
  arma::mat dense(n_cols, n_cols, arma::fill::zeros);
  sym_band_mat b_mat(dim_dia, dim_off, n_bands);
  b_mat.zeros();

  get_rngs rngs_gen;

  /* make non-symmetric matrix but we later copy to the lower part */
  for(unsigned i = 0; i < n_bands; ++i){
    arma::mat diag_mat(dim_dia, dim_dia);
    for(auto &d : diag_mat)
      d = rngs_gen();

    unsigned i_start = i * dim_dia, i_end = (i + 1L) * dim_dia - 1L;
    dense.submat(i_start, i_start, i_end, i_end) = diag_mat;
    b_mat.set_diag_block(i, diag_mat);

    if(i % 3L == 0){
      double alpha = std::pow(-1., i % 2) * i;

      /* test the adding feature */
      dense.submat(i_start, i_start, i_end, i_end) += alpha * diag_mat;
      b_mat.set_diag_block(i, diag_mat, alpha);

    }

    if(i + 1 < n_bands){
      unsigned j_start = (i + 1L) * dim_dia,
        j_end = std::min(j_start + dim_off, n_cols) - 1L;

      arma::mat diag_off(dim_dia, dim_off);
      for(auto &d : diag_off)
        d = rngs_gen();

      dense.submat(i_start, j_start, i_end, j_end) =
        diag_off.cols(0L, j_end - j_start);
      b_mat.set_upper_block(i, diag_off);
    }
  }

  dense = arma::symmatu(dense);
  arma::mat dense_out = b_mat.get_dense();

  expect_true(is_all_equal(dense, dense_out));

  /* test that dot product are the same */
  for(unsigned i = 0; i < 5; ++i){
    arma::vec rng(dense.n_cols);
    for(auto &r : rng)
      r = rngs_gen();

    arma::vec expect = dense * rng;
    arma::vec val = b_mat.mult(rng);

    expect_true(is_all_aprx_equal(expect, val));
  }
}

context("Test sym_band_mat") {
  test_that("work with equal 'dim_dia' and 'dim_off'"){
    do_tests(1L, 1L, 2L);
    do_tests(1L, 1L, 15L);

    do_tests(3L, 3L, 2L);
    do_tests(3L, 3L, 10L);

    do_tests(5L, 5L, 2L);
    do_tests(5L, 5L, 10L);
  }

  test_that("work with unequal 'dim_dia' and 'dim_off'"){
    do_tests(1L, 2L, 2L);
    do_tests(1L, 2L, 15L);

    do_tests(3L, 12L, 2L);
    do_tests(3L, 12L, 10L);

    do_tests(5L, 10L, 2L);
    do_tests(5L, 10L, 10L);
  }

  test_that("log determinant and solve is correct"){
    /* R code
     X <- matrix(c(1, .5, 0, .5, 2, 1, 0 , 1, 3), 3, byrow = TRUE)
     dput(determinant(X)$modulus)
     z = c(4, 3, 1)
     dput(solve(X, z))
     */
    sym_band_mat b_mat(1L, 1L, 3L);

    b_mat.set_diag_block(0L, create_mat<1L, 1L>({ 1. }));
    b_mat.set_diag_block(1L, create_mat<1L, 1L>({ 2. }));
    b_mat.set_diag_block(2L, create_mat<1L, 1L>({ 3. }));

    b_mat.set_upper_block(0L, create_mat<1L, 1L>({ .5 }));
    b_mat.set_upper_block(1L, create_mat<1L, 1L>({ 1. }));

    const double expect = 1.44691898293633;
    expect_true(std::abs(b_mat.ldeterminant() - expect) < 1e-8);

    const arma::vec z = create_vec<3L>({ 4, 3, 1 });
    auto expect_solve = create_vec<3L>
      ({3.76470588235294, 0.470588235294118, 0.176470588235294 });

    {
      arma::vec z_solve = b_mat.solve(z);
      expect_true(is_all_aprx_equal(expect_solve, z_solve));
    }

    sym_band_mat b_copy = b_mat;
    expect_true(std::abs(b_copy.ldeterminant() - expect) < 1e-8);

    {
      arma::vec z_solve = b_copy.solve(z);
      expect_true(is_all_aprx_equal(expect_solve, z_solve));
    }

  }
}
