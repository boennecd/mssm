#include <testthat.h>
#include "utils-test.h"
#include "dup-mult.h"

context("Test duplication matrix multiplication") {
  test_that("Works in 1xk and kx1 case") {
    {
      constexpr unsigned int k = 5L;
      const auto X = create_mat<1L, k>(
      { -0.6, 0.2, -0.8, 1.6, 0.3 });

      arma::mat out(1L, k, arma::fill::zeros);
      D_mult(out, X, true, 1L);

      expect_true(is_all_aprx_equal(out, X));
    }
    {
      constexpr unsigned int k = 5L;
      const auto X = create_mat<k, 1L>(
      { -0.6, 0.2, -0.8, 1.6, 0.3 });

      arma::mat out(k, 1L, arma::fill::zeros);
      D_mult(out, X, false, 1L);

      expect_true(is_all_aprx_equal(out, X));
    }
  }

  test_that("Test that left multiplication works") {
  {
    /* R code
     library(matrixcalc)
     n <- 2L
     k <- 5L
     set.seed(1)
     dput(X <- matrix(round(rnorm(n * n * k), 1), n * n))
     D <- duplication.matrix(n)
     dput(crossprod(D, X))
     */
    constexpr unsigned int n = 2L, k = 5L, xtra = 2L;
    const auto X = create_mat<n * n, k>(
    { -0.6, 0.2, -0.8, 1.6, 0.3, -0.8, 0.5, 0.7, 0.6, -0.3,
      1.5, 0.4, -0.6, -2.2, 1.1, 0., 0., 0.9, 0.8, 0.6 });

    arma::mat out((n * (n + 1L)) / 2L, k, arma::fill::zeros);
    D_mult(out, X, true, n);

    auto expect = create_mat<(n * (n + 1L)) / 2L, k>(
    { -0.6, -0.6, 1.6, 0.3, -0.3, 0.7, 0.6, 1.2, 0.4, -0.6,
      -1.1, 0., 0., 1.7, 0.6 });

    expect_true(is_all_aprx_equal(out, expect));

    /* test with xtra leading dimensions */
    arma::mat o1((n * (n + 1L)) / 2L + xtra, k, arma::fill::zeros);

    D_mult_left(n, k, 1., o1.memptr() + xtra, o1.n_rows, X.memptr());
    arma::mat o2 = o1.rows(xtra, o1.n_rows - 1L);
    expect_true(is_all_aprx_equal(o2, expect));

    arma::mat o3 = o1.rows(0, xtra - 1L);
    expect_true(std::equal(o3.begin() + 1, o3.end(), o3.begin()));

    /* with multiplier */
    const double alpha = .5;
    arma::mat z((n * (n + 1L)) / 2L, k, arma::fill::zeros);
    D_mult_left(n, k, alpha, z.memptr(), z.n_rows, X.memptr());

    expect *= alpha;
    expect_true(is_all_aprx_equal(z, expect));
  }

  {
    /* R code
    library(matrixcalc)
    n <- 5L
    k <- 2L
    set.seed(1)
    dput(X <- matrix(round(rnorm(n * n * k), 1), n * n))
    D <- duplication.matrix(n)
    dput(crossprod(D, X))
    */
    constexpr unsigned int n = 5L, k = 2L, xtra = 4L;
    const auto X = create_mat<n * n, k>(
    { -0.6, 0.2, -0.8, 1.6, 0.3, -0.8, 0.5, 0.7, 0.6, -0.3,
      1.5, 0.4, -0.6, -2.2, 1.1, 0., 0., 0.9, 0.8, 0.6, 0.9, 0.8, 0.1,
      -2., 0.6, -0.1, -0.2, -1.5, -0.5, 0.4, 1.4, -0.1, 0.4, -0.1, -1.4,
      -0.4, -0.4, -0.1, 1.1, 0.8, -0.2, -0.3, 0.7, 0.6, -0.7, -0.7,
      0.4, 0.8, -0.1, 0.9 });

    arma::mat out((n * (n + 1L)) / 2L, k, arma::fill::zeros);
    D_mult(out, X, true, n);

    auto expect = create_mat<(n * (n + 1L) / 2L), k>(
    { -0.6, -0.6, 0.7, 1.6, 1.2, 0.5, 1.1, 0.6, 0.5, -0.6,
      -1.3, 1.2, 0.8, -1.4, 0.6, -0.1, 1.2, -1.9, -0.7, -0.3, -0.1,
      0., -0.4, -1., -0.1, 1.8, 1.6, 0.6, -0.8, 0.9 });

    expect_true(is_all_aprx_equal(out, expect));

    /* test with xtra leading dimensions */
    arma::mat o1((n * (n + 1L)) / 2L + xtra, k, arma::fill::zeros);

    D_mult_left(n, k, 1., o1.memptr() + xtra, o1.n_rows, X.memptr());
    arma::mat o2 = o1.rows(xtra, o1.n_rows - 1L);
    expect_true(is_all_aprx_equal(o2, expect));

    arma::mat o3 = o1.rows(0, xtra - 1L);
    expect_true(std::equal(o3.begin() + 1, o3.end(), o3.begin()));

    /* with multiplier */
    const double alpha = .5;
    arma::mat z((n * (n + 1L)) / 2L, k, arma::fill::zeros);
    D_mult_left(n, k, alpha, z.memptr(), z.n_rows, X.memptr());

    expect *= alpha;
    expect_true(is_all_aprx_equal(z, expect));
  }
  }

  test_that("Test that right multiplication works") {
    {
      /* R code
      library(matrixcalc)
      n <- 2L
      k <- 5L
      set.seed(1)
      dput(X <- matrix(round(rnorm(n * n * k), 1), k))
      D <- duplication.matrix(n)
      dput(X %*% D)
      */
      constexpr unsigned int n = 2L, k = 5L, xtra = 3L;
      const auto X = create_mat<k, n * n>(
      { -0.6, 0.2, -0.8, 1.6, 0.3, -0.8, 0.5, 0.7, 0.6, -0.3,
        1.5, 0.4, -0.6, -2.2, 1.1, 0., 0., 0.9, 0.8, 0.6 });

      arma::mat out(k, (n * (n + 1L)) / 2L, arma::fill::zeros);
      D_mult(out, X, false, n);

      auto expect = create_mat<k, (n * (n + 1L)) / 2L>(
      { -0.6, 0.2, -0.8, 1.6, 0.3, 0.7, 0.9, 0.1, -1.6, 0.8,
        0., 0., 0.9, 0.8, 0.6 });

      expect_true(is_all_aprx_equal(out, expect));

      /* test with xtra leading dimensions */
      arma::mat o1(k + xtra, (n * (n + 1L)) / 2L, arma::fill::zeros);

      D_mult_right(n, k, 1., o1.memptr() + xtra, o1.n_rows, X.memptr());
      arma::mat o2 = o1.rows(xtra, o1.n_rows - 1L);
      expect_true(is_all_aprx_equal(o2, expect));

      arma::mat o3 = o1.rows(0, xtra - 1L);
      expect_true(std::equal(o3.begin() + 1, o3.end(), o3.begin()));

      /* with multiplier */
      const double alpha = .5;
      arma::mat z(k, (n * (n + 1L)) / 2L, arma::fill::zeros);
      D_mult_right(n, k, alpha, z.memptr(), z.n_rows, X.memptr());

      expect *= alpha;
      expect_true(is_all_aprx_equal(z, expect));
    }

    {
      /* R code
       library(matrixcalc)
       n <- 5L
       k <- 2L
       set.seed(1)
       dput(X <- matrix(round(rnorm(n * n * k), 1), k))
       D <- duplication.matrix(n)
       dput(X %*% D)
      */
      constexpr unsigned int n = 5L, k = 2L, xtra = 5L;
      const auto X = create_mat<k, n * n>(
      { -0.6, 0.2, -0.8, 1.6, 0.3, -0.8, 0.5, 0.7, 0.6, -0.3,
        1.5, 0.4, -0.6, -2.2, 1.1, 0., 0., 0.9, 0.8, 0.6, 0.9, 0.8, 0.1,
        -2., 0.6, -0.1, -0.2, -1.5, -0.5, 0.4, 1.4, -0.1, 0.4, -0.1, -1.4,
        -0.4, -0.4, -0.1, 1.1, 0.8, -0.2, -0.3, 0.7, 0.6, -0.7, -0.7,
        0.4, 0.8, -0.1, 0.9 });

      arma::mat out(k, (n * (n + 1L)) / 2L, arma::fill::zeros);
      D_mult(out, X, false, n);

      auto expect = create_mat<k, (n * (n + 1L)) / 2L>(
      { -0.6, 0.2, 0.7, 2., 1.2, 0., 1.9, 0.6, 0.4, -0.6, -0.6,
        -2.2, 1.2, -2., 0.4, 0.8, 1.5, 1.2, 0.6, -0.1, -1.6, -1.9, -1.2,
        -0.3, -0.4, -0.1, 1.5, 1.6, -0.1, 0.9 });

      expect_true(is_all_aprx_equal(out, expect));

      /* test with xtra leading dimensions */
      arma::mat o1(k + xtra, (n * (n + 1L)) / 2L, arma::fill::zeros);

      D_mult_right(n, k, 1., o1.memptr() + xtra, o1.n_rows, X.memptr());
      arma::mat o2 = o1.rows(xtra, o1.n_rows - 1L);
      expect_true(is_all_aprx_equal(o2, expect));

      arma::mat o3 = o1.rows(0, xtra - 1L);
      expect_true(std::equal(o3.begin() + 1, o3.end(), o3.begin()));

      /* with multiplier */
      const double alpha = .5;
      arma::mat z(k, (n * (n + 1L)) / 2L, arma::fill::zeros);
      D_mult_right(n, k, alpha, z.memptr(), z.n_rows, X.memptr());

      expect *= alpha;
      expect_true(is_all_aprx_equal(z, expect));
    }
  }
}
