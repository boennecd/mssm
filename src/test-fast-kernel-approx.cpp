#include <testthat.h>
#include "fast-kernel-approx.h"
#include <array>
#include "utils-test.h"

using std::exp;
using std::log;

context("Test source_node") {
  test_that("source_node gives expected result in 2D") {
    /* second rows has higher variation so it will be selected */
    auto X = create_mat<2L, 4L>({0.,   4.,
                                  .5 ,  -2.,
                                  .25,   2.,
                                -.3,  -1.});
    auto ws = create_vec<4L>({ log(.1), log(.4), log(.3), log(.2) });

    thread_pool pool(1L);
    KD_note node = get_KD_tree(X, 2L, pool);
    expect_true(!node.is_leaf());

    source_node<false> pn(X, ws, node, nullptr);
    expect_true(!pn.node.is_leaf());

    {
      arma::vec expected(2L, arma::fill::zeros);
      for(unsigned int i = 0; i < 4L; ++i)
        expected += exp(ws[i]) * X.col(i);
      expect_true(is_all_aprx_equal(expected, pn.centroid));

      expect_true(std::abs(pn.weight - 1) < 1e-12);
    }
    {
      auto &left = *pn.left;
      expect_true(left.node.is_leaf());
      arma::vec expected(2L, arma::fill::zeros);
      double w = exp(ws[1L]) + exp(ws[3L]);
      expected += exp(ws[1L]) / w * X.col(1L);
      expected += exp(ws[3L]) / w * X.col(3L);
      expect_true(is_all_aprx_equal(expected, left.centroid));

      expect_true(std::abs(left.weight - w) < 1e-12);

    }
    {
      auto &right = *pn.right;
      expect_true(right.node.is_leaf());
      arma::vec expected(2L, arma::fill::zeros);
      double w = exp(ws[0L]) + exp(ws[2L]);
      expected += exp(ws[0L]) / w * X.col(0L);
      expected += exp(ws[2L]) / w * X.col(2L);
      expect_true(is_all_aprx_equal(expected, right.centroid));

      expect_true(std::abs(right.weight - w) < 1e-12);
    }
  }
}

context("Test FSKA_cpp") {
  test_that("FSKA_cpp permutates the input and return a vector to undo the permutation") {
    std::array<double, 8> x_mem ;
    std::array<double, 4> x_w_mem {  };
    std::array<double, 10> y_mem ;

    arma::mat X = create_mat<2L, 4L>(
      { 0.38, -2.00, -0.24, -0.72, -0.20,  0.49,  0.13,  0.09} );
    arma::vec X_w = create_vec<4L>({ -2.1628, -1.2694, -1.4740, -0.9808 });
    arma::mat Y = create_mat<2L, 5L>(
    { 0.071,  0.350, -0.740, -0.250, -0.220,  1.300, -1.000, -1.800, 0.930,
      1.500 });
    arma::vec Y_w(Y.n_cols, arma::fill::none);
    Y_w.fill(-std::numeric_limits<double>::infinity());

    arma::mat X_org = X, Y_org = Y;
    arma::vec w_org = X_w;
    thread_pool pool(1L);
    const mvs_norm kernel(X.n_rows);

    auto permu = FSKA_cpp(Y_w, X, Y, X_w, 2L, .01, kernel, pool);

    expect_true(!is_all_equal(X, X_org));
    expect_true(!is_all_equal(X_w, w_org));
    expect_true(!is_all_equal(Y, Y_org));

    X = X.cols(permu.X_perm);
    X_w = X_w (permu.X_perm);
    Y = Y.cols(permu.Y_perm);

    expect_true( is_all_equal(X, X_org));
    expect_true( is_all_equal(X_w, w_org));
    expect_true( is_all_equal(Y, Y_org));
  }
}
