#include <testthat.h>
#include "fast-kernel-approx.h"
#include <array>

context("Test source_node") {
  test_that("source_node gives expected result in 2D") {
    /* second rows has higher variation so it will be selected */
    std::array<double, 8> dat = { 0 ,   4,
                                  .5 ,  -2,
                                  .25,   2,
    -.3,  -1};
    arma::mat X(dat.begin(), 2L, 4L, false);
    std::array<double, 4> weights = {.1, .4, .3, .2};
    arma::vec ws(weights.begin(), 4L, false);

    KD_note node = get_KD_tree(X, 2L);
    expect_true(!node.is_leaf());

    source_node pn(X, ws, node);
    expect_true(!pn.node.is_leaf());

    {
      arma::vec expected(2L, arma::fill::zeros);
      for(unsigned int i = 0; i < 4L; ++i)
        expected += ws[i] * X.col(i);
      const arma::vec &cen = pn.centroid;
      for(unsigned int i = 0; i < 2L; ++i)
        expect_true(std::abs(cen[i] - expected[i]) < 1e-12);

      expect_true(std::abs(pn.weight - 1) < 1e-12);
    }
    {
      auto &left = *pn.left;
      expect_true(left.node.is_leaf());
      arma::vec expected(2L, arma::fill::zeros);
      double w = ws[1L] + ws[3L];
      expected += ws[1L] / w * X.col(1L);
      expected += ws[3L] / w * X.col(3L);
      const arma::vec &cen = left.centroid;
      for(unsigned int i = 0; i < 2L; ++i)
        expect_true(std::abs(cen[i] - expected[i]) < 1e-12);

      expect_true(std::abs(left.weight - w) < 1e-12);
    }
    {
      auto &right = *pn.right;
      expect_true(right.node.is_leaf());
      arma::vec expected(2L, arma::fill::zeros);
      double w = ws[0L] + ws[2L];
      expected += ws[0L] / w * X.col(0L);
      expected += ws[2L] / w * X.col(2L);
      const arma::vec &cen = right.centroid;
      for(unsigned int i = 0; i < 2L; ++i)
        expect_true(std::abs(cen[i] - expected[i]) < 1e-12);

      expect_true(std::abs(right.weight - w) < 1e-12);
    }
  }
}
