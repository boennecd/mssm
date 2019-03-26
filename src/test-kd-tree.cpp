#include <testthat.h>
#include "fast-kernal-approx.hpp"
#include<array>

context("Test KD-tree") {
  test_that("kd-tree splits as expected in 1D") {
    std::array<double, 4> dat = { 4., 1., 2., 3.};
    arma::mat X(dat.begin(), 1L, 4L, false);

    {
      KD_note note = get_KD_tree(X, 10L);
      expect_true(note.is_leaf());
    }
    {
      KD_note note = get_KD_tree(X, 4L);
      expect_true(note.is_leaf());
    }
    {
      KD_note note = get_KD_tree(X, 3L);
      expect_true(!note.is_leaf());
      auto leafs = note.get_leafs();
      expect_true(leafs.size() == 2L);

      {
        auto &left = note.get_left();
        std::vector<arma::uword> idx = left.get_indices();
        std::sort(idx.begin(), idx.end());
        std::array<arma::uword, 2L> expected = {1L, 2L};
        for(unsigned int i = 0; i < 2L; ++i)
          expect_true(idx[i] == expected[i]);
      }
      {
        auto &right = note.get_right();
        std::vector<arma::uword> idx = right.get_indices();
        std::sort(idx.begin(), idx.end());
        std::array<arma::uword, 2L> expected = {0L, 3L};
        for(unsigned int i = 0; i < 2L; ++i)
          expect_true(idx[i] == expected[i]);
      }
    }
  }
}

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

context("Test hyper_rectangle") {
  test_that("hyper_rectangle gives expected result in 2D") {
    /* [0, 1] x [0, 1] */
    std::array<double, 6L> d1 = { 0, 0,
                                .5, 0,
                                 1, 1};
    /* [2, 5] x [2, 4] */
    std::array<double, 6L> d2 = {  3,  3,
                                   5,  4,
                                   2,  2};
    arma::mat X1(d1.begin(), 2L, 3L, false),
              X2(d2.begin(), 2L, 3L, false);
    std::array<arma::uword, 6L> d3 = { 0L, 1L, 2L };
    arma::uvec idx(d3.begin(), 3L, false);

    hyper_rectangle r1(X1, idx), r2(X2, idx);

    {
      std::array<double, 2L> dists = r1.min_max_dist(r2);
      expect_true(std::abs(dists[0] - 1. * 1. - 1. * 1.) < 1e-12);
      expect_true(std::abs(dists[1] - 5. * 5. - 4. * 4.) < 1e-12);
    }

    /* [0, 5] x [0, 4] */
    hyper_rectangle r3(r1, r2);
    {
      std::array<double, 2L> dists = r3.min_max_dist(r3);
      expect_true(std::abs(dists[0]) < 1e-12);
      expect_true(std::abs(dists[1] - 5. * 5. - 4. * 4.) < 1e-12);
    }
  }
}
