#include <testthat.h>
#include "kd-tree.h"
#include <array>
#include "utils-test.h"

context("Test KD-tree") {
  test_that("kd-tree splits as expected in 1D") {
    auto X = create_mat<1L, 4L>({ 4., 1., 2., 3. });

    thread_pool pool(1L);

    {
      KD_note note = get_KD_tree(X, 10L, pool);
      expect_true(note.is_leaf());
    }
    {
      KD_note note = get_KD_tree(X, 4L, pool);
      expect_true(note.is_leaf());
    }
    {
      KD_note note = get_KD_tree(X, 3L, pool);
      expect_true(!note.is_leaf());
      auto leafs = note.get_leafs();
      expect_true(leafs.size() == 2L);

      {
        auto &left = note.get_left();
        std::vector<arma::uword> idx = left.get_indices();
        std::sort(idx.begin(), idx.end());
        std::array<arma::uword, 2L> expected = {1L, 2L};
        expect_true(is_all_equal(idx, expected));
      }
      {
        auto &right = note.get_right();
        std::vector<arma::uword> idx = right.get_indices();
        std::sort(idx.begin(), idx.end());
        std::array<arma::uword, 2L> expected = {0L, 3L};
        expect_true(is_all_equal(idx, expected));
      }
    }
  }
}

context("Test hyper_rectangle") {
  test_that("hyper_rectangle gives expected result in 2D") {
    /* [0, 1] x [0, 1] */
    auto X1 = create_mat<2L, 3L>({ 0., 0., .5, 0., 1., 1. });
    /* [2, 5] x [2, 4] */
    auto X2 = create_mat<2L, 3L>({ 3,  3, 5,  4, 2,  2});
    auto idx = create_vec<3L, arma::uvec::fixed>({
      (unsigned int)0, (unsigned int)1, (unsigned int)2 });

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
