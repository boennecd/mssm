#include <testthat.h>
#include "kd-tree.h"
#include<array>

context("Sample unit tests") {
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
        auto left = leafs[0];
        std::vector<arma::uword> idx = left->get_indices();
        std::sort(idx.begin(), idx.end());
        std::array<arma::uword, 2L> expected = {1L, 2L};
        for(unsigned int i = 0; i < 2L; ++i)
          expect_true(idx[i] == expected[i]);
      }
      {
        auto right = leafs[1];
        std::vector<arma::uword> idx = right->get_indices();
        std::sort(idx.begin(), idx.end());
        std::array<arma::uword, 2L> expected = {0L, 3L};
        for(unsigned int i = 0; i < 2L; ++i)
          expect_true(idx[i] == expected[i]);
      }
    }
  }
}
