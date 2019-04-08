#include <testthat.h>
#include "utils-test.h"
#include "cloud.h"

context("Test particle_cloud") {
  test_that("Test particle_cloud and member functions") {
    /* R code
     X <- matrix(c(
      -1.54, -0.628, 1.25, 0.416, -0.416, 0.292, 1.25, -1.13, -0.176, 0.842),
      2L)
     w <- c(0.19011407, 0.18631179, 0.09125475, 0.22053232, 0.31178707)
     dput(colSums(t(X) * w))
     */
    constexpr arma::uword N_particles = 5L, state_dim = 2L, N_stats = 1L;
    particle_cloud pc(N_particles, state_dim, N_stats);

    expect_true(pc.particles.n_cols == N_particles);
    expect_true(pc.N_particles() == N_particles);

    expect_true(pc.particles.n_rows == state_dim);
    expect_true(pc.dim_particle() == state_dim);

    expect_true(pc.stats.n_cols == N_particles);
    expect_true(pc.stats.n_rows == N_stats);
    expect_true(pc.dim_stats() == N_stats);

    expect_true(pc.ws.n_elem == N_particles);
    expect_true(pc.ws_normalized.n_elem == N_particles);

    auto X = create_mat<state_dim, N_particles>(
      {-1.54, -0.628, 1.25, 0.416, -0.416, 0.292, 1.25, -1.13, -0.176, 0.842});
    auto ws = create_vec<N_particles>(
    {0.19011407, 0.18631179, 0.09125475, 0.22053232, 0.31178707});
    ws = arma::log(ws);

    /* should not do it this way in code (should not potentially change dim)*/
    pc.particles = X;
    pc.ws_normalized = ws;

    auto mea = pc.get_cloud_mean();
    auto expected = create_vec<2L>({ 0.12294296938, -0.00191635297999996 });

    expect_true(is_all_aprx_equal(expected, mea));
  }
}
