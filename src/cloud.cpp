#include "cloud.h"

particle_cloud::particle_cloud
  (const arma::uword N_particles, const arma::uword dim_particle,
   const arma::uword dim_stats):
  particles(dim_particle, N_particles, arma::fill::none),
  stats(dim_stats, N_particles, arma::fill::none), ws(N_particles),
  ws_normalized(N_particles) { }

arma::vec particle_cloud::get_cloud_mean() const {
  arma::vec out(dim_particle(), arma::fill::zeros);
  const arma::uword n_particles = N_particles();
  const double *w;
  arma::uword i;
  for(i = 0, w = ws_normalized.cbegin();
      i < n_particles; ++i, ++w)
    out += std::exp(*w) * particles.col(i);

  return out;
}

arma::vec particle_cloud::get_stats_mean() const {
  arma::vec out(dim_stats(), arma::fill::zeros);
  const arma::uword n_particles = N_particles();
  const double *w;
  arma::uword i;
  for(i = 0, w = ws_normalized.cbegin();
      i < n_particles; ++i, ++w)
    out += std::exp(*w) * stats.col(i);

  return out;
}
