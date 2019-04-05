#include "cloud.h"

particle_cloud::particle_cloud
  (const arma::uword N_particles, const arma::uword dim_particle,
   const arma::uword dim_stats):
  particles(dim_particle, N_particles, arma::fill::none),
  stats(dim_stats, N_particles, arma::fill::none), ws(N_particles),
  ws_normalized(N_particles) { }
