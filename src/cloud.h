#ifndef CLOUD_H
#define CLOUD_H
#include "arma.h"

class particle_cloud {
public:
  /* [dim] x [N particles] object. It is not const but the dimension __should
   * not__ be altered */
  arma::mat particles;
  /* [stats dim] x [N particles] object. This can e.g., contain sufficient
   * statistics if an EM algorithm is used */
  arma::mat stats;

  /* log particle weights */
  arma::vec ws;
  arma::vec ws_normalized; /* normalized log weights */

  /* number of particles, dimension of particles, and dimension of
   * statistics. The memory is uninitialized and should be initialized by
   * the caller */
  particle_cloud(const arma::uword, const arma::uword, const arma::uword);

  const arma::uword N_particles() const {
    return particles.n_cols;
  }
  const arma::uword dim_particle() const {
    return particles.n_rows;
  }
  const arma::uword dim_stats() const {
    return stats.n_rows;
  }
};

#endif
