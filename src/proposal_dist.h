#ifndef PROPOSAL_DIST_H
#define PROPOSAL_DIST_H
#include "arma.h"
#include "dists.h"

/* makes a mode approximation using the conditional distributions. The
 * approximation may be a multivariate normal or multivariate t-distribution.
 * The covariance matrix can be scaled by a constant factor. */
struct mode_approximation_output;
mode_approximation_output mode_approximation
  (std::initializer_list<const cdist*>, const arma::vec&,
   const double, const double, const double);

struct mode_approximation_output {
  /* proposal distribution */
  std::unique_ptr<proposal_dist> proposal;
  /* did mode approximation have any errors */
  bool any_errors;
};

#endif
