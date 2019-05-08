#ifndef SMOOTHER_H
#define SMOOTHER_H
#include "problem_data.h"

/* Performs backward smoothing given a data, marix with particles, and vector
 * with normalized log weights. It returns the normalized log smoothing
 * weights. */
std::vector<arma::vec> smoother
  (problem_data&, const std::vector<const arma::mat *>&,
   const std::vector<const arma::vec *>&);

/* same as above but using a dual k-d tree approximation */
std::vector<arma::vec> smoother_aprx
  (problem_data&, const std::vector<const arma::mat *>&,
   const std::vector<const arma::vec *>&);

#endif
