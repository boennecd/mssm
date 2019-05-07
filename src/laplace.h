#ifndef LAPLACE_H
#define LAPLACE_H
#include "utils.h"
#include "problem_data.h"

/* takes matrices in conditional density in the state equation and returns
 * the concentration matrix. */
sym_band_mat get_concentration
  (const arma::mat&, const arma::mat&, const arma::mat&, const unsigned);

/* estimates parameters with Laplace approximation */
struct Laplace_aprx_output {
  arma::vec cfix;
  arma::mat F;
  arma::mat Q;
  double logLik;
  unsigned n_it;
  arma::vec disp;
  int code;
};
Laplace_aprx_output Laplace_aprx
  (problem_data&, const double, const double, const double, const double,
   const unsigned, const unsigned);

#endif
