#ifndef MSSM_UTILS_H
#define MSSM_UTILS_H
#include "arma.h"

inline double log_sum_log(const double old, const double new_term){
  double max = std::max(old, new_term);
  double d1 = std::exp(old - max), d2 = std::exp(new_term - max);

  return std::log(d1 + d2) + max;
}

inline double log_sum_log(const arma::vec &ws, const double max_weight){
  double norm_constant = 0;
  for(auto w : ws)
    norm_constant += std::exp(w - max_weight);

  return std::log(norm_constant) + max_weight;
}

inline double norm_square(const double *d1, const double *d2, arma::uword N){
  double  dist = 0.;
  for(arma::uword i = 0; i < N; ++i, ++d1, ++d2){
    double diff = *d1 - *d2;
    dist += diff * diff;
  }

  return dist;
}

class chol_decomp {
  /* upper triangular matrix R */
  const arma::mat chol_;
public:
  /* computes R in the decomposition X = R^\top R */
  chol_decomp(const arma::mat&);

  /* returns R^{-\top}Z where Z is the input */
  void solve_half(arma::mat&) const;
  void solve_half(arma::vec&) const;
  arma::mat solve_half(const arma::mat&) const;
  arma::vec solve_half(const arma::vec&) const;

  /* Computes Z^\top X */
  void mult(arma::mat &X) const {
    X = chol_.t() * X;
  }

  /* returns the log determinant */
  const double log_det() const {
    double out = 0.;
    for(arma::uword i = 0; i < chol_.n_cols; ++i)
      out += 2. * std::log(chol_(i, i));

    return out;
  }
};

#endif
