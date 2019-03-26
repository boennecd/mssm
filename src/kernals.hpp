#ifndef KERNELS_H
#define KERNELS_H

#include <RcppArmadillo.h>

struct mvariate {
  const double norm_const_log;
  const double norm_const;
public:
  mvariate(const arma::uword&);

  double operator()(const double, const bool) const;
};

inline void log_sum_log(double &old, const double new_term){
  double max = std::max(old, new_term);
  double d1 = std::exp(old - max), d2 = std::exp(new_term - max);

  old = std::log(d1 + d2) + max;
}

inline double log_sum_log(const arma::vec &ws, const double max_weight){
  double norm_constant = 0;
  for(auto w : ws)
    norm_constant += std::exp(w - max_weight);

  return std::log(norm_constant) + max_weight;
}

inline double norm_square(const arma::vec &X, const arma::vec &Y){
#ifdef KFA_DEBUG
  if(X.n_elem != Y.n_elem)
    throw "element dimensions do not match";
#endif
  const double *d2 = Y.begin();
  double dist = 0.;
  for(auto d1 : X){
    double diff = d1 - *(d2++);
    dist += diff * diff;
  }

  return dist;
}

#endif
