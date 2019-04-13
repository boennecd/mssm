#ifndef MSSM_UTILS_H
#define MSSM_UTILS_H
#include "arma.h"
#include <mutex>

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
public:
  /* original matrix */
  const arma::mat X;

private:
  /* upper triangular matrix R */
  const arma::mat chol_;
  std::unique_ptr<std::once_flag> is_inv_set =
    std::unique_ptr<std::once_flag>(new std::once_flag());
  std::unique_ptr<arma::mat> inv_ =
    std::unique_ptr<arma::mat>(new arma::mat());

public:
  /* computes R in the decomposition X = R^\top R */
  chol_decomp(const arma::mat&);

  /* returns R^{-\top}Z where Z is the input. You get R^- Z if  `transpose`
   * is true */
  void solve_half(arma::mat&, const bool transpose = false) const;
  void solve_half(arma::vec&, const bool transpose = false) const;
  arma::mat solve_half(const arma::mat&, const bool transpose = false) const;
  arma::vec solve_half(const arma::vec&, const bool transpose = false) const;

  /* inverse of the above */
  template<typename T>
  void mult_half(T& X, const bool transpose = false) const {
    if(transpose)
      X = chol_ * X;
    else
      X = chol_.t() * X;
  }
  template<typename T>
  T mult_half(const T& X, const bool transpose = false) const {
    if(transpose)
      return chol_ * X;
    else
      return chol_.t() * X;
  }

  /* Return X^{-1}Z */
  void solve(arma::mat&) const;
  arma::mat solve(const arma::mat&) const;
  arma::vec solve(const arma::vec&) const;

  /* Computes Z^\top X */
  void mult(arma::mat &X) const {
    X = chol_.t() * X;
  }

  const arma::mat& get_inv() const;

  /* returns the log determinant */
  const double log_det() const {
    double out = 0.;
    for(arma::uword i = 0; i < chol_.n_cols; ++i)
      out += 2. * std::log(chol_(i, i));

    return out;
  }
};

/* wrapper for dsyr. Only updates the upper half */
void arma_dsyr(arma::mat&, const arma::vec&, const double);

#endif
