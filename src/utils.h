#ifndef MSSM_UTILS_H
#define MSSM_UTILS_H
#include "arma.h"
#include <mutex>
#include <type_traits>

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

/* class for arma object which takes a copy of the current value, set the
 * elements to zero and adds the copy back when this objects is
 * destructed. */
template<typename T>
class add_back {
  using cp_type  = typename std::remove_reference<T>::type;
  using org_type = typename std::add_lvalue_reference<T>::type;

  const cp_type copy;
  org_type org;

public:
  add_back(T& org):  copy(org), org(org)
  {
    org.zeros();
  }

  ~add_back(){
    if(arma::size(org) == arma::size(copy))
      org += copy;
    else
      Rcpp::Rcout << "'add_back' failed due to changed size\n";
  }
};

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

  chol_decomp() = delete;
  chol_decomp(const chol_decomp&) = delete;
  chol_decomp& operator=(const chol_decomp&) = delete;

  chol_decomp(chol_decomp&&) = default;

  /* returns R^{-\top}Z where Z is the input. You get R^- Z if  `transpose`
   * is true */
  void solve_half(arma::mat&, const bool transpose = false) const;
  void solve_half(arma::vec&, const bool transpose = false) const;
  arma::mat solve_half(const arma::mat&, const bool transpose = false) const;
  arma::vec solve_half(const arma::vec&, const bool transpose = false) const;

  /* inverse of the above */
  void mult_half(arma::mat&, const bool transpose = false) const;
  void mult_half(arma::vec&, const bool transpose = false) const;
  arma::mat mult_half(const arma::mat&, const bool transpose = false) const;
  arma::vec mult_half(const arma::vec&, const bool transpose = false) const;

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

  operator const arma::mat&() const {
    return X;
  }
};

/* LU factorization of matrix */
class LU_fact {
public:
  /* original matrix */
  const arma::mat X;

private:
  const int m = X.n_rows, n = X.n_cols;

  /* bool for whether the factorization is computed */
  std::unique_ptr<std::once_flag> is_comp =
    std::unique_ptr<std::once_flag>(new std::once_flag());
  std::unique_ptr<arma::mat> LU =
    std::unique_ptr<arma::mat>(new arma::mat());
  std::unique_ptr<int[]> ipiv =
    std::unique_ptr<int[]>(new int[std::min(m, n)]);

  /* return a reference to the LU decomposition. Must be called e.g., before
   * calling solve */
  const arma::mat& get_LU() const;

public:
  LU_fact(const arma::mat &X): X(X) { }

  LU_fact() = delete;
  LU_fact(const LU_fact&) = delete;
  LU_fact& operator=(const LU_fact&) = delete;

  LU_fact(LU_fact&&) = default;

  void solve(arma::mat&) const;
  void solve(arma::vec&) const;

  operator const arma::mat&() const {
    return X;
  }
};

/* normalizes log weights and returns the effective sample size */
inline double normalize_log_weights(arma::vec &low_ws)
{
  double max_w = -std::numeric_limits<double>::infinity();
  for(const auto d: low_ws)
    if(d > max_w)
      max_w = d;

  double norm_const = 0;
  for(auto &d : low_ws){
    d = std::exp(d - max_w);
    norm_const += d;
  }

  double ess_inv = 0.;
  for(auto &d: low_ws){
    d /= norm_const;
    ess_inv += d * d;
    d = std::log(d);
  }

  return 1. / ess_inv;
}

/* wrapper for dsyr. Only updates the upper half */
void arma_dsyr(arma::mat&, const arma::vec&, const double);

template<std::size_t size_outer, std::size_t size_inner>
class loop_nest_util {
  const std::size_t N_outer, N_inner;
  bool first_call = true;
  std::size_t i = 0L, j = 0L;

public:
  const std::size_t
    N_blocks_outer =
      N_outer / size_outer + (N_outer % size_outer > 0L),
    N_blocks_inner =
      N_inner / size_inner + (N_inner % size_inner > 0L),
    N_it = N_blocks_outer * N_blocks_inner;

  loop_nest_util(std::size_t N_outer, std::size_t N_inner):
    N_outer(N_outer), N_inner(N_inner)
  {
    static_assert(size_outer > 0L,
                  "size_outer must be stricly greater than zero");
    static_assert(size_inner > 0L,
                  "size_inner must be stricly greater than zero");
  }

  struct loop_data{
    std::size_t outer_start, outer_end, inner_start, inner_end;
  };

  /* return [start, end) for nested loop */
  loop_data operator()()
  {
    if(!first_call){
      ++j;
      if(j >= N_blocks_inner){
        ++i;
        j = 0L;
      }
    } else
      first_call = false;

#ifdef MSSM_DEBUG
    if(i >= N_blocks_outer)
      throw std::invalid_argument("to many calls to loop_nest_util::operator()");
#endif

    std::size_t outer_start = i * size_outer, inner_start = j * size_inner;

    return {
      outer_start, std::min(outer_start + size_outer, N_outer),
      inner_start, std::min(inner_start + size_inner, N_inner)
    };
  }
};

#endif
