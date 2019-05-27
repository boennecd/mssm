#ifndef PROBLEM_DATA_H
#define PROBLEM_DATA_H
#include "arma.h"
#include "dists.h"
#include "thread_pool.h"

/* util class to hold information and objects used for the computations */
class control_obj {
  std::unique_ptr<thread_pool> pool;
public:
  /* input needed for proposal distribution */
  const double nu, covar_fac, ftol_rel;
  /* number of particles */
  const arma::uword N_part;
  /* what to compute */
  const comp_out what_stat;
  const unsigned int trace;
  const arma::uword KD_N_min;
  const double aprx_eps;

  control_obj
    (const arma::uword, const double, const double, const double,
     const arma::uword, const std::string&, const unsigned int,
     const arma::uword, const double);
  control_obj& operator=(const control_obj&) = delete;
  control_obj(const control_obj&) = delete;
  control_obj(control_obj&&) = default;

  thread_pool& get_pool() const;
};

class problem_data {
  using cvec = const arma::vec;
  using cmat = const arma::mat;

  /* objects related to observed outcomes */
  cvec &Y;
  arma::vec cfix;
  cvec &ws, &offsets;
  arma::vec disp;
  cmat &X, &Z;
  const std::vector<arma::uvec> time_indices;

  /* objects related to state-space model */
  arma::mat F, Q, Q0;

  const std::string fam;

  /* objects related to computations */
  const std::unique_ptr<thread_pool> pool;
public:
  cvec mu0;
  const arma::uword n_periods;
  const control_obj ctrl;

  problem_data(
    cvec&, cvec&, cvec&, cvec&, cvec&, cmat&, cmat&,
    const std::vector<arma::uvec>&, cmat&, cmat&, cmat&,
    const std::string&, cvec&, control_obj&&);
  problem_data(const problem_data&) = delete;
  problem_data& operator=(const problem_data&) = delete;

  /* returns an object to compute the conditional distribution of the
   * observed outcome at a given time given a state vector */
  std::unique_ptr<cdist> get_obs_dist(const arma::uword) const;
  /* returns an object to compute the conditional distribution of the state
   * at a given time given a state vector at the previous time point */
  template<typename T>
  std::unique_ptr<T> get_sta_dist(const arma::uword) const;


  void set_cfix(const arma::vec &cnew){
#ifdef MSSM_DEBUG
    if(arma::size(cnew) != arma::size(cfix))
      throw std::invalid_argument("Invalid new value");
#endif
    cfix = cnew;
  }

  arma::vec get_cfix() const {
    return cfix;
  }

  void set_disp(const arma::vec &newdisp){
#ifdef MSSM_DEBUG
    if(arma::size(newdisp) != arma::size(disp))
      throw std::invalid_argument("Invalid new value");
#endif
    disp = newdisp;
  }

  arma::vec get_disp() const {
    return disp;
  }

  void set_F(const arma::mat &Fnew){
#ifdef MSSM_DEBUG
    if(arma::size(Fnew) != arma::size(F))
      throw std::invalid_argument("Invalid new value");
#endif
    F = Fnew;
  }

  arma::mat get_F() const {
    return F;
  }

  void set_Q(const arma::mat &Qnew){
#ifdef MSSM_DEBUG
    if(arma::size(Qnew) != arma::size(Q))
      throw std::invalid_argument("Invalid new value");
#endif
    Q = Qnew;
  }

  arma::mat get_Q() const {
    return Q;
  }

  void set_Q0(const arma::mat &Q0new){
#ifdef MSSM_DEBUG
    if(arma::size(Q0new) != arma::size(Q0))
      throw std::invalid_argument("Invalid new value");
#endif
    Q0 = Q0new;
  }

  arma::mat get_Q0() const {
    return Q0;
  }
};

#endif
