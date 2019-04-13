#include "problem_data.h"

inline comp_out set_what_compute(const std::string &what){
  if(what == "log_density")
    return log_densty;
  else if(what == "gradient")
    return gradient;
  else if(what == "Hessian")
    return Hessian;

  throw std::logic_error("'" + what + "' not supported for 'what'");
}

control_obj::control_obj
  (const arma::uword n_threads, const double nu, const double covar_fac,
   const double ftol_rel, const arma::uword N_part, const std::string &what):
  pool(new thread_pool(std::max(n_threads, (unsigned int)1L))), nu(nu),
  covar_fac(covar_fac), ftol_rel(ftol_rel), N_part(N_part),
  what_stat(set_what_compute(what)) { }

thread_pool& control_obj::get_pool() const {
  return *pool;
}

problem_data::problem_data(
  cvec &Y, cvec &cfix, cvec &ws, cmat &X, cmat &Z,
  const std::vector<arma::uvec> &time_indices,
  cmat &F, cmat &Q, cmat &Q0, cvec &mu0, control_obj &&ctrl):
  Y(Y), cfix(cfix), ws(ws), X(X), Z(Z), time_indices(time_indices),
  F(F), Q(Q), Q0(Q0),
  /* public members */
  mu0(mu0), n_periods(time_indices.size()), ctrl(std::move(ctrl))
  { }

std::unique_ptr<cdist> problem_data::get_obs_dist(const arma::uword) const {
  throw std::logic_error("not implemented");
}

template<>
std::unique_ptr<cdist> problem_data::get_sta_dist(const arma::uword ti) const
{
  if(ti == 0)
    return std::unique_ptr<cdist>(new mv_norm_reg(F, Q0, mu0));
  return std::unique_ptr<cdist>(new mv_norm_reg(F, Q));
}

template<>
std::unique_ptr<trans_obj> problem_data::get_sta_dist(const arma::uword ti) const
{
  if(ti == 0)
    return std::unique_ptr<trans_obj>(new mv_norm_reg(F, Q0, mu0));
  return std::unique_ptr<trans_obj>(new mv_norm_reg(F, Q));
}
