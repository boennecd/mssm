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
   const double ftol_rel, const arma::uword N_part, const std::string &what,
   const unsigned int trace, const arma::uword KD_N_min,
   const double aprx_eps):
  pool(new thread_pool(std::max(n_threads, (unsigned int)1L))), nu(nu),
  covar_fac(covar_fac), ftol_rel(ftol_rel), N_part(N_part),
  what_stat(set_what_compute(what)), trace(trace), KD_N_min(KD_N_min),
  aprx_eps(aprx_eps) { }

thread_pool& control_obj::get_pool() const {
  return *pool;
}

problem_data::problem_data(
  cvec &Y, cvec &cfix, cvec &ws, cvec &offsets, cvec &disp, cmat &X, cmat &Z,
  const std::vector<arma::uvec> &time_indices,
  cmat &F, cmat &Q, cmat &Q0, const std::string &fam, cvec &mu0,
  control_obj &&ctrl):
  Y(Y), cfix(cfix), ws(ws), offsets(offsets), disp(disp), X(X), Z(Z),
  time_indices(time_indices), F(F), Q(Q), Q0(Q0), fam(fam),
  /* public members */
  mu0(mu0), n_periods(time_indices.size()), ctrl(std::move(ctrl))
  {
    if(ctrl.trace > 1L)
      Rcpp::Rcout << "problem_data\n"
                  << "------------\n"
                  << "Family '" + fam + "'\n"
                  << "F:\n" << F
                  << "Q:\n" << Q
                  << "Q0\n" << Q0
                  << "mu0\n" << mu0.t()
                  << "cfix\n" << cfix.t();
  }

std::unique_ptr<cdist> problem_data::get_obs_dist(const arma::uword ti) const {
#ifdef MSSM_DEBUG
  if(ti >= n_periods)
    throw std::invalid_argument("'ti' greater than 'n_periods'");
#endif

  const arma::uvec &indices = time_indices[ti];
  arma::vec y = Y(indices), ws_ = ws(indices), offs = offsets(indices);
  arma::mat x = X.cols(indices), z = Z.cols(indices);

  if(ctrl.trace > 2L){
    Rprintf("Time %5d\n", ti + 1L);
    Rcpp::Rcout << "----------\n"
                << "Y\n" << y.t()
                << "Weights\n" << ws_.t()
                << "Offsets\n" << offs.t()
                << "X\n" << x
                << "Z\n" << z;
  }

  return get_family(
    fam, std::move(y), std::move(x), cfix, std::move(z), &ws_,
    disp, std::move(offs));
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
