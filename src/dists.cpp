#include "dists.h"
#include <R_ext/Random.h>
#include <cmath>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void mv_norm::sample(arma::mat &out) const {
#ifdef MSSM_DEBUG
  if(out.n_rows != dim)
    throw invalid_argument("'out' and 'dim' does not match");
#endif

  /* sample standard normal distributed variables */
  for(auto &x : out)
    x = norm_rand();

  /* account for covariance matrix and add mean */
  chol_.mult(out);
  if(mu)
    out.each_col() += *mu;
}

void mv_tdist::sample(arma::mat &out) const {
#ifdef MSSM_DEBUG
  if(out.n_rows != dim)
    throw invalid_argument("'out' and 'dim' does not match");
#endif

  /* sample standard normal distributed variables */
  for(auto &x : out)
    x = norm_rand();

  /* account for covariance matrix */
  chol_.mult(out);

  /* sample chi^2 variables */
  Rcpp::NumericVector chis = Rcpp::rchisq(out.n_cols, nu);
  arma::mat arma_chis(chis.begin(), 1L, out.n_cols, false);
  arma_chis.for_each([&](arma::vec::elem_type& val) {
    val = std::sqrt(val / nu) ; } );
  out.each_row() /= arma_chis;

  /* add mean */
  if(mu)
    out.each_col() += *mu;
}

std::array<double, 3> binomial_logit::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  std::array<double, 3> out;
  const double eta_use = MAX(MIN(eta, 20.), -20.);
  const double eta_exp = exp(eta_use);
  const double expp1 = eta_exp + 1;
  const double mu = eta_exp / expp1;

  out[0L] = y * log(mu) + (1. - y) * log1p(-mu);
  if(what == gradient or what == Hessian){
    out[1L] = (eta_exp * (y - 1) + y) / expp1;
    if(what == Hessian)
      out[2L] = - eta_exp / expp1 / expp1;
  }

  return out;
}

std::array<double, 3> poisson_log::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  const double lambda = MAX(exp(eta), std::numeric_limits<double>::epsilon());
  std::array<double, 3> out;
  out[0] = ([&]{
    if(y <= lambda * std::numeric_limits<double>::min())
      return -lambda;

    /* TODO: maybe look at aproximation used in r-source/src/nmath/dpois.c */
      return y * eta - lambda - std::lgamma(y + 1.);
  })();

  if(what == gradient or what == Hessian)
    out[1] = y - lambda;

  if(what == Hessian)
    out[2] = - lambda;

  return out;
}

inline arma::vec* scalar_pos_dist(const arma::vec &in_vec)
{
#ifdef MSSM_DEBUG
  if(in_vec.n_elem != 1L or in_vec(0) <= 0.)
    throw std::invalid_argument("Invalid dispersion parameter");
#endif
  /* we store the log dispersion parameter as the second element */
  arma::vec *out = new arma::vec(2L);
  out->operator()(0L) =          in_vec(0L);
  out->operator()(1L) = std::log(in_vec(0L));
  return out;
}

arma::vec* Gamma_log::set_disp(const arma::vec &in_vec){
  return scalar_pos_dist(in_vec);
}

std::array<double, 3> Gamma_log::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  const double mu = MAX(exp(eta), std::numeric_limits<double>::epsilon()),
    phi       = disp->operator()(0L),
    log_scale = eta - disp->operator()(1L);
  const double shape = 1. / phi,
    scale = mu * phi,
    log_y = std::log(y);

  std::array<double, 3> out;
  out[0] =
   - std::lgamma(shape) - shape * log_scale  + (shape - 1) * log_y -
   y / scale;

  if(what == gradient or what == Hessian)
    out[1] = - shape + y / scale;

  if(what == Hessian)
    out[2] = - y / scale;

  return out;
}

arma::vec* gaussian_identity::set_disp(const arma::vec &in_vec){
  return scalar_pos_dist(in_vec);
}

std::array<double, 3> gaussian_identity::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  static constexpr double norm_term = -.5 * std::log(2. * M_PI);
  const double var = disp->operator()(0L),
    log_var = disp->operator()(1L),
    diff = y - eta;
  std::array<double, 3> out;

  out[0L] = norm_term - .5 * log_var - diff * diff / (2.  * var);
  if(what == gradient or what == Hessian)
    out[1L] = diff / var;

  if(what == Hessian)
    out[2L] = - 1. / var;

  return out;
}
