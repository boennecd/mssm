#include "dists.h"
#include <R_ext/Random.h>
#include <cmath>
#include "blas-lapack.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static int I_ONE = 1L;

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

void mv_norm_reg::comp_stats_state_state
  (const double *x, const double *y, const double w, double *stat,
   const comp_out what) const
{
  gaurd_new_comp_out(what);
  if(what == log_densty)
    return;
  else if(what == Hessian)
    throw logic_error("not implemented");

  /* Need to compute
   \begin{aligned}
    Q^{-1}(y - Fx)x^\top
       &= R^{-1}(\tilde y - \tilde x)(F^{-1}R^\top \tilde x)^\top \\
    \frac 12 Q^{-1}((y-Fx)(y-Fx)^\top Q^{-1} - I)
       &= \frac 12 R^{-1}(\tilde y-\tilde x)(\tilde y-\tilde x)^\top R^{-\top} - \frac 12 Q^{-1} \\
    Q&=R^\top R \\
    Q^{-1}&= R^{-1}R^{-\top} \\
    \tilde x &= R^{-\top}Fx \\
    \tilde y &= R^{-\top}y
   \end{aligned}
   */

  /* start with Q */
  arma::vec xv(x, dim), yv(y, dim);
  double w_half = w * .5, w_half_neg = -w_half;
  yv -= xv;                    /* R^{-\top}(y - Fx) */
  chol_.solve_half(yv, true);  /* R^{-1}R^{-\top}(y - Fx) */

  double *d_Q_begin = stat + dim * dim;
  const int nm = dim, nm_sq = nm * nm;
  F77_CALL(dger)(
      &nm, &nm, &w_half, yv.memptr(), &I_ONE, yv.memptr(), &I_ONE,
      d_Q_begin, &nm);
  F77_CALL(daxpy)(
      &nm_sq, &w_half_neg, chol_.get_inv().memptr(), &I_ONE, d_Q_begin,
      &I_ONE);

  /* then F */
  chol_.mult_half(xv);
  F.solve(xv);         /* get original x */
  double *D_f_begin = stat;

  F77_CALL(dger)(
      &nm, &nm, &w, yv.memptr(), &I_ONE, xv.memptr(), &I_ONE,
      D_f_begin, &nm);
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

std::array<double, 3> binomial_cloglog::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  std::array<double, 3> out;
  static constexpr double
    /* log(-log1p(-.Machine$double.eps)) ~ log(.Machine$double.eps)  */
    eta_lower =
      log(std::numeric_limits<double>::epsilon()),
    /* log(-log  (.Machine$double.eps)) */
    eta_upper =
      log(-log  (std::numeric_limits<double>::epsilon()));
  const double eta_use = MAX(MIN(eta, eta_upper), eta_lower);
  const double eta_exp = exp(eta_use);
  const double mu = -std::expm1(-eta_exp);

  out[0L] = y * log(mu) + (1. - y) * log1p(-mu);
  if(what == gradient or what == Hessian){
    /* TODO: maybe issues with cancellation */
    out[1L] = (y - mu) / mu * eta_exp;
    if(what == Hessian){
      const double mumu   = mu * mu;
      /* TODO: maybe issues with cancellation */
      out[2L] = eta_exp / mumu * (y * (eta_exp * (mu - 1) + mu) - mumu);
    }
  }

  return out;
}

std::array<double, 3> binomial_probit::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  std::array<double, 3> out;
  static constexpr double
    /* qnorm(.Machine$double.eps). TODO: may yield issues on machines with
     * differnt epsilon */
    eta_lower = -8.12589066470191,
    eta_upper = -eta_lower;
  const double eta_use = MAX(MIN(eta, eta_upper), eta_lower);
  const double mu = R::pnorm5(eta_use, 0, 1, 1, 0);

  out[0L] = y * log(mu) + (1. - y) * log1p(-mu);
  if(what == gradient or what == Hessian){
    static constexpr double norm_const = 1. / sqrt(2. * M_PI);
    const double
      dmu_deta = norm_const * exp(-eta_use * eta_use / 2.),
      denom = mu * (1 - mu),
      dy_dmu = (y - mu) / denom;
    out[1L] = dy_dmu * dmu_deta;
    if(what == Hessian)
      out[2L] =
        (2 * y * mu - mu * mu - y) / (denom * denom) * dmu_deta * dmu_deta -
        out[1L] * eta_use;
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

std::array<double, 3> poisson_sqrt::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  const double lambda = eta * eta, y2 = 2. * y;
  std::array<double, 3> out;
  out[0] = ([&]{
    if(y <= lambda * std::numeric_limits<double>::min())
      return -lambda;

    /* TODO: maybe look at aproximation used in r-source/src/nmath/dpois.c */
    return y * log(lambda) - lambda - std::lgamma(y + 1.);
  })();

  if(what == gradient or what == Hessian)
    out[1] = y2 / eta - 2. * eta;

  if(what == Hessian)
    out[2] = - y2 / lambda  - 2.;

  return out;
}

inline arma::vec* scalar_pos_dist(const arma::vec &in_vec)
{
  if(in_vec.n_elem != 1L or in_vec(0) <= 0.)
    throw std::invalid_argument("Invalid dispersion parameter");
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

#define EXP_CLASS_PTR(fam)                                     \
  if(which == #fam)                                            \
    return(std::unique_ptr<exp_family>(new fam(                \
      Y, X, cfix, Z, ws, di, offset)))

std::unique_ptr<exp_family> get_family
  (const std::string &which, const arma::vec &Y, const arma::mat &X,
   const arma::vec &cfix, const arma::mat &Z, const arma::vec *ws,
   const arma::vec &di, const arma::vec &offset) {
  EXP_CLASS_PTR(binomial_logit);
  EXP_CLASS_PTR(binomial_cloglog);
  EXP_CLASS_PTR(binomial_probit);
  EXP_CLASS_PTR(poisson_log);
  EXP_CLASS_PTR(poisson_sqrt);
  EXP_CLASS_PTR(Gamma_log);
  EXP_CLASS_PTR(gaussian_identity);

  throw invalid_argument("'" + which + "' is not supported");
}
