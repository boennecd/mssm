#include "dists.h"
#include <R_ext/Random.h>
#include <cmath>
#include "blas-lapack.h"
#include "dup-mult.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static constexpr int I_ONE = 1L;
static constexpr double D_M_ONE = -1.;

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

  /* allocate the memory we need */
  const int nm = dim, nm_sq = nm * nm, nm_lw = (nm * (nm + 1L)) / 2L;
  thread_local static std::vector<double> work_mem;
  if(what == Hessian){
    /* We need memory for
         an intermediary    [state dim]   x [state dim]   matrix
         an intermediary    [state dim]^2 x [state dim]^2 matrix
         same as above but only the lower triangular part * [state dim]^2
     */
    const unsigned int needed_dim = nm_sq * (1L + nm_sq + nm_lw);
    if(work_mem.size() < needed_dim)
      work_mem.resize(needed_dim);

  } else {
    /* We need memory for
         an intermediary [state dim] x [state dim] matrix */
    const unsigned int needed_dim = nm_sq;
    if(work_mem.size() < needed_dim)
      work_mem.resize(needed_dim);

  }
  double * const pwork_mem = work_mem.data();

  /* Need to compute
   \begin{aligned}
   \,Q^{-1}(y - Fx)x^\top
   &= R^{-1}(\tilde y - \tilde x)(F^{-1}R^\top \tilde x)^\top \\
   D^\top\text{vec}\,\frac 12 Q^{-1}((y-Fx)(y-Fx)^\top Q^{-1} - I)
   &= D^\top\text{vec}\,(\frac 12 R^{-1}(\tilde y-\tilde x)(\tilde y-\tilde x)^\top R^{-\top} - \frac 12 Q^{-1}) \\
   Q&=R^\top R \\
   Q^{-1}&= R^{-1}R^{-\top} \\
   \tilde x &= R^{-\top}Fx \\
   \tilde y &= R^{-\top}y
   \end{aligned}

   where D is the duplication matrix
   */

  /* start with Q */
  arma::vec xv(x, dim), yv(y, dim);
  double w_half = w * .5, w_half_neg = -w_half;
  yv -= xv;                    /* R^{-\top}(y - Fx) */
  chol_.solve_half(yv, true);  /* R^{-1}R^{-\top}(y - Fx) */

  {
    /* compute result */
    std::fill(pwork_mem, pwork_mem + nm_sq, 0.);
    F77_CALL(dger)(
        &nm, &nm, &w_half, yv.memptr(), &I_ONE, yv.memptr(), &I_ONE,
        pwork_mem, &nm);
    F77_CALL(daxpy)(
        &nm_sq, &w_half_neg, chol_.get_inv().memptr(), &I_ONE, pwork_mem,
        &I_ONE);
    D_mult_left(dim, 1L, 1., stat + nm_sq, nm_lw, pwork_mem);

  }

  /* then F */
  chol_.mult_half(xv);
  F.solve(xv);         /* get original x */
  double * const D_f_begin = stat;

  F77_CALL(dger)(
      &nm, &nm, &w, yv.memptr(), &I_ONE, xv.memptr(), &I_ONE,
      D_f_begin, &nm);

  if(what != Hessian)
    return;

  /* Need to compute.
   \begin{pmatrix}
   -\vec x_i \vec x_i^\top \otimes Q^{-1} & \cdot \\
   -D^\top ((Q^{-1}(\vec y_i - F^\top \vec x_i)\vec x_i^\top)\otimes Q^{-1}) &
   -D^\top (Q^{-1} \otimes \left(Q^{-1}(Z - \frac 12 Q\right)Q^{-1}))D
   \end{pmatrix}

   where D is the duplication matrix. We first compute the lower triangular and
   then we copy the output to the upper triangular. We define a few intermediary
   matrices
   */

  const int gdim = nm_sq + nm_lw;
  arma::mat kron_arg(pwork_mem        , nm   , nm   , false),
            kron_res(pwork_mem + nm_sq, nm_sq, nm_sq, false);
  double * const hess_ptr = stat + gdim,
    /* pointer to extra memory */
    *xtra_mem = pwork_mem + nm_sq * (nm_sq + 1L);

  /* lambda to add result */
  auto add_res = [&](const unsigned int inc, const arma::mat &X,
                     const bool is_upper, const bool is_left){
    double *res = hess_ptr + inc;
    const unsigned int inc_col = is_upper ? nm_lw : nm_sq,
                     n_row_ele = is_upper ? nm_sq : nm_lw,
                        n_cols = is_left  ? nm_sq : nm_lw;
    const double * new_term = X.memptr();
    for(unsigned int i = 0L; i < n_cols; ++i, res += inc_col)
      for(unsigned int j = 0L; j < n_row_ele; ++j, ++res, ++new_term)
        *res += w * *new_term;
  };

  /* compute upper left block */
  {
    kron_arg.zeros();
    F77_CALL(dger)(
      &nm, &nm, &D_M_ONE, xv.memptr(), &I_ONE, xv.memptr(), &I_ONE,
      kron_arg.memptr(), &nm);
    kron_res = arma::kron(kron_arg, chol_.get_inv());

    add_res(0L, kron_res, true, true);
  }

  /* compute lower left block */
  {
    kron_arg.zeros();
    F77_CALL(dger)(
        &nm, &nm, &D_M_ONE, yv.memptr(), &I_ONE, xv.memptr(), &I_ONE,
        kron_arg.memptr(), &nm);
    kron_res = arma::kron(kron_arg, chol_.get_inv());

    /* add result */
    D_mult_left(
      nm, kron_res.n_cols, w, hess_ptr + nm_sq, gdim, kron_res.memptr());

  }

  /* compute lower right block */
  {
    kron_arg.zeros();
    F77_CALL(dger)(
        &nm, &nm, &D_M_ONE, yv.memptr(), &I_ONE, yv.memptr(), &I_ONE,
        kron_arg.memptr(), &nm);
    kron_arg += .5 * chol_.get_inv();
    kron_res = arma::kron(chol_.get_inv(), kron_arg);

    std::fill(xtra_mem, xtra_mem + nm_lw * nm_sq, 0.);
    D_mult_left(nm, kron_res.n_cols, 1., xtra_mem, nm_lw, kron_res.memptr());

    /* add result */
    D_mult_right(nm, nm_lw, w, hess_ptr + (gdim + 1L) * nm_sq,
                 gdim, xtra_mem);
  }

  /* copy lower to upper part. TODO: do this in a smart way */
  arma::mat tmp(hess_ptr, gdim, gdim, false);
  tmp = arma::symmatl(tmp);
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

arma::vec* gaussian_log::set_disp(const arma::vec &in_vec){
  return scalar_pos_dist(in_vec);
}

std::array<double, 3> gaussian_log::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  static constexpr double norm_term = -.5 * std::log(2. * M_PI),
    eta_min = log(std::numeric_limits<double>::epsilon());

  const double var = disp->operator()(0L),
    log_var = disp->operator()(1L),
    eta_use = std::max(eta, eta_min),
    mu = std::exp(eta_use),
    diff = y - mu;
  std::array<double, 3> out;

  out[0L] = norm_term - .5 * log_var - diff * diff / (2.  * var);
  if(what == gradient or what == Hessian)
    out[1L] = diff / var * mu;

  if(what == Hessian)
    out[2L] = (y - 2 * mu) * mu / var;

  return out;
}

arma::vec* gaussian_inverse::set_disp(const arma::vec &in_vec){
  return scalar_pos_dist(in_vec);
}

std::array<double, 3> gaussian_inverse::log_density_state_inner
  (const double y, const double eta, const comp_out what) const
{
  gaurd_new_comp_out(what);

  static constexpr double norm_term = -.5 * std::log(2. * M_PI);

  const double var = disp->operator()(0L),
    log_var = disp->operator()(1L),
    mu = 1 / eta, diff = y - mu;
  std::array<double, 3> out;

  out[0L] = norm_term - .5 * log_var - diff * diff / (2.  * var);
  if(what == gradient or what == Hessian){
    const double veee = var * eta * eta * eta, ey = eta * y;
    out[1L] = (1. - ey) / veee;

    if(what == Hessian)
      out[2L] = (2. * ey - 3.) / (veee * eta);
  }

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
  EXP_CLASS_PTR(gaussian_log);
  EXP_CLASS_PTR(gaussian_inverse);

  throw invalid_argument("'" + which + "' is not supported");
}
