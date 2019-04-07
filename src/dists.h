#ifndef DISTS_H
#define DISTS_H
#include "arma.h"
#include "utils.h"
#include "kd-tree.h"
#include <array>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

using std::logic_error;
using std::invalid_argument;
using std::log;
using std::exp;

/* class to compute conditional densities */
class cdist {
public:
  virtual ~cdist() = default;

  enum compute { log_densty, gradient, Hessian };

  /* gets the dimension of the state vector */
  virtual arma::uword state_dim() const = 0;
  /* computes the density and gradient and/or Hessian w.r.t. the state
   * vector if requested. */
  virtual double log_density_state
    (const arma::vec&, arma::vec*, arma::mat*, const compute) const = 0;
};

/* class for porposal distributions */
class proposal_dist {
public:
  virtual ~proposal_dist() = default;

  /* samples states and places them in input */
  virtual void sample(arma::mat&) const = 0;
  /* returns the log density of the proposal distribution */
  virtual double log_prop_dens(const arma::vec&) const = 0;
};

/* class to be used for kernel methods */
class trans_obj {
public:
  virtual ~trans_obj() = default;

  /* compute log kernel given two points, the number of elements, and a
   * log weight */
  virtual double operator()
    (const double*, const double*, const arma::uword, const double) const = 0;
  /* compute the smallest and largest log kernel distance between two
   * hyper rectangles */
  virtual std::array<double, 2> operator()
    (const hyper_rectangle&, const hyper_rectangle&) const = 0;
};

inline void check_input_mv_log_density_state
  (const arma::uword dim, const arma::vec *mu, const arma::vec &x,
   const arma::vec *gr, const arma::mat *H, const cdist::compute what)
{
#ifdef MSSM_DEBUG
  if(!mu)
    throw invalid_argument("'mu' not set");
  if(x.n_elem != mu->n_elem)
    throw invalid_argument(
        "invalid 'x' (" + std::to_string(x.n_elem) + " and " +
          std::to_string(mu->n_elem) + " elements)");
  if((what == cdist::gradient or what == cdist::Hessian) and
       (!gr or gr->n_elem != dim))
    throw invalid_argument("invalid 'gr'");
  if(what == cdist::Hessian and (!H or H->n_rows != dim or H->n_cols != dim))
    throw invalid_argument("invalid 'H'");
#endif
}

/* multivariate standard normal distribution */
class mvs_norm final : public cdist, public trans_obj {
  const std::unique_ptr<arma::vec> mu;
  const arma::uword dim;
  const double norm_const_log = -(double)dim / 2. * std::log(2. * M_PI);

  double log_dens_(const double dist) const
  {
    return norm_const_log - dist / 2.;
  }

public:
  mvs_norm(const arma::uword dim):
    mu(nullptr), dim(dim) { }
  mvs_norm(const arma::vec &mu):
    mu(new arma::vec(mu)), dim(mu.n_elem) { }

  double operator()(
      const double *x, const double *y, const arma::uword N,
      const double x_log_w) const override
  {
    const double dist = norm_square(x, y, N);
    return log_dens_(dist) + x_log_w;
  }

  std::array<double, 2> operator()
    (const hyper_rectangle &r1, const hyper_rectangle &r2) const override {
    auto dists = r1.min_max_dist(r2);
    return { log_dens_(dists[1L]), log_dens_(dists[0L]) };
  }

  arma::uword state_dim() const override {
    return dim;
  }

  double log_density_state
  (const arma::vec &x, arma::vec *gr, arma::mat *H, const compute what)
  const override
  {
    check_input_mv_log_density_state(state_dim(), mu.get(), x, gr, H, what);
    if(what != log_densty)
      throw logic_error("'mvs_norm': not implemented");

    return this->operator()(x.begin(), mu->begin(), x.n_elem, 0.);
  }
};

/* multivariate normal distribution */
class mv_norm final : public cdist, public trans_obj, public proposal_dist {
  const chol_decomp chol_;
  const std::unique_ptr<arma::vec> mu;
  const arma::uword dim;
  const double norm_const_log =
    -(double)dim / 2. * std::log(2. * M_PI) - .5 * chol_.log_det();

  double log_dens_(const double dist) const
  {
    return norm_const_log - dist / 2.;
  }

public:
  mv_norm(const arma::mat &Q):
    chol_(Q), mu(nullptr), dim(Q.n_cols) { }
  mv_norm(const arma::mat &Q, const arma::vec &mu):
    chol_(Q), mu(new arma::vec(mu)), dim(mu.n_elem) { }

  double operator()(
      const double *x, const double *y, const arma::uword N,
      const double x_log_w) const override
  {
    arma::vec x1(x, N), y1(y, N);
    chol_.solve_half(x1);
    chol_.solve_half(y1);
    const double dist = norm_square(x1.begin(), y1.begin(), N);
    return log_dens_(dist) + x_log_w;
  }

  std::array<double, 2> operator()
  (const hyper_rectangle &r1, const hyper_rectangle &r2) const override {
    throw logic_error("'mv_norm': not implemented");
  }

  arma::uword state_dim() const override {
    return dim;
  }

  double log_density_state
  (const arma::vec &x, arma::vec *gr, arma::mat *H, const compute what)
  const override
  {
    check_input_mv_log_density_state(state_dim(), mu.get(), x, gr, H, what);
    if(what == gradient or what == Hessian){
      const arma::vec diff = *mu - x; /* account for - later */
      *gr += chol_.solve(diff);
    }
    if(what == Hessian)
      *H -= chol_.get_inv();

    return operator()(x.begin(), mu->begin(), x.n_elem, 0.);
  }

  void sample(arma::mat&) const override;

  double log_prop_dens(const arma::vec &x) const override {
    return log_density_state(x, nullptr, nullptr, cdist::log_densty);
  }

  const arma::mat& mean(){
    if(!mu)
      throw std::logic_error("no mean");
    return *mu;
  }

  arma::mat vCov() const {
    return chol_.X;
  };
};

/* multivariate t distribution */
class mv_tdist : public cdist, public trans_obj, public proposal_dist {
  /* decomposition of __scale__ matrix */
  const chol_decomp chol_;
  const std::unique_ptr<arma::vec> mu;
  const arma::uword dim;
  const double nu;
  static double set_const
    (const double nu, const double dim, const chol_decomp &chol_){
    return std::lgamma((dim + nu) * .5) - lgamma(nu * .5) -
      std::log(nu * M_PI) * dim * .5 - .5 * chol_.log_det();
  }
  const double norm_const_log = set_const(nu, dim, chol_);

  static double check_nu(const double nu){
#ifdef MSSM_DEBUG
    if(nu <= 2.)
      throw invalid_argument("invalid 'nu'");
#endif
    return nu;
  }

  double log_dens_(const double dist) const
  {
    return norm_const_log - (nu + (double)dim) * .5 * std::log1p(dist / nu);
  }

public:
  mv_tdist(const arma::mat &Q, const double nu):
    chol_(Q), mu(nullptr), dim(Q.n_cols), nu(check_nu(nu)) { }
  mv_tdist(const arma::mat &Q, const arma::vec &mu, const double nu):
    chol_(Q), mu(new arma::vec(mu)), dim(Q.n_cols), nu(check_nu(nu)) { }

  double operator()
  (const double *x, const double *y, const arma::uword N,
   const double x_log_w) const override {
    arma::vec x1(x, N), y1(y, N);
    chol_.solve_half(x1);
    chol_.solve_half(y1);
    const double dist = norm_square(x1.begin(), y1.begin(), N);
    return log_dens_(dist) + x_log_w;
  }

  std::array<double, 2> operator()
  (const hyper_rectangle &r1, const hyper_rectangle &r2) const override {
    throw logic_error("'mv_tdist': not implemented");
  }

  arma::uword state_dim() const override {
    return dim;
  }

  double log_density_state
  (const arma::vec &x, arma::vec *gr, arma::mat *H, const compute what)
  const override
  {
    check_input_mv_log_density_state(state_dim(), mu.get(), x, gr, H, what);
    if(what != log_densty)
      throw logic_error("'mv_tdist': not implemented");

    return operator()(x.begin(), mu->begin(), x.n_elem, 0.);
  }

  void sample(arma::mat&) const override;

  double log_prop_dens(const arma::vec &x) const override {
    return log_density_state(x, nullptr, nullptr, cdist::log_densty);
  }

  const arma::mat& mean(){
    if(!mu)
      throw std::logic_error("no mean");
    return *mu;
  }

  arma::mat vCov() const {
    return chol_.X * (nu / (nu - 2.));
  };
};

class exp_family : public cdist {
protected:
  /* outcome */
  const arma::vec &Y;
  /* design matrix for fixed effects */
  const arma::mat &X;
  /* coefficients for fixed effects */
  const arma::vec &cfix;
  /* offset from fixed effects */
  const arma::vec offset = X.t() * cfix;
  /* design matrix for random effects */
  const arma::mat &Z;
  /* dispersion parameter. Default is no dispersion parameter. This function
   * should not allocate memory. */
  virtual arma::vec* set_disp(const arma::vec&) {
    return nullptr;
  };
  const arma::vec *disp;
  /* case weights */
  const arma::vec ws;

  /* Given a linear predictor, computes the log density and potentially
   * the derivatives. */
  virtual std::array<double, 3> log_density_state_inner
    (const double, const double, const compute) const = 0;

public:
  virtual ~exp_family() = default;

  exp_family
  (const arma::vec &Y, const arma::mat &X, const arma::vec &cfix,
   const arma::mat &Z, const arma::vec &disp, const arma::vec *ws):
  Y(Y), X(X), cfix(cfix), Z(Z), disp(set_disp(disp)),
  ws(ws ? arma::vec(*ws) : arma::vec(X.n_cols, arma::fill::ones))
  {
#ifdef MSSM_DEBUG
    if(X.n_cols != Y.n_elem)
      throw invalid_argument("invalid 'X'");
    if(X.n_rows != cfix.n_elem)
      throw invalid_argument("invalid 'cfix'");
    if(X.n_cols != Z.n_cols)
      throw invalid_argument("invalid 'Z'");
    if(X.n_cols != ws->n_elem)
      throw invalid_argument("invalid 'ws'");
#endif
  }

  arma::uword state_dim() const override final {
    return Z.n_rows;
  }

  double log_density_state
    (const arma::vec &x, arma::vec *gr, arma::mat *H, const compute what)
    const override final
  {
    const bool compute_gr = what == gradient or what == Hessian,
      compute_H = what == Hessian;

#ifdef MSSM_DEBUG
    if(x.n_elem != state_dim())
      throw invalid_argument("invalid 'x'");
    if(compute_gr and (!gr or gr->n_elem != state_dim()))
      throw invalid_argument("invalid 'gr'");
    if(compute_H and (!H or H->n_cols != state_dim() or
                        H->n_cols != H->n_rows))
      throw invalid_argument("invalid 'H'");
#endif
    const arma::vec &eta = offset + Z.t() * x;
    const double *e, *w, *y;
    double out = 0.;
    arma::uword i;
    for(i = 0, e = eta.begin(), w = ws.begin(), y = Y.begin();
        i < eta.n_elem; ++i, ++e, ++w, ++y)
    {
      const std::array<double, 3> log_den_eval =
        log_density_state_inner(*y, *e, what);

      out += *w * log_den_eval[0];
      if(compute_gr)
        *gr += *w * log_den_eval[1] * Z.col(i);
      if(compute_H)
        arma_dsyr(*H, Z.unsafe_col(i), *w * log_den_eval[2]);
    }

    if(compute_H)
      *H = arma::symmatu(*H);

    return out;
  }
};

#define EXP_CLASS(fname)                                            \
class fname final : public exp_family {                             \
  std::array<double, 3> log_density_state_inner                     \
  (const double, const double, const compute) const override final; \
public:                                                             \
  using exp_family::exp_family;                                     \
}

EXP_CLASS(binomial_logit);
inline std::array<double, 3> binomial_logit::log_density_state_inner
  (const double y, const double eta, const compute what) const
{
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

#undef MIN
#undef MAX
#undef EXP_CLASS
#endif
