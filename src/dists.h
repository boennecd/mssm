#ifndef DISTS_H
#define DISTS_H
#include "arma.h"
#include "utils.h"
#include "kd-tree.h"
#include <array>

using std::logic_error;
using std::invalid_argument;

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
    (const arma::vec&, arma::vec*, arma::mat*, compute) const = 0;
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
  (const arma::vec &x, arma::vec *gr, arma::mat *H, compute what)
  const override
  {
#ifdef MSSM_DEBUG
    if(!mu or x.n_elem != mu->n_elem)
      throw invalid_argument("invalid 'x' or 'mu'");
#endif
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
  (const arma::vec &x, arma::vec *gr, arma::mat *H, compute what)
  const override
  {
#ifdef MSSM_DEBUG
    if(!mu or x.n_elem != mu->n_elem)
      throw invalid_argument("invalid 'x' or 'mu'");
#endif
    if(what != log_densty)
      throw logic_error("'mv_norm': not implemented");

    return operator()(x.begin(), mu->begin(), x.n_elem, 0.);
  }

  void sample(arma::mat&) const override;

  double log_prop_dens(const arma::vec &x) const override {
    return log_density_state(x, nullptr, nullptr, cdist::log_densty);
  }
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
  (const arma::vec &x, arma::vec *gr, arma::mat *H, compute what)
  const override
  {
#ifdef MSSM_DEBUG
    if(!mu or x.n_elem != mu->n_elem)
      throw invalid_argument("invalid 'x' or 'mu'");
#endif
    if(what != log_densty)
      throw logic_error("'mv_tdist': not implemented");

    return operator()(x.begin(), mu->begin(), x.n_elem, 0.);
  }

  void sample(arma::mat&) const override;

  double log_prop_dens(const arma::vec &x) const override {
    return log_density_state(x, nullptr, nullptr, cdist::log_densty);
  }
};

#endif
