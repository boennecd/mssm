#ifndef DISTS_H
#define DISTS_H
#include "arma.h"
#include "utils.h"
#include "kd-tree.h"
#include <array>

using std::logic_error;

/* class to compute conditional densities */
class cdist {
public:
  virtual ~cdist() = default;

  enum compute { log_densty, gradient, Hessian };

  /* gets the dimension of the state vector */
  virtual arma::uword state_dim() const {
    throw logic_error("not implemented");
  }
  /* computes the density and gradient and/or Hessian w.r.t. the state
   * vector if requested. */
  virtual double log_density_state(
      const arma::vec&, arma::vec*, arma::mat*, compute) const {
    throw logic_error("not implemented");
  }
  /* Samples from the distribution */
  virtual arma::vec sample() const {
    throw logic_error("not implemented");
  }
};

/* class to be used for kernal methods */
class trans_obj {
public:
  virtual ~trans_obj() = default;

  /* compute log kernel given two points, the number of elements, and a
   * log weight */
  virtual double operator()
    (const double*, const double*, const arma::uword, const double) const {
    throw logic_error("not implemented");
  }
  /* compute the smallest and largest log kernel distance between two
   * hyper rectangles */
  virtual std::array<double, 2> operator()
    (const hyper_rectangle&, const hyper_rectangle&) const {
    throw logic_error("not implemented");
  }
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
  (const arma::vec &x, arma::vec *gr, arma::mat *H, compute what) const final
  {
#ifdef MSSM_DEBUG
    if(!mu or x.n_elem != mu->n_elem)
      throw std::invalid_argument("invalid 'x' or 'mu'");
#endif
    if(what != log_densty)
      throw logic_error("not implemented");

    return this->operator()(x.begin(), mu->begin(), x.n_elem, 0.);
  }
};

#endif
