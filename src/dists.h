#ifndef DISTS_H
#define DISTS_H
#include "arma.h"
#include "utils.h"
#include "kd-tree.h"
#include <array>

using std::logic_error;
using std::invalid_argument;
using std::log;
using std::exp;

enum comp_out { log_densty, gradient, Hessian };

inline void gaurd_new_comp_out(const comp_out what){
#ifdef MSSM_DEBUG
  if(what != log_densty and what != gradient and what != Hessian)
    throw std::logic_error("Unkown 'comp_out'");
#endif
}

inline arma::uword get_grad_dim(const arma::uword stat_dim, const comp_out what)
{
  gaurd_new_comp_out(what);

  if(what == gradient)
    return stat_dim;
  else if(what == Hessian){
    /* positive solution to d(d + 1) = k */
    double x = .5 * (std::sqrt((double)stat_dim * 4. + 1.) - 1.);
#ifdef MSSM_DEBUG
    if(x - round(x) >= std::numeric_limits<double>::epsilon())
      throw std::runtime_error("invalid dimension in 'get_grad_dim'");
#endif
    return std::round(x);
  }

  return 0L;
}

inline arma::uword get_hess_dim(const arma::uword stat_dim, const comp_out what)
{
  gaurd_new_comp_out(what);

  if(what == gradient)
    return 0L;
  else if(what == Hessian){
    /* positive solution to d(d + 1) = k */
    double x = .5 * (std::sqrt((double)stat_dim * 4. + 1.) - 1.);
#ifdef MSSM_DEBUG
    if(std::abs(x - round(x)) >= std::numeric_limits<double>::epsilon())
      throw std::runtime_error("invalid dimension in 'get_grad_dim'");
#endif
    return std::lround(x * x);
  }

  return 0L;
}

/* class to compute conditional densities */
class cdist {
public:
  virtual ~cdist() = default;

  /* gets the dimension of the state vector */
  virtual arma::uword state_dim() const = 0;
  /* gets statistics dimension for state statistics */
  virtual arma::uword state_stat_dim(const comp_out) const = 0;
  /* yields the gradient dimension w.r.t. terms involving the old and new
   * state */
  arma::uword state_stat_dim_grad(const comp_out what) const {
    return get_grad_dim(state_stat_dim(what), what);
  }
  /* yields the Hessian dimension w.r.t. terms involving the old and new
   * state */
  arma::uword state_stat_dim_hess(const comp_out what) const {
    return get_hess_dim(state_stat_dim(what), what);
  }

  /* gets statistics dimension for observation statistics */
  virtual arma::uword obs_stat_dim(const comp_out) const = 0;
  /* yields the gradient dimension w.r.t. terms involving the new state only */
  arma::uword obs_stat_dim_grad(const comp_out what) const {
    return get_grad_dim(obs_stat_dim(what), what);
  }
  /* yields the Hessian dimension w.r.t. terms involving the new state only */
  arma::uword obs_stat_dim_hess(const comp_out what) const {
    return get_hess_dim(obs_stat_dim(what), what);
  }

  /* gets dimension of statistics */
  arma::uword stat_dim(const comp_out what) const {
    return state_stat_dim(what) + obs_stat_dim(what);
  }
  /* takes a new state and add the statistics to the third argument.
   * This only includes the part that depends on the new state only. */
  virtual void comp_stats_state_only
    (const arma::vec&, double*, const comp_out) const = 0;
  /* computes the density and gradient and/or Hessian w.r.t. the state
   * vector if requested. */
  virtual double log_density_state
    (const arma::vec&, arma::vec*, arma::mat*, const comp_out) const = 0;
  double log_density_state(const arma::vec &state) const
  {
    return log_density_state(state, nullptr, nullptr, log_densty);
  }
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

  /* functions to be called on matrices of states before calling the other
   * member functions. Some function may be symmetrical in the two random
   * variable arguments after this is called */
  virtual void trans_X(arma::mat&) const = 0;
  virtual void trans_Y(arma::mat&) const = 0;
  /* functions to be called on matrices to undo the above transformations */
  virtual void trans_inv_X(arma::mat&) const = 0;
  virtual void trans_inv_Y(arma::mat&) const = 0;
  /* compute log kernel given two points, the number of elements, and a
   * log weight */
  virtual double operator()
    (const double*, const double*, const arma::uword, const double) const = 0;
  /* compute the smallest and largest log kernel distance between two
   * hyper rectangles */
  virtual std::array<double, 2> operator()
    (const hyper_rectangle&, const hyper_rectangle&) const = 0;
  /* takes the old, new state, and weight of the pair (ignoring
   * a normalization term) and add the requested stat to the third argument.
   * This only includes the part that depends on the pair. */
  virtual void comp_stats_state_state
    (const double*, const double*, const double, double*, const comp_out)
    const = 0;

  /* returns the normalization constant */
  virtual double get_log_norm_const() const = 0;
};

inline void check_input_mv_log_density_state
  (const arma::uword dim, const arma::vec *mu, const arma::vec &x,
   const arma::vec *gr, const arma::mat *H, const comp_out what)
{
  gaurd_new_comp_out(what);
#ifdef MSSM_DEBUG
  if(!mu)
    throw invalid_argument("'mu' not set");
  if(x.n_elem != mu->n_elem)
    throw invalid_argument(
        "invalid 'x' (" + std::to_string(x.n_elem) + " and " +
          std::to_string(mu->n_elem) + " elements)");
  if((what == gradient or what == Hessian) and
       (!gr or gr->n_elem != dim))
    throw invalid_argument("invalid 'gr'");
  if(what == Hessian and (!H or H->n_rows != dim or H->n_cols != dim))
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

  /* trans_obj overrides */
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

  void trans_X(arma::mat&) const override final {
    return;
  }
  void trans_Y(arma::mat&) const override final {
    return;
  }
  void trans_inv_X(arma::mat&) const override final {
    return;
  }
  void trans_inv_Y(arma::mat&) const override final {
    return;
  }

  void comp_stats_state_state
  (const double*, const double*, const double, double*, const comp_out)
  const override final {
    throw logic_error("not implemented");
  }

  double get_log_norm_const() const override final {
    return norm_const_log;
  }

  /* cdist overrides */
  arma::uword state_dim() const override {
    return dim;
  }

  double log_density_state
  (const arma::vec &x, arma::vec *gr, arma::mat *H, const comp_out what)
  const override
  {
    check_input_mv_log_density_state(state_dim(), mu.get(), x, gr, H, what);
    if(what != log_densty)
      throw logic_error("'mvs_norm': not implemented");

    return this->operator()(x.begin(), mu->begin(), x.n_elem, 0.);
  }

  arma::uword state_stat_dim(const comp_out) const override {
    throw logic_error("not implemented");
  }
  arma::uword obs_stat_dim(const comp_out) const override {
    throw logic_error("not implemented");
  }

  void comp_stats_state_only
  (const arma::vec&, double*, const comp_out) const override final {
    return;
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

  /* trans_obj overrides */
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

  void trans_X(arma::mat &X) const override final {
    chol_.solve_half(X);
  }
  void trans_Y(arma::mat &Y) const override final {
    chol_.solve_half(Y);
  }
  void trans_inv_X(arma::mat &X) const override final {
    chol_.mult_half(X);
  }
  void trans_inv_Y(arma::mat &Y) const override final {
    chol_.mult_half(Y);
  }

  void comp_stats_state_state
  (const double*, const double*, const double, double*, const comp_out)
   const override final
  {
    throw logic_error("not implemented");
  }

  double get_log_norm_const() const override final {
    return norm_const_log;
  }

  /* proposal_dist overrides */
  void sample(arma::mat&) const override;

  double log_prop_dens(const arma::vec &x) const override {
    return log_density_state(x, nullptr, nullptr, log_densty);
  }

  /* cdist overrides */
  arma::uword state_dim() const override {
    return dim;
  }

  double log_density_state
  (const arma::vec &x, arma::vec *gr, arma::mat *H, const comp_out what)
  const override
  {
    gaurd_new_comp_out(what);
    check_input_mv_log_density_state(state_dim(), mu.get(), x, gr, H, what);
    if(what == gradient or what == Hessian){
      const arma::vec diff = *mu - x; /* account for - later */
      *gr += chol_.solve(diff);
    }
    if(what == Hessian)
      *H -= chol_.get_inv();

    arma::mat x1 = x, mu1 = *mu; /* mat does not matter as we only need to
                                    to pass a pointer */
    trans_Y(x1);
    trans_X(mu1); /* TODO: do this once */
    return operator()(x1.begin(), mu1.begin(), x.n_elem, 0.);
  }

  arma::uword state_stat_dim(const comp_out) const override {
    throw logic_error("not implemented");
  }
  arma::uword obs_stat_dim(const comp_out) const override {
    throw logic_error("not implemented");
  }

  void comp_stats_state_only
    (const arma::vec&, double*, const comp_out) const override final
  {
    return;
  }

  /* own members */
  const arma::mat& mean() const {
    if(!mu)
      throw std::logic_error("no mean");
    return *mu;
  }

  arma::mat vCov() const {
    return chol_.X;
  }
};

/* multivariate normal distribution with y ~ N(Fx, Q) */
class mv_norm_reg final : public cdist, public trans_obj {
  const LU_fact F;
  const chol_decomp chol_;
  const arma::uword dim;
  const std::unique_ptr<const arma::vec> mu;
  const double norm_const_log =
    -(double)dim / 2. * std::log(2. * M_PI) - .5 * chol_.log_det();

    double log_dens_(const double dist) const
    {
      return norm_const_log - dist / 2.;
    }

  arma::vec *set_mu(const arma::vec& m) const {
    arma::vec *out = new arma::vec(m);
    trans_X(*out);
    return out;
  }

public:
  mv_norm_reg(const arma::mat &F, const arma::mat &Q):
  F(F), chol_(Q), dim(Q.n_cols), mu(nullptr) { }
  mv_norm_reg(const arma::mat &F, const arma::mat &Q, const arma::vec &mu):
  F(F), chol_(Q), dim(Q.n_cols), mu(set_mu(mu)) { }

  /* trans_obj overrides */
  double operator()(
      const double *x, const double *y, const arma::uword N,
      const double x_log_w) const override {
    const double dist = norm_square(x, y, N);
    return log_dens_(dist) + x_log_w;
  }

  std::array<double, 2> operator()
  (const hyper_rectangle &r1, const hyper_rectangle &r2) const override {
    auto dists = r1.min_max_dist(r2);
    return { log_dens_(dists[1L]), log_dens_(dists[0L]) };
  }

  void trans_X(arma::mat &X) const override final {
    X = F.X * X;
    chol_.solve_half(X);
  }
  void trans_Y(arma::mat &Y) const override final {
    chol_.solve_half(Y);
  }
  void trans_inv_X(arma::mat &X) const override final {
    chol_.mult_half(X);
    F.solve(X);
  }
  void trans_inv_Y(arma::mat &Y) const override final {
    chol_.mult_half(Y);
  }

  void comp_stats_state_state
  (const double *x, const double *y, const double log_w, double *stat,
   const comp_out what) const override final;

  double get_log_norm_const() const override final {
    return norm_const_log;
  }

  /* cdist overrides */
  arma::uword state_dim() const override {
    return dim;
  }

  double log_density_state
  (const arma::vec &y, arma::vec *gr, arma::mat *H, const comp_out what)
  const override
  {
#ifdef MSSM_DEBUG
    if(!mu)
      throw std::logic_error("'mu' not set");
    if(what != log_densty)
      throw std::logic_error("not implemented");
#endif

    arma::vec yc = y;
    trans_Y(yc);
    return operator()(mu->memptr(), yc.memptr(), dim, 0.);
  }

  arma::uword state_stat_dim(const comp_out what) const override {
    gaurd_new_comp_out(what);
    if(what == log_densty)
      return 0L;
    if(what == gradient)
      return dim * dim + (dim * (dim + 1L)) / 2L;
    else if(what == Hessian){
      const arma::uword gdim = dim * dim + (dim * (dim + 1L)) / 2L;
      return gdim * (gdim + 1L);
    }
    else
      throw std::logic_error("not implemented");
  }
  arma::uword obs_stat_dim(const comp_out) const override {
    return 0L;
  }

  void comp_stats_state_only
  (const arma::vec&, double*, const comp_out) const override final
  {
    return;
  }

  /* own members */
  const arma::mat mean(const arma::vec &x) const {
    return F.X * x;
  }

  arma::mat vCov() const {
    return chol_.X;
  }
};

/* multivariate t distribution */
class mv_tdist : public cdist, public trans_obj, public proposal_dist {
  /* decomposition of __scale__ matrix */
  const chol_decomp chol_;
  const std::unique_ptr<arma::vec> mu;
  const arma::uword dim;
  const double nu;
  const double norm_const_log = ([&]{
    return std::lgamma((dim + nu) * .5) - lgamma(nu * .5) -
      std::log(nu * M_PI) * dim * .5 - .5 * chol_.log_det();
  })();

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

  /* trans_obj overrides */
  double operator()
  (const double *x, const double *y, const arma::uword N,
   const double x_log_w) const override {
    arma::vec x1(x, N), y1(y, N);
    const double dist = norm_square(x1.begin(), y1.begin(), N);
    return log_dens_(dist) + x_log_w;
  }

  std::array<double, 2> operator()
  (const hyper_rectangle &r1, const hyper_rectangle &r2) const override {
    auto dists = r1.min_max_dist(r2);
    return { log_dens_(dists[1L]), log_dens_(dists[0L]) };
  }

  void trans_X(arma::mat &X) const override final {
    chol_.solve_half(X);
  }
  void trans_Y(arma::mat &Y) const override final {
    chol_.solve_half(Y);
  }
  void trans_inv_X(arma::mat &X) const override final {
    chol_.mult_half(X);
  }
  void trans_inv_Y(arma::mat &Y) const override final {
    chol_.mult_half(Y);
  }

  void comp_stats_state_state
  (const double *x, const double *y, const double log_w, double *stat,
   const comp_out what)
  const override final
  {
    throw logic_error("not implemented");
  }

  double get_log_norm_const() const override final {
    return norm_const_log;
  }

  /* proposal_dist overrides */
  void sample(arma::mat&) const override;

  double log_prop_dens(const arma::vec &x) const override {
    return log_density_state(x, nullptr, nullptr, log_densty);
  }

  /* cdist overrides */
  arma::uword state_dim() const override {
    return dim;
  }

  double log_density_state
  (const arma::vec &x, arma::vec *gr, arma::mat *H, const comp_out what)
  const override
  {
    gaurd_new_comp_out(what);
    check_input_mv_log_density_state(state_dim(), mu.get(), x, gr, H, what);
    if(what != log_densty)
      throw logic_error("'mv_tdist': not implemented");

    arma::mat x1 = x, mu1 = *mu; /* mat does not matter as we only need to
                                    to pass a pointer */
    trans_Y(x1);
    trans_X(mu1); /* TODO: do this once */
    return operator()(x1.begin(), mu1.begin(), x.n_elem, 0.);
  }

  arma::uword state_stat_dim(const comp_out) const override {
    throw logic_error("not implemented");
  }
  arma::uword obs_stat_dim(const comp_out) const override {
    throw logic_error("not implemented");
  }

  void comp_stats_state_only
    (const arma::vec&, double*, const comp_out) const override final
  {
    return;
  }

  /* own members */
  const arma::mat& mean()  const {
    if(!mu)
      throw std::logic_error("no mean");
    return *mu;
  }

  arma::mat vCov() const {
    return chol_.X * (nu / (nu - 2.));
  }
};

#ifdef MSSM_DEBUG
constexpr bool exp_family_do_check () { return true; }
#else
constexpr bool exp_family_do_check () { return false; }
#endif


/* Likely overkill with macro and multiple inheritance would be simpler */
#define EXP_BASE_PROTECTED(fname)                                         \
  /* outcome */                                                           \
  const arma::vec Y;                                                      \
  /* design matrix for fixed effects */                                   \
  const arma::mat X;                                                      \
  /* coefficients for fixed effects. Notice the reference */              \
  const arma::vec &cfix;                                                  \
  mutable arma::vec cfix_cache = cfix;                                    \
  /* design matrix for random effects */                                  \
  const arma::mat Z;                                                      \
  /* case weights */                                                      \
  const arma::vec ws;                                                     \
                                                                          \
  /* offset from fixed effects and offsets */                             \
  const arma::vec offs;                                                   \
  mutable arma::vec lp = offs + X.t() * cfix;                             \
                                                                          \
  /* returns the non-random part of the linear predictor */               \
  mutable std::mutex get_lp_mutex;                                        \
  arma::vec &get_lp() const {                                             \
    /* check whether the coefficients have changed. If so then update the \
     * linear predictor */                                                \
    auto has_changed = [&]{                                               \
      return !std::equal(cfix.begin(), cfix.end(), cfix_cache.begin());   \
    };                                                                    \
                                                                          \
    if(has_changed()){                                                    \
      std::lock_guard<std::mutex> lc(get_lp_mutex);                       \
      if(has_changed()){                                                  \
        lp = offs + X.t() * cfix;                                         \
        cfix_cache = cfix;                                                \
      }                                                                   \
    }                                                                     \
                                                                          \
    return lp;                                                            \
  }                                                                       \
                                                                          \
  /* Given a linear predictor, computes the log density and potentially   \
   * the derivatives. */                                                  \
  virtual std::array<double, 3> log_density_state_inner                   \
    (const double, const double, const comp_out, const double) const = 0;

#define EXP_BASE_PUBLIC(fname)                                            \
  virtual ~fname() = default;                                             \
                                                                          \
  arma::uword state_dim() const override final {                          \
    return Z.n_rows;                                                      \
  }                                                                       \
                                                                          \
  arma::uword state_stat_dim(const comp_out) const override {             \
    return 0L;                                                            \
  }                                                                       \
                                                                          \
  void check_param_udpate() const;                                        \
                                                                          \
  double log_density_state                                                \
    (const arma::vec &x, arma::vec *gr, arma::mat *H,                     \
     const comp_out what) const override final                            \
    {                                                                     \
      if(Y.n_elem < 1L)                                                   \
        return 0.;                                                        \
      check_param_udpate();                                               \
                                                                          \
      gaurd_new_comp_out(what);                                           \
      const bool compute_gr = what == gradient or what == Hessian,        \
        compute_H = what == Hessian;                                      \
                                                                          \
      if(exp_family_do_check()){                                          \
        if(x.n_elem != state_dim())                                       \
          throw invalid_argument("invalid 'x'");                          \
        if(compute_gr and (!gr or gr->n_elem != state_dim()))             \
          throw invalid_argument("invalid 'gr'");                         \
        if(compute_H and (!H or H->n_cols != state_dim() or               \
                            H->n_cols != H->n_rows))                      \
          throw invalid_argument("invalid 'H'");                          \
      }                                                                   \
      const arma::vec eta = get_lp() + Z.t() * x;                         \
      const double *e, *w, *y;                                            \
      double out = 0.;                                                    \
      arma::uword i;                                                      \
      for(i = 0, e = eta.begin(), w = ws.begin(), y = Y.begin();          \
          i < eta.n_elem; ++i, ++e, ++w, ++y)                             \
      {                                                                   \
        const std::array<double, 3> log_den_eval =                        \
          log_density_state_inner(*y, *e, what, *w);                      \
                                                                          \
        out += log_den_eval[0];                                           \
        if(compute_gr)                                                    \
          *gr += log_den_eval[1] * Z.col(i);                              \
        if(compute_H)                                                     \
          arma_dsyr(*H, Z.unsafe_col(i), log_den_eval[2]);                \
      }                                                                   \
                                                                          \
      if(compute_H)                                                       \
        *H = arma::symmatu(*H);                                           \
                                                                          \
      return out;                                                         \
    }

/* exponential family w/o dispersion parameter */
class exp_family_wo_disp : public cdist {
protected:
  EXP_BASE_PROTECTED(exp_family_wo_disp)

public:
  exp_family_wo_disp
  (const arma::vec &Y, const arma::mat &X, const arma::vec &cfix,
   const arma::mat &Z, const arma::vec *ws, const arma::vec &offset):
  Y(Y), X(X), cfix(cfix), Z(Z),
  ws(ws ? arma::vec(*ws) : arma::vec(X.n_cols, arma::fill::ones)),
  offs(offset)
  {
    if(exp_family_do_check()){
      if(X.n_cols != Y.n_elem)
        throw invalid_argument("invalid 'X'");
      if(X.n_rows != cfix.n_elem)
        throw invalid_argument("invalid 'cfix'");
      if(X.n_cols != Z.n_cols)
        throw invalid_argument("invalid 'Z'");
      if(X.n_cols != ws->n_elem)
        throw invalid_argument("invalid 'ws'");
    }
  }

  EXP_BASE_PUBLIC(exp_family_wo_disp)

  arma::uword obs_stat_dim(const comp_out what) const override final {
    gaurd_new_comp_out(what);
    arma::uword out = 0L, n_fixed = X.n_rows;
    if(what == gradient or what == Hessian)
      out += n_fixed;
    if(what == Hessian)
      out += out * out;
    return out;
  }

  void comp_stats_state_only
  (const arma::vec &x, double *out, const comp_out what) const override final {
    if(Y.n_elem < 1L)
      return;

    gaurd_new_comp_out(what);
    const bool compute_gr = what == gradient or what == Hessian,
      compute_H = what == Hessian;
    if(!compute_gr and !compute_H)
      return;

    check_param_udpate();

#ifdef MSSM_DEBUG
    if(x.n_elem != state_dim())
      throw invalid_argument("invalid 'x'");
#endif
    const arma::vec eta = get_lp() + Z.t() * x;
    const double *e, *w, *y;
    const arma::uword p = X.n_rows;
    arma::vec gr(out, p, false);
    const std::unique_ptr<arma::mat> H = ([&]{
      if(compute_H)
        return std::unique_ptr<arma::mat>(
          new arma::mat(out + p, p, p, false));
      return std::unique_ptr<arma::mat>();
    })();
    arma::uword i;
    for(i = 0, e = eta.begin(), w = ws.begin(), y = Y.begin();
        i < eta.n_elem; ++i, ++e, ++w, ++y)
    {
      const std::array<double, 3> log_den_eval =
        log_density_state_inner(*y, *e, what, *w);

      if(compute_gr)
        gr += log_den_eval[1] * X.col(i);
      if(compute_H)
        arma_dsyr(*H, X.unsafe_col(i), log_den_eval[2]);
    }

    if(compute_H)
      *H = arma::symmatu(*H);
  }
};

/* exponential family w/ dispersion parameter */
class exp_family_w_disp : public cdist {
protected:
  EXP_BASE_PROTECTED(exp_family_w_disp)

  /* additional dispersion parameter the `_` version also contains some
   * e.g., traditional transform */
  mutable arma::vec disp;
  const arma::vec &disp_in;
  mutable arma::vec disp_cache;

  /* sets the disperions parameter */
  virtual void set_disp() const = 0;

  /* should be called in methods before using disperion parameter */
  mutable std::mutex disp_mutex;
  void update_disp() const {
    auto has_changed = [&]{
      return arma::size(disp_in) != arma::size(disp_cache) or
        !std::equal(disp_in.begin(), disp_in.end(), disp_cache.begin());
    };

    if(has_changed()){
      std::lock_guard<std::mutex> lc(disp_mutex);
      if(has_changed()){
        set_disp();
        disp_cache = disp_in;
      }
    }
  }

  /* Given a linear predictor, computes the log density and potentially
   * the derivatives. The derivatives are also w.r.t. the dispersion
   * parameter */
  virtual std::array<double, 6> log_density_state_inner_w_disp
    (const double, const double, const comp_out, const double) const = 0;

public:
  exp_family_w_disp
  (const arma::vec &Y, const arma::mat &X, const arma::vec &cfix,
   const arma::mat &Z, const arma::vec *ws, const arma::vec &di,
   const arma::vec &offset):
  Y(Y), X(X), cfix(cfix), Z(Z),
  ws(ws ? arma::vec(*ws) : arma::vec(X.n_cols, arma::fill::ones)),
  offs(offset), disp_in(di)
  {
    if(exp_family_do_check()){
      if(X.n_cols != Y.n_elem)
        throw invalid_argument("invalid 'X'");
      if(X.n_rows != cfix.n_elem)
        throw invalid_argument("invalid 'cfix'");
      if(X.n_cols != Z.n_cols)
        throw invalid_argument("invalid 'Z'");
      if(X.n_cols != ws->n_elem)
        throw invalid_argument("invalid 'ws'");
    }
  }

  EXP_BASE_PUBLIC(exp_family_w_disp)

  arma::uword obs_stat_dim(const comp_out what) const override {
    gaurd_new_comp_out(what);
    arma::uword out = 0L, n_fixed = X.n_rows;
    if(what == gradient or what == Hessian)
      out += n_fixed + 1L;
    if(what == Hessian)
      out += out * out;
    return out;
  }

  void comp_stats_state_only
  (const arma::vec&, double*, const comp_out)
  const override final;
};

#define EXP_CLASS(fname)                                            \
class fname final : public exp_family_wo_disp {                     \
  std::array<double, 3> log_density_state_inner                     \
  (const double, const double, const comp_out, const double)        \
  const override final;                                             \
public:                                                             \
  fname                                                             \
  (const arma::vec &Y, const arma::mat &X, const arma::vec &cfix,   \
   const arma::mat &Z, const arma::vec *ws, const arma::vec &di,    \
   const arma::vec &offset):                                        \
  exp_family_wo_disp(Y, X, cfix, Z, ws, offset) { }                 \
}

#define EXP_CLASS_W_DISP(fname)                                       \
class fname final : public exp_family_w_disp {                        \
  std::array<double, 3> log_density_state_inner                       \
    (const double, const double, const comp_out, const double)        \
    const override final;                                             \
  std::array<double, 6> log_density_state_inner_w_disp                \
    (const double, const double, const comp_out, const double)        \
    const override final;                                             \
public:                                                               \
  using exp_family_w_disp::exp_family_w_disp;                         \
  void set_disp() const override final;                               \
}

EXP_CLASS(binomial_logit);
EXP_CLASS(binomial_cloglog);
EXP_CLASS(binomial_probit);
EXP_CLASS(poisson_log);
EXP_CLASS(poisson_sqrt);
EXP_CLASS_W_DISP(Gamma_log);
EXP_CLASS_W_DISP(gaussian_identity);
EXP_CLASS_W_DISP(gaussian_log);
EXP_CLASS_W_DISP(gaussian_inverse);

std::unique_ptr<cdist> get_family
  (const std::string&, const arma::vec&, const arma::mat&, const arma::vec&,
   const arma::mat&, const arma::vec*, const arma::vec&,const arma::vec&);

#undef EXP_BASE_PROTECTED
#undef EXP_BASE_PUBLIC
#undef EXP_CLASS
#undef EXP_CLASS_W_DISP
#endif
