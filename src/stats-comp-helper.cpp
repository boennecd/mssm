#include "stats-comp-helper.h"
#include "blas-lapack.h"
#include "thread_pool.h"
#include "fast-kernel-approx.h"

static constexpr double D_ONE = 1., D_M_ONE = -1.;
static constexpr int I_ONE = 1L;
static constexpr char C_U = 'U';
using std::ref;
using std::cref;
using namespace std::placeholders;

/* Class to help with computation. The most complicated part is the Hessian
 * approximation. Here we do the following

  Following ``Poyiadjis, G., Doucet, A., \& Singh, S. S. (2011). Particle approximations of the score and observed information matrix in state space models with application to parameter estimation. Biometrika, 98(1), 65-80''
  we need to evaluate

  \begin{align*}
  \tilde W_{ij} &= W_j f(X_t^{(i)} \mid X_{t -1}^{(j)}) \    \
  \tilde W_i &= \sum_j W_j f(X_t^{(i)} \mid X_{t -1}^{(j)}) \\
  \alpha_t^{(i)} &=
  \frac{\sum_j \tilde W_{ij}}{\sum_k \tilde W_{ik}}
  [\nabla g_t(X_t^{(i)}) + \nabla f(X_t^{(i)} \mid X_{t -1}^{(j)})
  + \alpha_{t-1}^{(j)}] \\
  &= \nabla g_t(X_t^{(i)}) +
  \underbrace{\tilde W_i^{-1}
  \sum_j \tilde W_{ij}\underbrace{
  [\nabla f(X_t^{(i)} \mid X_{t -1}^{(j)}) + \alpha_{t-1}^{(j)}]}_{
  \nabla f_{ij}}}_{
  \nabla wf_i} \\
  \beta_t^{(i)} &=
  \frac{\sum_j \tilde W_{ij}}{\sum_k \tilde W_{ik}}[
  (\nabla g_t(X_t^{(i)}) + \nabla f_{ij})
  (\nabla g_t(X_t^{(i)}) + \nabla f_{ij})^\top +
  \nabla^2 g_t(X_t^{(i)}) + \\
  &\hspace{40pt}\underbrace{
  \nabla^2 f(X_t^{(i)} \mid X_{t -1}^{(j)}) + \beta_{t-1}^{(j)}}_{
  \nabla^2f_{ij}}]
  - \alpha_t^{(i)}\alpha_t^{(i)\top} \\
  &= \nabla g_t(X_t^{(i)})\nabla wf_i^\top +
  \nabla wf_i\nabla g_t(X_t^{(i)})^\top + \                         \
  &\hspace{40pt} \nabla g_t(X_t^{(i)})\nabla g_t(X_t^{(i)})^\top + \\
  &\hspace{40pt} \underbrace{\tilde W_i^{-1} \sum_j \tilde W_{ij} (\nabla f_{ij}\nabla f_{ij}^\top
  + \nabla^2 f_{ij})}_{\nabla^2 wf_i} \\
  &\hspace{40pt} \nabla^2 g_t(X_t^{(i)})
  - \alpha_t^{(i)}\alpha_t^{(i)\top}
  \end{align*}%
  %
  First we set%
  %
  \begin{align*}
  a_t^{(i)} &= \sum_j \tilde W_{ij} \nabla f_{ij} \\
  b_t^{(i)} &= \sum_j \tilde W_{ij}(\nabla^2 f_{ij} + \nabla f_{ij}\nabla f_{ij}^\top)
  \end{align*}%
  %
  Then we normalize both of them %
  %
  \begin{align*}
  a_t^{(i)} = \nabla wf_i &\leftarrow \tilde W_i^{-1} a_t^{(i)}  \\
  b_t^{(i)} = \nabla^2 wf_i &\leftarrow \tilde W_i^{-1} b_t^{(i)}
  \end{align*}%
  Then we compute %
  %
  \begin{align*}
  b_t^{(i)} &\leftarrow b_t^{(i)} +
  \nabla g_t(X_t^{(i)})a_t^{(i)\top} + a_t^{(i)} g_t(X_t^{(i)})^\top   \    \
  &\hspace{40pt} + \nabla g_t(X_t^{(i)})\nabla g_t(X_t^{(i)})^\top \        \
  &\hspace{40pt} + \nabla^2 g_t(X_t^{(i)}) \                                \
  a_t^{(i)} = \alpha_t^{(i)} &\leftarrow a_t^{(i)} + \nabla g_t(X_t^{(i)}) \\
  b_t^{(i)} = \beta_t^{(i)} &\leftarrow b_t^{(i)}  - a_t^{(i)}a_t^{(i)\top}
  \end{align*}
 */

/* object to store temporary terms in computation */
thread_local static arma::vec stat_tmp_terms;

class comp_stat_util {
  const comp_out what;

  struct dist {
    const cdist     &di;
    const trans_obj *trans_dist;
    const arma::uword state_stat_dim, obs_stat_dim;
    const int
      state_stat_dim_grad, state_stat_dim_hess,
      obs_stat_dim_grad, obs_stat_dim_hess;

    dist() = delete;
    dist(const cdist &di, const comp_out what):
      di(di),
      trans_dist(dynamic_cast<const trans_obj*>(&di)),
      state_stat_dim(di.state_stat_dim(what)),
      obs_stat_dim(di.obs_stat_dim(what)),

      state_stat_dim_grad(di.state_stat_dim_grad(what)),
      state_stat_dim_hess(di.state_stat_dim_hess(what)),

      obs_stat_dim_grad(di.obs_stat_dim_grad(what)),
      obs_stat_dim_hess(di.obs_stat_dim_hess(what))

      { }
  };
  const std::array<dist, 2L> dists;
  const int stat_dim;

  void state_only_gradient(const arma::vec &state, double *stats) const
  {
    double *stat_i = stats;
    for(auto &d : dists){
      d.di.comp_stats_state_only(state, stat_i, what);
      stat_i += d.state_stat_dim;
      stat_i += d.obs_stat_dim;
    }
  }

  void state_only_Hessian(const arma::vec &state, double *stats) const
  {
    if((int)stat_tmp_terms.size() < stat_dim)
      stat_tmp_terms.set_size(stat_dim);

    /* we only update the upper part of the Hessian. We copy it to the
     * lower part at the end */
    double *tmp_i = stat_tmp_terms.memptr(), *stats_i = stats;
    for(auto &d : dists){
      if(d.obs_stat_dim > 0L){
        std::fill(tmp_i, tmp_i + d.obs_stat_dim, 0.);

        /* compute gradient and Hessian terms */
        d.di.comp_stats_state_only(state, tmp_i, what);

        const int grad_dim = d.obs_stat_dim, hess_dim = d.obs_stat_dim_hess;
        const double *d_g = tmp_i, *dd_g = tmp_i + grad_dim;
        double *d_out = stats_i, *dd_out = stats_i + grad_dim;

        /* first make additions to Hessian */
        /* \nabla g \nabla g^\top */
        F77_CALL(dsyr)(
            &C_U, &grad_dim, &D_ONE, d_g, &I_ONE,
            dd_out, &grad_dim);
        /* \nabla^2 g */
        F77_CALL(daxpy)(
          &hess_dim, &D_ONE, dd_g, &I_ONE, dd_out, &I_ONE);
        /* cross product terms */
        F77_CALL(dsyr2)(
          &C_U, &grad_dim, &D_ONE, d_g, &I_ONE, d_out, &I_ONE,
          dd_out, &grad_dim);

        /* add terms to gradient */
        F77_CALL(daxpy)(
            &grad_dim, &D_ONE, d_g, &I_ONE, d_out, &I_ONE);
      }

      /* subtract outer product of gradient from Hessian. Assumes that the
       * distribution only has either terms that involve the old and new state
       * or only the new state */
      const int gr_dim = d.obs_stat_dim_grad + d.state_stat_dim;
      const double *d_out  = stats_i;
            double *dd_out = stats_i + gr_dim;
      F77_CALL(dsyr)(
          &C_U, &gr_dim, &D_M_ONE, d_out, &I_ONE,
          dd_out, &gr_dim);

      /* copy upper half to lower half. TODO: do this in a smarter way*/
      {
        arma::mat tmp(dd_out, gr_dim, gr_dim, false);
        tmp = arma::symmatu(tmp);
      }

      tmp_i   += d.state_stat_dim;
      stats_i += d.state_stat_dim;
      tmp_i   += d.obs_stat_dim;
      stats_i += d.obs_stat_dim;
    }
  }

  void state_state_gradient
    (const double *state_old, const double *state_new,
     const double *stats_old, double *stats_new, const double log_weight) const
  {
    /* first add old stat */
    const double weight = std::exp(log_weight);
    F77_CALL(daxpy)(
        &stat_dim, &weight, stats_old, &I_ONE, stats_new, &I_ONE);

    /* then compute the terms that is a function of the pair */
    double *stat_i = stats_new;
    for(auto &d : dists){
      if(d.trans_dist)
        d.trans_dist->comp_stats_state_state(
            state_old, state_new, weight, stat_i, what);

      stat_i += d.state_stat_dim;
      stat_i += d.obs_stat_dim;
    }
  }

  void state_state_Hessian
    (const double *state_old, const double *state_new,
     const double *stats_old, double *stats_new, const double log_weight) const
  {
    if((int)stat_tmp_terms.size() < stat_dim)
      stat_tmp_terms.set_size(stat_dim);
    stat_tmp_terms.zeros();

    /* first add old stat */
    F77_CALL(daxpy)(
        &stat_dim, &D_ONE, stats_old, &I_ONE, stat_tmp_terms.memptr(),
        &I_ONE);

    /* then compute the terms that is a function of the pair */
    double *stat_i = stat_tmp_terms.memptr();
    for(auto &d : dists){
      if(d.trans_dist){
        d.trans_dist->comp_stats_state_state(
            state_old, state_new, D_ONE, stat_i, what);

        /* make rank-one update. TODO: maybe use dsyr */
        const int grad_dim = d.state_stat_dim_grad;
        const double *d_f  = stat_i;
              double *dd_f = stat_i + grad_dim;
        F77_CALL(dger)(
          &grad_dim, &grad_dim,
          &D_ONE, d_f, &I_ONE, d_f, &I_ONE,
          dd_f, &grad_dim);
      }

      stat_i += d.state_stat_dim;
      stat_i += d.obs_stat_dim;
    }

    /* add terms */
    const double weight = std::exp(log_weight);
    F77_CALL(daxpy)(
        &stat_dim, &weight, stat_tmp_terms.memptr(), &I_ONE, stats_new,
        &I_ONE);
  }

public:
  const bool any_work = stat_dim > 0L;
  comp_stat_util(const comp_out what, const cdist &d1, const cdist &d2):
  what(what), dists({ dist(d1, what), dist(d2, what) }),
  stat_dim(dists[0].di.stat_dim(what) + dists[1].di.stat_dim(what)) { }

  void state_only(const arma::vec &state, double *stats) const
  {
    gaurd_new_comp_out(what);

    if(!any_work)
      return;

    if(what == gradient)
      state_only_gradient(state, stats);
    else if(what == Hessian)
      state_only_Hessian (state, stats);
  }

  void state_state
  (const double *state_old, const double *state_new,
   const double *stats_old, double *stats_new, const double log_weight) const
  {
    gaurd_new_comp_out(what);
    if(what == Hessian)
      throw std::invalid_argument("'Hessian' not implemeneted with 'comp_stat_util'");

    if(!any_work)
      return;

    if(what == gradient)
      state_state_gradient(state_old, state_new, stats_old, stats_new,
                           log_weight);
    else if (what == Hessian)
      state_state_Hessian(state_old, state_new, stats_old, stats_new,
                          log_weight);
  }
};

inline void set_ll_state_only_
  (const cdist &obs_dist, particle_cloud &new_cloud,
   const comp_stat_util &util, const arma::uword i_start,
   const arma::uword i_end)
{
  const arma::mat &states = new_cloud.particles;
  arma::mat &stats = new_cloud.stats;
  double *log_w = new_cloud.ws.begin() + i_start;

  for(arma::uword i = i_start; i < i_end; ++i, ++log_w){
    *log_w += obs_dist.log_density_state(states.unsafe_col(i));
    util.state_only(states.unsafe_col(i), stats.colptr(i));
  }
}

void stats_comp_helper::set_ll_state_only
  (const cdist &obs_dist, particle_cloud &new_cloud,
   const comp_stat_util &util, thread_pool &pool) const
{
  const arma::uword n_particles = new_cloud.N_particles();
  auto loop_figs = get_inc_n_block(n_particles, pool);

  std::vector<std::future<void> > futures;
  futures.reserve(loop_figs.n_tasks);

  for(arma::uword start = 0L; start < n_particles;){
    arma::uword end = std::min(start + loop_figs.inc, n_particles);
    futures.push_back(pool.submit(std::bind(
        set_ll_state_only_, cref(obs_dist), ref(new_cloud), cref(util),
        start, end)));
    start = end;
  }

  while(!futures.empty()){
    futures.back().get();
    futures.pop_back();
  }
}

void stats_comp_helper::set_ll_n_stat_
  (const problem_data &dat, particle_cloud *old_cloud,
   particle_cloud &new_cloud, const cdist &obs_dist,
   const arma::uword ti) const
{
#ifdef MSSM_DEBUG
  if((!old_cloud) != (ti == 0L))
    throw std::invalid_argument("set_ll_n_stat_: invalid combination of 'ti' and 'old_cloud'");
#endif

  const auto trans_func = dat.get_sta_dist<trans_obj>(ti);
  const cdist *trans_func_dist =
    dynamic_cast<const cdist*>(trans_func.get());
  if(!trans_func_dist)
    throw std::logic_error("'get_sta_dist' did not return a 'cdist'");

  comp_stat_util util(
      dat.ctrl.what_stat, obs_dist, *trans_func_dist);

  /* flip sign of weights. Assumes that they are the log density of the
   * proposal distribution */
  new_cloud.ws *= -1;
  add_back<arma::vec> ad(new_cloud.ws);
  new_cloud.stats.zeros();

  if(old_cloud)
    set_ll_state_state(
      obs_dist, *old_cloud, new_cloud, util, dat.ctrl, *trans_func);
  else {
    const arma::uword n_particles = new_cloud.N_particles();
    /* TODO: do this in parallel? */
    double *log_w = new_cloud.ws.begin();
    for(arma::uword i = 0; i < n_particles; ++i, ++log_w)
      *log_w += trans_func_dist->log_density_state(
        new_cloud.particles.unsafe_col(i));
  }
  set_ll_state_only (
      obs_dist,            new_cloud, util, dat.ctrl.get_pool());
}

void stats_comp_helper::set_ll_n_stat
  (const problem_data &dat, particle_cloud &old_cloud,
   particle_cloud &new_cloud, const cdist &obs_dist,
   const arma::uword ti) const
{
  set_ll_n_stat_(dat, &old_cloud, new_cloud, obs_dist, ti);
}

void stats_comp_helper::set_ll_n_stat
  (const problem_data &dat,
   particle_cloud &new_cloud, const cdist &obs_dist) const
{
  set_ll_n_stat_(dat, nullptr, new_cloud, obs_dist, 0L);
}

inline void set_trans_ll_n_comp_stats_no_aprx
  (particle_cloud &old_cloud, particle_cloud &new_cloud,
   const trans_obj &trans_func, const comp_stat_util &util,
   const arma::uword start, const arma::uword end)
{
  const arma::uword n_old = old_cloud.N_particles(),
    dim_particle = new_cloud.dim_particle();
  arma::vec new_log_ws(n_old);
  for(arma::uword i = start; i < end; ++i){
    const double *d_new = new_cloud.particles.colptr(i);
    double *stats_new = new_cloud.stats.colptr(i),
      *n_w = new_log_ws.begin(),
      max_w = -std::numeric_limits<double>::infinity();

    for(arma::uword j = 0; j < n_old; ++j, ++n_w){
      const double *d_old = old_cloud.particles.colptr(j),
        *stats_old = old_cloud.stats.colptr(j);

      *n_w = trans_func(
        d_old, d_new, dim_particle, old_cloud.ws_normalized(j));

      util.state_state(
        d_old, d_new, stats_old, stats_new, *n_w);
      if(*n_w > max_w)
        max_w = *n_w;
    }

    new_cloud.ws(i) = log_sum_log(new_log_ws, max_w);
  }
}

void stats_comp_helper_no_aprx::set_ll_state_state
  (const cdist &obs_dist, particle_cloud &old_cloud, particle_cloud &new_cloud,
   const comp_stat_util &util, const control_obj &ctrl, const trans_obj &trans_func)
  const
{
  /* transform*/
  trans_func.trans_X(old_cloud.particles);
  trans_func.trans_Y(new_cloud.particles);
  thread_pool &pool = ctrl.get_pool();

  {
    {
      const arma::uword n_particles = new_cloud.N_particles();
      auto loop_figs = get_inc_n_block(n_particles, pool);
      std::vector<std::future<void> > futures;
      futures.reserve(loop_figs.n_tasks);

      for(arma::uword start = 0L; start < n_particles;){
        arma::uword end = std::min(start + loop_figs.inc, n_particles);
        futures.push_back(pool.submit(std::bind(
            set_trans_ll_n_comp_stats_no_aprx, ref(old_cloud), ref(new_cloud),
            cref(trans_func), cref(util), start, end)));
        start = end;
      }

      while(!futures.empty()){
        futures.back().get();
        futures.pop_back();
      }
    }

    /* normalize statistics */
    {
      arma::vec norm_conts = arma::exp(new_cloud.ws);
      new_cloud.stats.each_row() /= norm_conts.t();
    }
  }

  /* transform back */
  trans_func.trans_inv_X(old_cloud.particles);
  trans_func.trans_inv_Y(new_cloud.particles);
}

void stats_comp_helper_aprx_KD::set_ll_state_state
  (const cdist &obs_dist, particle_cloud &old_cloud, particle_cloud &new_cloud,
   const comp_stat_util &util, const control_obj &ctrl,
   const trans_obj &trans_func)
  const
{
  {
    const bool any_work = util.any_work;

    arma::vec &ws = new_cloud.ws,
          &old_ws = old_cloud.ws_normalized;

    ws.fill(-std::numeric_limits<double>::infinity());

    arma::mat &new_particles = new_cloud.particles,
              &old_particles = old_cloud.particles,
              &new_stat = new_cloud.stats,
              &old_stat = old_cloud.stats;

    const arma::uword N_min = ctrl.KD_N_min;
    const double eps = ctrl.aprx_eps;

    thread_pool &pool = ctrl.get_pool();

    auto permu_indices = ([&]{
      if(any_work){
        auto state_state_func = std::bind(
          &comp_stat_util::state_state, &util, _1, _2, _3, _4, _5);

        return FSKA_cpp<true>(
          ws, old_particles, new_particles, old_ws, N_min, eps, trans_func,
          pool, &old_stat, &new_stat, state_state_func);
      }

      return FSKA_cpp<false>(
        ws, old_particles, new_particles, old_ws, N_min, eps, trans_func,
        pool);
    })();

    /* normalize statistics */
    {
      arma::vec norm_conts = arma::exp(ws);
      new_cloud.stats.each_row() /= norm_conts.t();
    }

    /* permutate */
    ws = ws(permu_indices.Y_perm);
    new_particles = new_particles.cols(permu_indices.Y_perm);

    old_ws = old_ws(permu_indices.X_perm);
    old_particles = old_particles.cols(permu_indices.X_perm);

    if(any_work){
      new_stat = new_stat.cols(permu_indices.Y_perm);
      old_stat = old_stat.cols(permu_indices.X_perm);
    }
  }
}

