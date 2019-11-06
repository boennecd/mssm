#include "stats-comp-helper.h"
#include "blas-lapack.h"
#include "thread_pool.h"
#include "fast-kernel-approx.h"
#include "misc.h"

static constexpr double D_ONE = 1., D_M_ONE = -1.;
static constexpr int I_ONE = 1L;
static constexpr char C_L = 'L';
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

class comp_stat_util {
public:
  const comp_out what;
private:

  struct dist_util {
    const comp_out what;
    const cdist     &di;
    const trans_obj *trans_dist;
    const int
      grad_dim  = trans_dist ? di.state_stat_dim_grad(what) :
                               di.obs_stat_dim_grad  (what),
      hess_size = trans_dist ? di.state_stat_dim_hess(what) :
                               di.obs_stat_dim_hess  (what),
      total_size = grad_dim + hess_size;

    dist_util() = delete;
    dist_util(const cdist &di, const comp_out what):
      what(what), di(di),
      trans_dist(dynamic_cast<const trans_obj*>(&di))
      { }
  };
  const dist_util dobs, dstat;
public:
  const int stat_dim, grad_dim = dobs.grad_dim + dstat.grad_dim;

private:
  void state_only_gradient(const arma::vec &state, double *stats) const
  {
    dobs.di.comp_stats_state_only(state, stats, what);
  }

  void state_only_Hessian(const arma::vec &state, double *stats) const
  {
    M_THREAD_LOCAL std::vector<double> stat_tmp_terms;
    if((int)stat_tmp_terms.size() < dobs.total_size)
      stat_tmp_terms.resize(dobs.total_size);

    /* we only compute the lower part of the Hessian. We copy it to the
     * upper part at the end. First, we create pointer to the different
     * elements */
    const int obs_grad_dim = dobs.grad_dim;
    double * const grad_obs   = stats,
           * const grad_state = stats + obs_grad_dim,
           * const hess_obs   = stats + grad_dim,
           * const hess_cross = stats + grad_dim + obs_grad_dim;

    /* we compute the new gradient and hessian terms */
    double * tmp_mem = stat_tmp_terms.data();
    std::fill(tmp_mem, tmp_mem + dobs.total_size, 0.);
    dobs.di.comp_stats_state_only(state, tmp_mem, what);
    const double * const dg  = tmp_mem,
                 * const dgg = tmp_mem + obs_grad_dim;

    /* compute the outer products which we need to add to the Hessian.
     * \nabla g \nabla g^\top */
    dsyr(
        &C_L, &obs_grad_dim, &D_ONE, dg, &I_ONE,
        hess_obs, &grad_dim);

    /* next, we do the two outer products to the upper right block*/
    dsyr2(
        &C_L, &obs_grad_dim, &D_ONE, dg, &I_ONE, grad_obs, &I_ONE,
        hess_obs, &grad_dim);

    /* then the outer product in the lower left block matrix */
    dger(
      &dstat.grad_dim, &obs_grad_dim, &D_ONE, grad_state, &I_ONE,
      dg, &I_ONE, hess_cross, &grad_dim);

    /* add the gradient and Hessian terms themself */
    {
      const double *x = dgg;
      double *y = hess_obs;
      for(int i = 0; i < obs_grad_dim;
          ++i, x += obs_grad_dim, y += grad_dim)
        daxpy(
            &obs_grad_dim, &D_ONE, x, &I_ONE, y, &I_ONE);
    }
    daxpy(
      &obs_grad_dim, &D_ONE, dg, &I_ONE, grad_obs, &I_ONE);

    /* subtract outer product of gradient from the Hessian */
    dsyr(
        &C_L, &grad_dim, &D_M_ONE, stats, &I_ONE,
        stats + grad_dim, &grad_dim);

    /* copy upper half to lower half. TODO: do this in a smarter way */
    {
      arma::mat tmp(stats + grad_dim, grad_dim, grad_dim, false);
      tmp = arma::symmatl(tmp);
    }
  }

  void state_state_gradient
    (const double *state_old, const double *state_new,
     const double *stats_old, double *stats_new, const double log_weight) const
  {
    /* first add old stat */
    const double weight = std::exp(log_weight);
    daxpy(
        &stat_dim, &weight, stats_old, &I_ONE, stats_new, &I_ONE);

    /* then compute the terms that is a function of the pair of the states */
    double * stat_i = stats_new + dobs.grad_dim;
    dstat.trans_dist->comp_stats_state_state(
        state_old, state_new, weight, stat_i, what);
  }

  void state_state_Hessian
    (const double *state_old, const double *state_new,
     const double *stats_old, double *stats_new, const double log_weight) const
  {
    M_THREAD_LOCAL std::vector<double> stat_tmp_terms;
    unsigned int needed_size = stat_dim + dstat.total_size;
    if(stat_tmp_terms.size() < needed_size)
      stat_tmp_terms.resize(needed_size);
    double * const stat_out = stat_tmp_terms.data();
    std::fill(stat_out, stat_out + needed_size, 0.);

    /* first add old stat */
    daxpy(
        &stat_dim, &D_ONE, stats_old, &I_ONE, stat_out, &I_ONE);

    /* then compute the terms that is a function of the pair of the states */
    double * const grad_start = stat_out + dobs.grad_dim,
           * const hess_start = stat_out + grad_dim * (1L + dobs.grad_dim) +
                                dobs.grad_dim,
           * const tmp_mem    = stat_out + stat_dim;
    const int state_grad_dim = dstat.grad_dim;
    {
      dstat.trans_dist->comp_stats_state_state(
        state_old, state_new, 1., tmp_mem, what);
      /* add gradient terms */
      daxpy(
          &state_grad_dim, &D_ONE, tmp_mem, &I_ONE, grad_start, &I_ONE);
      /* add hessian terms */
      const double *t = tmp_mem + state_grad_dim;
            double *x = hess_start;
      for(int i = 0; i < state_grad_dim;
          ++i, t += state_grad_dim, x += grad_dim)
        daxpy(
            &state_grad_dim, &D_ONE, t, &I_ONE, x, &I_ONE);
    }

    /* make rank-one update. Use dsyr and assume we update the upper part
     * later */
    dsyr(
        &C_L, &grad_dim,
        &D_ONE, stat_out, &I_ONE,
        stat_out + grad_dim, &grad_dim);

    /* add terms */
    const double weight = std::exp(log_weight);
    daxpy(
        &stat_dim, &weight, stat_out, &I_ONE, stats_new, &I_ONE);
  }

public:
  const bool any_work = stat_dim > 0L;
  comp_stat_util(const comp_out what, const cdist &d1, const cdist &d2):
  what(what), dobs(d1, what), dstat(d2, what),
  stat_dim(([&]{
    unsigned int out = dobs.grad_dim + dstat.grad_dim;
    gaurd_new_comp_out(what);

    if(what != Hessian)
      return out;

    out = out * (1L + out);
    return out;
  }())) { }

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
    util.state_only(
      states.unsafe_col(i),
      /* avoid UBSAN error */
      (util.what == log_densty) ? nullptr : stats.colptr(i));
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

#ifdef MSSM_DEBUG
  auto gen_err_msg = []
  (const unsigned expected_size, const unsigned actual_size){
    throw std::runtime_error("incorrect 'util.stat_dim'. Size is " +
                             std::to_string(actual_size) + " but expected " +
                             std::to_string(expected_size));
  };

  if(util.stat_dim != (int)new_cloud.dim_stats())
    gen_err_msg(util.stat_dim, new_cloud.dim_stats());
  if(old_cloud and util.stat_dim != (int)old_cloud->dim_stats())
    gen_err_msg(util.stat_dim, old_cloud->dim_stats());
#endif

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
    double *stats_new =
      (util.what == log_densty) ? nullptr : new_cloud.stats.colptr(i),
      *n_w = new_log_ws.begin(),
      max_w = -std::numeric_limits<double>::infinity();

    for(arma::uword j = 0; j < n_old; ++j, ++n_w){
      const double
        *d_old = old_cloud.particles.colptr(j),
        *stats_old =
        (util.what == log_densty) ? nullptr : old_cloud.stats.colptr(j);

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
    if(new_cloud.stats.n_elem > 0L){
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
          pool, false, &old_stat, &new_stat, state_state_func);
      }

      return FSKA_cpp<false>(
        ws, old_particles, new_particles, old_ws, N_min, eps, trans_func,
        pool);
    })();

    /* normalize statistics */
    if(new_cloud.stats.n_elem > 0L){
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

