#include "stats-comp-helper.h"
#include "blas-lapack.h"
#include "thread_pool.h"

static constexpr int I_ONE = 1L;
using std::ref;
using std::cref;

class comp_stat_util {
  const comp_out what;

  struct dist {
    const cdist     &di;
    const trans_obj *trans_dist;
    const arma::uword state_stat_dim, obs_stat_dim;

    dist() = delete;
    dist(const cdist &di, const comp_out what):
      di(di),
      trans_dist(dynamic_cast<const trans_obj*>(&di)),
      state_stat_dim(di.state_stat_dim(what)),
      obs_stat_dim(di.obs_stat_dim(what)) { }
  };
  const std::array<dist, 2L> dists;
  const int stat_dim;
  const bool any_work = stat_dim > 0L;

public:
  comp_stat_util(const comp_out what, const cdist &d1, const cdist &d2):
  what(what), dists({ dist(d1, what), dist(d2, what) }),
  stat_dim(dists[0].di.stat_dim(what) + dists[1].di.stat_dim(what)) { }

  void state_only(const arma::vec &state, double *stats) const
  {
    gaurd_new_comp_out(what);
    if(what == Hessian)
      throw std::invalid_argument("'Hessian' not implemeneted with 'comp_stat_util'");

    if(!any_work)
      return;

    double *stat_i = stats;
    for(auto &d : dists){
      d.di.comp_stats_state_only(state, stat_i, what);
      stat_i += d.state_stat_dim;
      stat_i += d.obs_stat_dim;
    }
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

    /* first add old stat */
    const double weight = std::exp(log_weight);
    F77_CALL(daxpy)(
        &stat_dim, &log_weight, stats_old, &I_ONE, stats_new, &I_ONE);

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

  new_cloud.stats.zeros();

  if(old_cloud)
    set_ll_state_state(
      obs_dist, *old_cloud, new_cloud, util, dat.ctrl.get_pool(), *trans_func);
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
  for(arma::uword i = start; i < end; ++i){
    const double *d_new = new_cloud.particles.colptr(i);
    double * stats_new = new_cloud.stats.colptr(i);
    double &w = new_cloud.ws(i);

    for(arma::uword j = 0; j < n_old; ++j){
      const double *d_old = old_cloud.particles.colptr(j),
        *stats_old = old_cloud.stats.colptr(j);

      double new_term = trans_func(
        d_old, d_new, dim_particle, old_cloud.ws_normalized[j]);
      w += new_term;

      util.state_state(
        d_old, d_new, stats_old, stats_new, new_term);
    }
  }
}


void stats_comp_helper_no_aprx::set_ll_state_state
  (const cdist &obs_dist, particle_cloud &old_cloud, particle_cloud &new_cloud,
   const comp_stat_util &util, thread_pool &pool, const trans_obj &trans_func)
  const
  {
    /* transform*/
    trans_func.trans_X(old_cloud.particles);
    trans_func.trans_Y(new_cloud.particles);

    /* copy old log weights. We need this to do normalization later */
    const arma::vec old_w = new_cloud.ws;

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
      arma::vec norm_conts = arma::exp(new_cloud.ws - old_w);
      new_cloud.stats.each_row() /= norm_conts;
    }

    /* transform back */
    trans_func.trans_inv_X(old_cloud.particles);
    trans_func.trans_inv_Y(new_cloud.particles);
  }
