#include "PF.h"

void accout_for_obs_inner
  (const cdist &dist, const arma::uword start, const arma::uword end,
   particle_cloud &new_cloud)
{
  for(arma::uword i = start; i < end; ++i)
    /* log conditional distribution of outcome minus log proposal
    * distribution density */
    new_cloud.ws(i) =
    dist.log_density_state(new_cloud.particles.unsafe_col(i)) -
      new_cloud.ws(i);
}

void accout_for_obs
  (thread_pool &pool, particle_cloud &new_cloud, const cdist &dist_t)
{
  const arma::uword n_part = new_cloud.N_particles(),
    n_threads = pool.thread_count;
  const arma::uword n_task = (n_threads + n_part - 1L) / n_threads;
  std::vector<std::future<void>> tasks;
  tasks.reserve(n_task);
  const arma::uword inc = n_part / n_threads + 1L;

  arma::uword stop = 0L;
  for(arma::uword start = 0; start < n_part;){
    start = stop;
    stop = std::min(n_part, stop + inc);
    tasks.push_back(pool.submit(std::bind(
        accout_for_obs_inner, std::cref(dist_t), start, stop,
        std::ref(new_cloud))));
  }

  while(!tasks.empty()){
    tasks.back().get();
    tasks.pop_back();
  }
}

std::vector<particle_cloud> PF
  (const problem_data &prob, const sampler &samp, const trans_comp &trans)
{
  std::vector<particle_cloud> out;
  out.reserve(prob.n_periods);
  thread_pool &pool = prob.ctrl.get_pool();

  for(arma::uword i = 0; i < prob.n_periods; ++i){
    /* get conditional distribution at time i */
    std::unique_ptr<cdist> dist_t = prob.get_obs_dist(i);

    /* sample new cloud */
    if(i == 0)
      out.emplace_back(samp.sample_first(prob, *dist_t));
    else
      out.emplace_back(samp.sample      (prob, *dist_t, out.back(), i));

    particle_cloud &new_cloud = out.back();

    /* Update weights to account for previous states and update statistics */
    if(i == 0)
      trans.set      (prob, *(out.rbegin() + 1), out.back(), *dist_t);
    else
      trans.set_first(prob,                      out.back(), *dist_t);

    /* update weights to account for outcome */
    accout_for_obs(pool, new_cloud, *dist_t);

    /* normalize weights */
    // TODO: implement
  }

  return out;
}
