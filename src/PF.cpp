#include "PF.h"

std::vector<particle_cloud> PF
  (const problem_data &prob, const sampler &samp, const stats_comp_helper &trans)
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

    /* Update weights ad set stats */
    if(i > 0)
      trans.set_ll_n_stat(
        prob, *(out.rbegin() + 1), new_cloud, *dist_t, i);
    else
      trans.set_ll_n_stat(
        prob,                      new_cloud, *dist_t   );

    /* normalize weights */
    // TODO: implement
  }

  return out;
}
