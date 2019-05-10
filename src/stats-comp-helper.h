#ifndef STATS_COMP_HELPER_H
#define STATS_COMP_HELPER_H
#include "problem_data.h"
#include "dists.h"
#include "cloud.h"

class comp_stat_util;

class stats_comp_helper {
protected:
  /* does the computation that only depends on the present state */
  void set_ll_state_only
    (const cdist&, particle_cloud&, const comp_stat_util&,
     thread_pool&) const;

  /* does the computation that depends on the old and new particle cloud */
  virtual void set_ll_state_state
    (const cdist&, particle_cloud&, particle_cloud&, const comp_stat_util&,
     const control_obj&, const trans_obj&) const = 0;

  void set_ll_n_stat_
    (const problem_data&, particle_cloud*, particle_cloud&,
     const cdist&, const arma::uword) const;
public:
  virtual ~stats_comp_helper() = default;

  /* compute the needed conditional log likelihoods and the requested
   * statistics.
   * It assumes that the dimension of the statistic is correct and that the
   * present unnormalized weights contain the log density of the proposal
   * distribution. It may permutate both particle clouds and it does not
   * normalize the weights */
  void set_ll_n_stat
  (const problem_data&, particle_cloud&, particle_cloud&,
   const cdist&, const arma::uword) const;

  /* same as above but to be used at the first time point */
  void set_ll_n_stat
    (const problem_data&,                particle_cloud&,
     const cdist&) const;
};

/* return an object that does the all O(N^2) computations where N is the
 * number of particles */
class stats_comp_helper_no_aprx final : public stats_comp_helper {
protected:
  void set_ll_state_state
  (const cdist&, particle_cloud&, particle_cloud&, const comp_stat_util&,
   const  control_obj&, const trans_obj&) const final override;
};

/* return an object that makes an O(N log(N)) time approximation */
class stats_comp_helper_aprx_KD final : public stats_comp_helper {
protected:
  void set_ll_state_state
  (const cdist&, particle_cloud&, particle_cloud&, const comp_stat_util&,
   const control_obj&, const trans_obj&) const final override;
};

#endif
