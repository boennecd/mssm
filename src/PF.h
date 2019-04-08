#ifndef PF_H
#define PF_H
#include "cloud.h"
#include "problem_data.h"
#include "samplers.h"

class trans_comp {
public:
  /* update log weights to account for transition from previous periods.
   * Also sets the stats on the third argument */
  virtual void set
  (const problem_data&, const particle_cloud&, particle_cloud&,
   const cdist&) const = 0;
  virtual void set_first
  (const problem_data&,                        particle_cloud&,
   const cdist&) const = 0;
};

std::vector<particle_cloud> PF
  (const problem_data&, const sampler&, const trans_comp&);

#endif
