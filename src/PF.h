#ifndef PF_H
#define PF_H
#include "cloud.h"
#include "problem_data.h"
#include "stats-comp-helper.h"
#include "samplers.h"

std::vector<particle_cloud> PF
  (const problem_data&, const sampler&, const stats_comp_helper&);

#endif
