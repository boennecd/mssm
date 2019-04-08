#ifndef SAMPLERS_H
#define SAMPLERS_H
#include "problem_data.h"
#include "cloud.h"

class sampler {
public:
  /* sample new states and sets the log weights equal to the log proposal
   * distribution density */
  virtual particle_cloud sample_first
  (const problem_data&, const cdist&) const = 0;
  virtual particle_cloud sample
    (const problem_data&, const cdist&, const particle_cloud&,
     const arma::uword) const = 0;

  virtual ~sampler() = default;
};

std::unique_ptr<sampler> get_bootstrap_sampler();
std::unique_ptr<sampler> get_mode_aprx_sampler();

#endif
