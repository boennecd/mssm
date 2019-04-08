#include "samplers.h"
#include "proposal_dist.h"

inline particle_cloud sample_util
  (const proposal_dist &dist, const problem_data &prob,
   const cdist &state_dist, const cdist &obs_dist)
{
  const comp_out what = prob.ctrl.what_stat;
  const arma::uword dim_state = state_dist.state_dim(),
    stat_dim = state_dist.stat_dim(what) + obs_dist.stat_dim(what);
  particle_cloud out(prob.ctrl.N_part, dim_state, stat_dim);

  dist.sample(out.particles);
  double *w;
  arma::uword i;
  for(i = 0, w = out.ws.begin(); i < prob.ctrl.N_part; ++i, ++w)
    *w = dist.log_prop_dens(out.particles.col(i));

  return out;
}

class bootstrap_sampler final : public sampler {
  particle_cloud smp_inner
  (const problem_data &prob, const arma::uword ti, const arma::vec &old_mean,
   const cdist &obs_dist)
  const
  {
    std::unique_ptr<cdist> state_dist = prob.get_obs_dist(ti);
    mv_norm_reg *dist = dynamic_cast<mv_norm_reg*>(state_dist.get());
    if(!dist)
      throw std::logic_error("not 'mv_norm_reg' pointer");

    const std::unique_ptr<proposal_dist> sampler_((
        [&]() -> proposal_dist* {
          arma::mat vCov = dist->vCov();
          arma::vec mu = dist->mean(old_mean);
          const double covar_fac = prob.ctrl.covar_fac;
          if(covar_fac != 1.)
            vCov *= covar_fac;
          const double nu = prob.ctrl.nu;
          if(nu <= 2.)
            return new mv_norm(vCov, mu);

          vCov *= ((nu - 2.) / nu);
          return new mv_tdist(vCov, mu, nu);
        })());

    return sample_util(*sampler_, prob, *state_dist, obs_dist);
  }

public:
  particle_cloud sample_first
  (const problem_data &prob, const cdist &obs_dist) const override final {
    return smp_inner(prob, 0L, prob.mu0, obs_dist);
  }
  particle_cloud sample
  (const problem_data &prob, const cdist &obs_dist, const particle_cloud &old_cl,
   const arma::uword ti)
  const override final
  {
    arma::vec old_mean = old_cl.get_cloud_mean();
    return smp_inner(prob, ti, old_mean, obs_dist);
  }
};

std::unique_ptr<sampler> get_bootstrap_sampler(){
  return std::unique_ptr<sampler>(new bootstrap_sampler());
}

class mode_aprx_sampler final : public sampler {
  particle_cloud smp_inner
  (const problem_data &prob, const arma::uword ti, const arma::vec &old_mean,
   const cdist &obs_dist)
  const
  {
    std::unique_ptr<cdist> state_dist = prob.get_obs_dist(ti);
    mv_norm_reg *dist = dynamic_cast<mv_norm_reg*>(state_dist.get());
    if(!dist)
      throw std::logic_error("not 'mv_norm_reg' pointer");

    const std::unique_ptr<proposal_dist> sampler_ = ([&]{
      arma::vec start = dist->mean(old_mean);
      arma::mat Q = dist->vCov();
      mv_norm dist_state(Q, start);

      auto out = mode_approximation(
      { &obs_dist, &dist_state }, start, prob.ctrl.nu, prob.ctrl.covar_fac,
      prob.ctrl.ftol_rel);

      if(out.any_errors)
        throw std::runtime_error("'mode_approximation' failed");

      return std::move(out.proposal);
    })();

    return sample_util(*sampler_, prob, *state_dist, obs_dist);
  }

public:
  particle_cloud sample_first
  (const problem_data &prob, const cdist &obs_dist) const override final {
    return smp_inner(prob, 0L, prob.mu0, obs_dist);
  }
  particle_cloud sample
  (const problem_data &prob, const cdist &obs_dist, const particle_cloud &old_cl,
   const arma::uword ti)
  const override final
  {
    arma::vec old_mean = old_cl.get_cloud_mean();
    return smp_inner(prob, ti, old_mean, obs_dist);
  }
};

std::unique_ptr<sampler> get_mode_aprx_sampler(){
  return std::unique_ptr<sampler>(new mode_aprx_sampler());
}
