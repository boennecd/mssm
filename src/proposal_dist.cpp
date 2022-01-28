#include "proposal_dist.h"
#include <nloptrAPI.h>

using cdist_vec = std::initializer_list<const cdist*>;

inline double mode_objective(
    unsigned int n, const double *x, double *grad, void *data_in)
{
  cdist_vec *data = (cdist_vec*) data_in;
  arma::vec state(x, n);

  comp_out what;
  std::unique_ptr<arma::vec> gr(nullptr);
  const bool do_grad = grad;
  if(do_grad){
    gr.reset(new arma::vec(grad, n, false));
    gr->zeros();
    what = gradient;

  } else
    what = log_densty;

  double o = 0.;
  std::unique_ptr<arma::vec> gr_new(nullptr);
  if(do_grad)
    gr_new.reset(new arma::vec(gr->n_elem, arma::fill::none));
  for(auto d = data->begin(); d != data->end(); ++d){
    if(do_grad)
      gr_new->zeros();

    o += (*d)->log_density_state(state, gr_new.get(), nullptr, what);

    if(do_grad)
      *gr += *gr_new;
  }

  return o;
}

mode_approximation_output mode_approximation
  (cdist_vec cdists, const arma::vec &start,
   const double nu, const double covar_fac, const double ftol_rel)
{
#ifdef MSSM_DEBUG
  if(nu <= 2. and nu != -1.)
    throw std::invalid_argument("invalid 'nu'");
  if(covar_fac <= 0.)
    throw std::invalid_argument("invalid 'covar_fac'");
  if(ftol_rel <= 0.)
    throw std::invalid_argument("invalid 'ftol_rel'");
  if(cdists.size() < 1L)
    throw std::invalid_argument("invalid 'cdists'");
#endif

  mode_approximation_output out;

  arma::vec val = start;
  const arma::uword n = (*cdists.begin())->state_dim();
  {
    /* find mode */
    nlopt_opt opt;
    opt = nlopt_create(NLOPT_LD_SLSQP, n);
    nlopt_set_max_objective(opt, mode_objective, &cdists);
    nlopt_set_ftol_rel(opt, ftol_rel);
    nlopt_set_maxeval(opt, 10000L);

    double maxf;
    int nlopt_result_code = nlopt_optimize(opt, val.memptr(), &maxf);
    nlopt_destroy(opt);
    out.any_errors = nlopt_result_code < 1L or nlopt_result_code > 4L;
  }

  arma::vec g(n, arma::fill::zeros);
  arma::mat H(n, n, arma::fill::zeros);
  for(auto c : cdists)
    c->log_density_state(val, &g, &H, Hessian);

  arma::mat vCov = -H.i();
  if(covar_fac != 1.)
    vCov *= covar_fac;

  if(nu <= 2.){
    /* return multivariate normal distribution */
    out.proposal.reset(new mv_norm(vCov, val));
    return out;
  }

  /* scale to get same covariance matrix */
  vCov *= (nu - 2.) / nu;
  out.proposal.reset(new mv_tdist(vCov, val, nu));
  return out;
}
