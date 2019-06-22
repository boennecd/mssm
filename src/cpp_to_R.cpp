#include "fast-kernel-approx.h"
#include "utils.h"
#include "dists.h"
#include "PF.h"
#include "laplace.h"
#include "smoother.h"

#ifdef MSSM_PROF
#include "profile.h"
#endif

using Rcpp::Named;

// [[Rcpp::export]]
Rcpp::List test_KD_note(const arma::mat &X, const arma::uword N_min){
  thread_pool pool(1L);
  KD_note root = get_KD_tree(X, N_min, pool);

  /* find leafs */
  auto leafs = root.get_leafs();
  arma::uvec n_elems(leafs.size());
  arma::uvec indices(X.n_cols);

  auto n_el = n_elems.begin();
  auto idx = indices.begin();
  for(auto l : leafs){
    auto l_indices = l->get_indices();
    *(n_el++) = l_indices.size();

    for(auto l_indices_i : l_indices)
      *(idx++) = l_indices_i;
  }

  return Rcpp::List::create(
    Rcpp::Named("indices") = std::move(indices),
    Rcpp::Named("n_elems") = std::move(n_elems));
}

/* TODO: could use std::bind */
struct naive_inner_loop {
  static constexpr std::size_t N_outer = 10L;

  const arma::uword i_start, i_end;
  const arma::vec &ws_log;
  const arma::mat &X, &Y;
  const trans_obj &kernel;
  arma::vec &out;

  arma::mat weights_inner;
  const arma::uword N;

  naive_inner_loop
    (const arma::uword i_start, const arma::uword i_end,
     const arma::vec &ws_log, const arma::mat &X, const arma::mat &Y,
     const trans_obj &kernel, arma::vec &out):
    i_start(i_start), i_end(i_end), ws_log(ws_log), X(X), Y(Y),
    kernel(kernel), out(out), weights_inner(X.n_cols, N_outer), N(Y.n_rows) { }

  void operator()(){
    static constexpr std::size_t N_X = 10L;
    loop_nest_util<N_outer, N_X> loop_util(i_end - i_start, X.n_cols);

    std::array<double, N_outer> max_ws;

    for(unsigned int k = 0; k < loop_util.N_it; ++k){
      auto dat = loop_util();
      const bool is_new_it = dat.inner_start < 1L,
        is_final_inner_it = dat.inner_end >= X.n_cols;
      if(is_new_it)
        max_ws.fill(-std::numeric_limits<double>::infinity());

      for(unsigned int ii = dat.outer_start, o = 0L; ii < dat.outer_end;
          ++ii, ++o){
        auto i = ii + i_start;
        double &max_weight = max_ws[o];
        const double *y = Y.colptr(i);
        double *wi = weights_inner.colptr(o) + dat.inner_start;

        for(unsigned int j = dat.inner_start; j < dat.inner_end; ++j, ++wi){
          *wi = kernel(X.colptr(j), y, N, ws_log[j]);
          if(*wi > max_weight)
            max_weight = *wi;
        }

        if(is_final_inner_it)
          out[i] = log_sum_log(weights_inner.unsafe_col(o), max_ws[o]);
      }
    }
  }
};

// [[Rcpp::export]]
arma::vec naive(const arma::mat &X, const arma::vec ws, const arma::mat Y,
                unsigned int n_threads){
#ifdef MSSM_PROF
  profiler prof("naive");
#endif

#ifdef MSSM_DEBUG
  if(n_threads < 1L or n_threads > Y.n_cols)
    Rcpp::stop("invalid 'n_threads'");
#endif

  thread_pool pool(n_threads);
  mvs_norm kernel(X.n_rows);
  arma::vec ws_log = arma::log(ws), out(Y.n_cols);
  std::vector<std::future<void> > futures;
  arma::uword inc = Y.n_cols / n_threads + 1L, start = 0L, end = 0L;
  futures.reserve(Y.n_cols / inc + 1L);

  for(; start < Y.n_cols; start = end){
    end = std::min(end + inc, Y.n_cols);
    futures.push_back(pool.submit(
        naive_inner_loop(start, end, ws_log, X, Y, kernel, out)));
  }

  while(!futures.empty()){
    futures.back().get();
    futures.pop_back();
  }

  return out;
}

// [[Rcpp::export]]
arma::vec FSKA(
    const arma::mat &X, const arma::vec &ws, const arma::mat &Y,
    const arma::uword N_min, const double eps,
    const unsigned int n_threads){
  arma::mat X_cp = X, Y_cp = Y;
  arma::vec ws_cp = arma::log(ws);
  const mvs_norm kernel(X.n_rows);
  thread_pool pool(n_threads);
  arma::vec out(Y.n_cols, arma::fill::none);
  out.fill(-std::numeric_limits<double>::infinity());

  auto perm = FSKA_cpp(out, X_cp, Y_cp, ws_cp, N_min, eps, kernel, pool);
  return out(perm.Y_perm);
}

// [[Rcpp::export]]
arma::mat sample_mv_normal
  (const arma::uword N, const arma::mat &Q, const arma::vec &mu)
{
  arma::mat out(Q.n_cols, N);
  mv_norm mv(Q, mu);
  mv.sample(out);

  return out;
}

// [[Rcpp::export]]
arma::mat sample_mv_tdist
  (const arma::uword N, const arma::mat &Q, const arma::vec &mu,
   const double nu)
{
  arma::mat out(Q.n_cols, N);
  mv_tdist dt(Q, mu, nu);
  dt.sample(out);

  return out;
}

inline std::unique_ptr<problem_data> get_problem_data
  (const arma::vec &Y, const arma::vec &cfix, const arma::vec &ws,
   const arma::vec &offsets, const arma::vec &disp, const arma::mat &X,
   const arma::mat &Z, const arma::uvec &time_indices_elems,
   const arma::uvec &time_indices_len, const arma::mat &F, const arma::mat &Q,
   const arma::mat &Q0, const std::string &fam, const arma::vec &mu0,
   const arma::uword n_threads, const double nu, const double covar_fac,
   const double ftol_rel, const arma::uword N_part, const std::string &what,
   const unsigned int trace, const arma::uword KD_N_max, const double aprx_eps,
   const bool use_antithetic){
  /* create vector with time indices */
  const std::vector<arma::uvec> time_indices = ([&]{
    std::vector<arma::uvec> indices;
    indices.reserve(time_indices_len.n_elem);
    auto ele_begin = time_indices_elems.cbegin();
    arma::uword n_in = 0.;
    for(auto n_ele : time_indices_len){
      n_in += n_ele;
      if(n_in > time_indices_elems.n_elem)
        throw std::invalid_argument(
            "invalid 'time_indices_elems' and 'time_indices_len'");

      indices.emplace_back(ele_begin, n_ele);
      ele_begin += n_ele;
    }

    return indices;
  })();

  /* setup problem data object */
  control_obj ctrl(n_threads, nu, covar_fac, ftol_rel, N_part, what, trace,
                   KD_N_max, aprx_eps, use_antithetic);
  std::unique_ptr<problem_data> out(new problem_data(
      Y, cfix, ws, offsets, disp, X, Z, std::move(time_indices), F, Q, Q0,
      fam, mu0, std::move(ctrl)));

  return out;
}

// [[Rcpp::export]]
Rcpp::List pf_filter
  (const arma::vec &Y, const arma::vec &cfix, const arma::vec &ws,
   const arma::vec &offsets, const arma::vec &disp, const arma::mat &X,
   const arma::mat &Z, const arma::uvec &time_indices_elems,
   const arma::uvec &time_indices_len, const arma::mat &F, const arma::mat &Q,
   const arma::mat &Q0, const std::string &fam, const arma::vec &mu0,
   const arma::uword n_threads, const double nu, const double covar_fac,
   const double ftol_rel, const arma::uword N_part, const std::string &what,
   const std::string &which_sampler, const std::string &which_ll_cp,
   const unsigned int trace, const arma::uword KD_N_max, const double aprx_eps,
   const bool use_antithetic)
{
  std::unique_ptr<problem_data> dat = get_problem_data(
    Y, cfix, ws, offsets, disp, X, Z, time_indices_elems, time_indices_len,
    F, Q, Q0, fam, mu0, n_threads, nu, covar_fac, ftol_rel, N_part,
    what, trace, KD_N_max, aprx_eps, use_antithetic);

  /* setup sampler */
  const std::unique_ptr<sampler> sampler_ = ([&]{
    if(which_sampler == "bootstrap")
      return get_bootstrap_sampler();
    if(which_sampler == "mode_aprx")
      return get_mode_aprx_sampler();

    throw std::invalid_argument("Unkown sampler: '" + which_sampler + "'");
  })();

  /* setup object to compute log likehood and stats */
  const std::unique_ptr<stats_comp_helper> stats_cp = ([&]{
    if(which_ll_cp == "no_aprx")
      return std::unique_ptr<stats_comp_helper>(
        new stats_comp_helper_no_aprx());
    if(which_ll_cp == "KD")
      return std::unique_ptr<stats_comp_helper>(
        new stats_comp_helper_aprx_KD());

    throw std::invalid_argument("Unkown ll_cp: '" + which_ll_cp + "'");
  })();

  /* run particle filter */
  auto comp_res = PF(*dat, *sampler_, *stats_cp);

  /* make list and return */
  Rcpp::List out(comp_res.size());

  auto add_res = [](particle_cloud &cl){
    return Rcpp::List::create(
      Named("particles")     = std::move(cl.particles),
      Named("stats")         = std::move(cl.stats),
      Named("ws")            = std::move(cl.ws),
      Named("ws_normalized") = std::move(cl.ws_normalized)
    );
  };

  auto p_cloud = comp_res.begin();
  for(auto &ele : out)
    ele = add_res(*(p_cloud++));

  return out;
}

// [[Rcpp::export]]
Rcpp::List run_Laplace_aprx
  (const arma::vec &Y, const arma::vec &cfix, const arma::vec &ws,
   const arma::vec &offsets, const arma::vec &disp, const arma::mat &X,
   const arma::mat &Z, const arma::uvec &time_indices_elems,
   const arma::uvec &time_indices_len, const arma::mat &F, const arma::mat &Q,
   const arma::mat &Q0, const std::string &fam, const arma::vec &mu0,
   const arma::uword n_threads, const double nu, const double covar_fac,
   const double ftol_rel, const arma::uword N_part, const std::string &what,
   const unsigned int trace, const arma::uword KD_N_max, const double aprx_eps,
   const double ftol_abs, const double la_ftol_rel,
   const double ftol_abs_inner, const double la_ftol_rel_inner,
   const unsigned maxeval, const unsigned maxeval_inner){
  std::unique_ptr<problem_data> dat = get_problem_data(
    Y, cfix, ws, offsets, disp, X, Z, time_indices_elems, time_indices_len,
    F, Q, Q0, fam, mu0, n_threads, nu, covar_fac, ftol_rel, N_part, what,
    trace, KD_N_max, aprx_eps, false);

  auto result = Laplace_aprx(*dat, ftol_abs, la_ftol_rel, ftol_abs_inner,
                             la_ftol_rel_inner, maxeval, maxeval_inner);

  return Rcpp::List::create(
    Named("F.") = std::move(result.F),
    Named("Q") = std::move(result.Q),
    Named("cfix") = std::move(result.cfix),
    Named("logLik") = result.logLik,
    Named("n_it") = result.n_it,
    Named("code") = result.code,
    Named("disp") = result.disp);
}

// [[Rcpp::export]]
Rcpp::List smoother_cpp
  (const arma::vec &Y, const arma::vec &cfix, const arma::vec &ws,
   const arma::vec &offsets, const arma::vec &disp, const arma::mat &X,
   const arma::mat &Z, const arma::uvec &time_indices_elems,
   const arma::uvec &time_indices_len, const arma::mat &F, const arma::mat &Q,
   const arma::mat &Q0, const std::string &fam, const arma::vec &mu0,
   const arma::uword n_threads, const double nu, const double covar_fac,
   const double ftol_rel, const arma::uword N_part, const std::string &what,
   const unsigned int trace, const arma::uword KD_N_max, const double aprx_eps,
   const std::string &which_ll_cp, const Rcpp::List pf_output,
   const bool use_antithetic){
  /* setup problem data */
  std::unique_ptr<problem_data> dat = get_problem_data(
    Y, cfix, ws, offsets, disp, X, Z, time_indices_elems, time_indices_len,
    F, Q, Q0, fam, mu0, n_threads, nu, covar_fac, ftol_rel, N_part, what,
    trace, KD_N_max, aprx_eps, use_antithetic);

  /* make list of particles and weights */
  const unsigned n_periods = pf_output.size();
  std::vector<arma::mat> particles;
  particles.reserve(n_periods);
  std::vector<arma::vec> particle_weights;
  particle_weights.reserve(n_periods);
  for(auto &x : pf_output){
    Rcpp::List z = Rcpp::List(x);
    particles.push_back(Rcpp::as<arma::mat>(z["particles"]));
    particle_weights.push_back(Rcpp::as<arma::vec>(z["ws_normalized"]));
  }

  std::vector<const arma::mat *> particles_ptr;
  particles_ptr.reserve(n_periods);
  for(auto &x : particles)
    particles_ptr.push_back(&x);
  std::vector<const arma::vec *> particle_weights_ptr;
  particle_weights_ptr.reserve(n_periods);
  for(auto &x : particle_weights)
    particle_weights_ptr.push_back(&x);

  /* compute result and return */
  auto prep_res = [](const std::vector<arma::vec> &res){
    Rcpp::List out(res.size());
    for(unsigned j = 0; j < res.size(); ++j)
      out[j] = std::move(res.at(j));

    return out;
  };

  if(which_ll_cp == "no_aprx")
    return prep_res(smoother     (*dat, particles_ptr, particle_weights_ptr));
  else if(which_ll_cp == "KD")
    return prep_res(smoother_aprx(*dat, particles_ptr, particle_weights_ptr));

  throw std::invalid_argument(
      "'which_ll_cp' '" + which_ll_cp + "' not implemented");
}

/* exported to test the samples */
// [[Rcpp::export]]
arma::mat t_dist_antithe_test
  (const unsigned n_sims, const arma::mat &Q, const arma::vec &mu,
   const double nu){
  mv_tdist tdist(Q, mu, nu);

  arma::mat out(Q.n_cols, n_sims);
  tdist.sample_anti(out);

  return out;
}
