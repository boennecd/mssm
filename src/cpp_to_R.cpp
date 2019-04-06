#include "fast-kernel-approx.h"
#include "utils.h"
#include "dists.h"

#ifdef MSSM_PROF
#include "profile.h"
#endif

// [[Rcpp::export]]
Rcpp::List test_KD_note(const arma::mat &X, const arma::uword N_min){
  KD_note root = get_KD_tree(X, N_min);

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

struct naive_inner_loop {
  const arma::uword i_start, i_end;
  const arma::vec &ws_log;
  const arma::mat &X, &Y;
  const trans_obj &kernel;
  arma::vec &out;

  arma::vec weights_inner;
  const arma::uword N;

  naive_inner_loop
    (const arma::uword i_start, const arma::uword i_end,
     const arma::vec &ws_log, const arma::mat &X, const arma::mat &Y,
     const trans_obj &kernel, arma::vec &out):
    i_start(i_start), i_end(i_end), ws_log(ws_log), X(X), Y(Y),
    kernel(kernel), out(out), weights_inner(X.n_cols), N(Y.n_rows) { }

  void operator()(){
    for(unsigned int i = i_start; i < i_end; ++i){
      const double *y = Y.colptr(i);
      double max_weight = std::numeric_limits<double>::lowest();
      for(unsigned int j = 0; j < X.n_cols; ++j){
        weights_inner[j] = kernel(X.colptr(j), y, N, ws_log[j]);
        if(weights_inner[j] > max_weight)
          max_weight = weights_inner[j];
      }

      out[i] = log_sum_log(weights_inner, max_weight);
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
  out.fill(std::numeric_limits<double>::quiet_NaN());

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
