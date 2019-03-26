#include "fast-kernal-approx.hpp"
#include "kernals.hpp"

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

// [[Rcpp::export]]
arma::vec naive(const arma::mat &X, const arma::vec ws, const arma::mat Y){
  mvariate kernal(X.n_rows);
  arma::vec ws_log = arma::log(ws);
  arma::vec weights_inner(X.n_cols), out(Y.n_cols);
  for(unsigned int i = 0; i < Y.n_cols; ++i){
    const arma::vec &y = Y.unsafe_col(i);
    double max_weight = std::numeric_limits<double>::lowest();
    for(unsigned int j = 0; j < X.n_cols; ++j){
      double dist = norm_square(X.unsafe_col(j), y);
      weights_inner[j] = ws_log[j] + kernal(dist, true);
      if(weights_inner[j] > max_weight)
        max_weight = weights_inner[j];
    }

    out[i] = log_sum_log(weights_inner, max_weight);
  }

  return out;
}
