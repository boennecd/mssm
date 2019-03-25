#include "kd-tree.h"

// [[Rcpp::export]]
Rcpp::List test_KD_note(const arma::mat X, const arma::uword N_min){
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
