#include "utils.h"
#include <R_ext/RS.h>	/* for F77_... */

extern "C" {
  void F77_NAME(dtrsm)(
      const char *side, const char *uplo,
      const char *transa, const char *diag,
      const int *m, const int *n, const double *alpha,
      const double *a, const int *lda,
      double *b, const int *ldb);
}

chol_decomp::chol_decomp(const arma::mat &X):
  chol_(arma::chol(X)) { }

static constexpr char C_U = 'U', C_N = 'N', C_L = 'L', C_T = 'T';
static constexpr double D_one = 1.;

inline void solve_half_(const arma::mat &chol_, arma::mat &X){
#ifdef MSSM_DEBUG
  if(X.n_rows != chol_.n_cols)
    throw std::invalid_argument("dims do not match with 'chol_'");
#endif
  int n = X.n_cols, m = X.n_rows;
  F77_CALL(dtrsm)(
      &C_L, &C_U, &C_T, &C_N, &m, &n, &D_one, chol_.begin(), &m, X.begin(),
      &m);
}

void chol_decomp::solve_half(arma::mat &X) const {
  solve_half_(chol_, X);
}

void chol_decomp::solve_half(arma::vec &x) const {
  arma::mat dum(x.begin(), x.n_elem, 1L, false);
  solve_half_(chol_, dum);
}

arma::mat chol_decomp::solve_half(const arma::mat &X) const
{
  arma::mat out = X;
  solve_half(out);
  return out;
}

arma::vec chol_decomp::solve_half(const arma::vec &x) const
{
  arma::vec out = x;
  solve_half(out);
  return out;
}
