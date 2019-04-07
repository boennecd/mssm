#include "utils.h"
#include <R_ext/RS.h>	/* for F77_... */

using std::invalid_argument;

extern "C" {
  void F77_NAME(dtrsm)(
      const char *side, const char *uplo,
      const char *transa, const char *diag,
      const int *m, const int *n, const double *alpha,
      const double *a, const int *lda,
      double *b, const int *ldb);
  void F77_NAME(dsyr)(
      const char *uplo, const int *n, const double *alpha,
      const double *x, const int *incx,
      double *a, const int *lda);
  void F77_NAME(dpotrs)(
      const char* uplo, const int* n,
      const int* nrhs, const double* a, const int* lda,
      double* b, const int* ldb, int* info);
  void F77_NAME(dpotri)(
      char* uplo, int* n, double* a, int* lda, int* info);
}

chol_decomp::chol_decomp(const arma::mat &X):
  X(X), chol_(arma::chol(X)) { }

static constexpr char C_U = 'U', C_N = 'N', C_L = 'L', C_T = 'T';
static constexpr double D_one = 1.;
static constexpr int I_one = 1L;

inline void solve_half_(const arma::mat &chol_, arma::mat &X){
#ifdef MSSM_DEBUG
  if(X.n_rows != chol_.n_cols)
    throw invalid_argument("dims do not match with 'chol_'");
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

void chol_decomp::solve(arma::mat &out) const
{
#ifdef MSSM_DEBUG
  if(out.n_rows != chol_.n_cols)
    throw invalid_argument("dims do not match with 'chol_'");
#endif

  int n = chol_.n_cols, nrhs = out.n_cols, info;
  F77_CALL(dpotrs)(&C_U, &n, &nrhs, chol_.memptr(), &n, out.memptr(),
                   &n, &info);
  if(info != 0)
    throw std::runtime_error("'dpotrs' failed with info " +
                             std::to_string(info));
}

arma::mat chol_decomp::solve(const arma::mat &X) const
{
  arma::mat out = X;
  solve(out);
  return out;
}
arma::vec chol_decomp::solve(const arma::vec &x) const
{
  arma::vec out = x;
  arma::mat dum(out.memptr(), out.n_elem, 1L, false);
  solve(dum);
  return out;
}

const arma::mat& chol_decomp::get_inv() const
{
  /* set inverse */
  std::call_once(*is_inv_set, [&](){
    arma::mat &inv_mat = *inv_;
    inv_mat = chol_;
    int n = chol_.n_cols, info;
    char upper = 'U';
    F77_CALL(dpotri)(&upper, &n, inv_mat.memptr(), &n, &info);
    if(info != 0)
      throw std::runtime_error("'dpotri' failed with info " +
                               std::to_string(info));

    inv_mat = arma::symmatu(inv_mat);
  });

  return *inv_;
}

void arma_dsyr(arma::mat &A, const arma::vec &x, const double alpha)
{
  int n = A.n_cols;
#ifdef MSSM_DEBUG
  if(A.n_cols != A.n_rows)
    throw invalid_argument("invalid 'A'");
  if(x.n_elem != A.n_cols)
    throw invalid_argument("invalid 'x'");
#endif

  F77_CALL(dsyr)(
    &C_U, &n, &alpha, x.memptr(), &I_one, A.memptr(), &n);
}
