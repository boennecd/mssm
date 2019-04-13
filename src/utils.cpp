#include "utils.h"
#include "blas-lapack.h"

using std::invalid_argument;

inline arma::mat set_chol_(const arma::mat &X)
{
  return arma::trimatu(arma::chol(X));
}

chol_decomp::chol_decomp(const arma::mat &X):
  X(X), chol_(set_chol_(X)) { }

static constexpr char C_U = 'U', C_N = 'N', C_L = 'L';
static constexpr double D_one = 1.;
static constexpr int I_one = 1L;

inline void solve_half_(const arma::mat &chol_, arma::mat &X,
                        const bool transpose){
#ifdef MSSM_DEBUG
  if(X.n_rows != chol_.n_cols)
    throw invalid_argument("dims do not match with 'chol_'");
#endif
  int n = X.n_cols, m = X.n_rows;
  char trans = transpose ? 'T' : 'N';
  F77_CALL(dtrsm)(
      &C_L, &C_U, &trans, &C_N, &m, &n, &D_one, chol_.begin(), &m, X.begin(),
      &m);
}

void chol_decomp::solve_half(arma::mat &X, const bool transpose) const {
  solve_half_(chol_, X, !transpose);
}

void chol_decomp::solve_half(arma::vec &x, const bool transpose) const {
  arma::mat dum(x.begin(), x.n_elem, 1L, false);
  solve_half_(chol_, dum, !transpose);
}

arma::mat chol_decomp::solve_half(const arma::mat &X, const bool transpose) const
{
  arma::mat out = X;
  solve_half(out, transpose);
  return out;
}

arma::vec chol_decomp::solve_half(const arma::vec &x, const bool transpose) const
{
  arma::vec out = x;
  solve_half(out, transpose);
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
