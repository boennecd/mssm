#include <memory>
#include "dup-mult.h"
/* class to store non-zero n x n |-> n(n+1)/2 indices of duplcation matrix.
 * See https://gist.github.com/boennecd/09ab5b0baae4738089530ae37bc9812e */
class dup_mult_indices {
public:
  const unsigned int n, nn = n * n, nq = (n * (n + 1L)) / 2L;
  const std::unique_ptr<unsigned int[]> indices =
    std::unique_ptr<unsigned int[]>(new unsigned int[nn]);
  dup_mult_indices(const unsigned int n): n(n)
  {
    int j = -1L;
    unsigned int next_diag = 0L, next_inc = n, i = 0L, jn = 0L,
      *o = indices.get();
    for(unsigned int k = 0; k < nq; ++k){
      if(k == next_diag){
        /* diagonal element */
        i = ++j;
        jn = j * n;
        next_diag += next_inc--;
        *(o + i + jn) = k;
        continue;
      }
      ++i;
      *(o + i + jn) = *(o + i * n + j) = k;
    }
  }
};

void D_mult(arma::mat &out, const arma::mat &X, const bool is_left,
            const unsigned int n)
{
  /* store local copy to avoid computations */
  thread_local static std::unique_ptr<dup_mult_indices> indices;
  if(!indices or indices->n != n)
    indices.reset(new dup_mult_indices(n));

  if(is_left){
#ifdef MSSM_DEBUG
    if(out.n_rows != indices->nq)
      throw std::invalid_argument("invalid 'out' in 'D_mult'");
    if(X.n_rows != indices->nn or X.n_cols != out.n_cols)
      throw std::invalid_argument("invalid 'X'");
#endif
    const unsigned int nn = indices->nn;
    for(unsigned int i = 0; i < X.n_cols; ++i){
      const double *x = X.colptr(i);
      double * const o = out.colptr(i);
      unsigned int *idx = indices->indices.get();
      for(unsigned int j = 0; j < nn; ++j, ++x, ++idx)
        *(o + *idx) += *x;
    }

    return;
  }

#ifdef MSSM_DEBUG
  if(out.n_cols != indices->nq)
    throw std::invalid_argument("invalid 'out' in 'D_mult'");
  if(X.n_cols != indices->nn or X.n_rows != out.n_rows)
    throw std::invalid_argument("invalid 'X'");
#endif

  const unsigned int nn = indices->nn, nx_row = X.n_rows;
  for(unsigned int i = 0; i < X.n_rows; ++i){
    const double *x = X.begin() + i;
    double * const o = out.begin() + i;
    unsigned int *idx = indices->indices.get();
    for(unsigned int j = 0; j < nn; ++j, x += nx_row, ++idx)
      *(o + *idx * nx_row) += *x;
  }
}
