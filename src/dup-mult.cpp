#include <memory>
#include "dup-mult.h"
#include "misc.h"

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

void D_mult_left
  (const unsigned int n, const unsigned int m, const double alpha,
   double * const o, const unsigned int ldo, const double * const x)
{
  /* store local copy to avoid computations */
  M_THREAD_LOCAL std::unique_ptr<dup_mult_indices> indices;

  if(!indices or indices->n != n)
    indices.reset(new dup_mult_indices(n));

  const unsigned int nn = indices->nn,
    * const indices_start = indices->indices.get();

#ifdef MSSM_DEBUG
  if(ldo < indices->nq)
    throw std::invalid_argument("'ldo' less than 'nq'");
#endif

  if(alpha == 1.){
    /* loop over rows of x */
    for(unsigned int i = 0; i < m; ++i){
      const double *       xi = x + i * nn;
            double * const oi = o + i * ldo;
      const unsigned int *idx = indices_start;
      for(unsigned int j =0; j < nn; ++j, ++xi, ++idx)
        *(oi + *idx) += *xi;
    }

    return;
  }

  /* loop over rows of x */
  for(unsigned int i = 0; i < m; ++i){
    const double *       xi = x + i * nn;
          double * const oi = o + i * ldo;
    const unsigned int *idx = indices_start;
    for(unsigned int j =0; j < nn; ++j, ++xi, ++idx)
      *(oi + *idx) += alpha * *xi;
  }
}

void D_mult_right
  (const unsigned int n, const unsigned int m, const double alpha,
   double * const o, const unsigned int ldo, const double * const x)
{
  /* store local copy to avoid computations */
  M_THREAD_LOCAL std::unique_ptr<dup_mult_indices> indices;
  if(!indices or indices->n != n)
    indices.reset(new dup_mult_indices(n));

  const unsigned int nn = indices->nn,
    * const indices_start = indices->indices.get();

#ifdef MSSM_DEBUG
  if(ldo < m)
    throw std::invalid_argument("'ldo' less than 'm'");
#endif

  if(alpha == 1.){
    const unsigned int *idx = indices_start;
    for(unsigned int j = 0; j < nn; ++j, ++idx){
      const double * xi = x + j    * m;
            double * oi = o + *idx * ldo;
      for(unsigned int i = 0; i < m; ++i, ++xi, ++oi)
        *oi += *xi;
    }

    return;
  }

  const unsigned int *idx = indices_start;
  for(unsigned int j = 0; j < nn; ++j, ++idx){
    const double * xi = x + j    * m;
          double * oi = o + *idx * ldo;
    for(unsigned int i = 0; i < m; ++i, ++xi, ++oi)
      *oi += alpha * *xi;
  }
}


void D_mult(arma::mat &out, const arma::mat &X, const bool is_left,
            const unsigned int n)
{
  if(is_left){
#ifdef MSSM_DEBUG
    if(X.n_cols != out.n_cols)
      throw std::invalid_argument("invalid 'X'");
#endif

    D_mult_left(n, X.n_cols, 1., out.memptr(), out.n_rows, X.memptr());

    return;
  }

#ifdef MSSM_DEBUG
  if(X.n_rows != out.n_rows)
    throw std::invalid_argument("invalid 'X'");
#endif

  D_mult_right(n, X.n_rows, 1., out.memptr(), out.n_rows, X.memptr());
}
