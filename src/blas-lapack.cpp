#include "blas-lapack.h"

extern "C" {
  void F77_NAME(dgetrf)(
      const int *m, const int *n, double *a, const int *lda,
      int *ipiv, int *info);
}

namespace lapack {
  void dgetrf(
      const int *m, const int *n, double *a, const int *lda,
      int *ipiv, int *info){
    F77_CALL(::dgetrf)(m, n, a, lda, ipiv, info);
  }
}

