#include "blas-lapack.h"

extern "C" {
  void F77_NAME(dgetrf)
    (const int* m, const int* n, double* a, const int* lda,
     int* ipiv, int* info);
  void F77_NAME(dpbtrf)
    (const char* uplo, const int* n, const int* kd,
     double* ab, const int* ldab, int* info,
     size_t);
  void F77_NAME(dpotri)
    (const char* uplo, const int* n,
     double* a, const int* lda, int* info,
     size_t);
}

namespace lapack {
  void dgetrf
    (const int *m, const int *n, double *a, const int *lda,
     int *ipiv, int *info){
    F77_CALL(::dgetrf)(m, n, a, lda, ipiv, info);
  }

  void dpbtrf
  (const char* uplo, const int* n, const int* kd,
   double* ab, const int* ldab, int* info){
    F77_CALL(::dpbtrf)(uplo, n, kd, ab, ldab, info, 1);
  }

  void dpotri
    (char* uplo, int *n, double *a, int *lda, int *info){
    F77_CALL(::dpotri)(uplo, n, a, lda, info, 1);
  }
}

