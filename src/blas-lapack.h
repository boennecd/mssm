#ifndef BLAS_LAPACK_H
#define BLAS_LAPACK_H
#include <R_ext/RS.h>	/* for F77_... */

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
  void F77_NAME(dger)(
      const int *m, const int *n, const double *alpha,
      const double *x, const int *incx,
      const double *y, const int *incy,
      double *a, const int *lda);
  void F77_NAME(daxpy)(
      const int *n, const double *alpha,
      const double *dx, const int *incx,
      double *dy, const int *incy);
  void F77_NAME(dgetrs)(
      const char *trans, const int *n, const int *nrhs,
      const double *a, const int *lda, const int *ipiv,
      double *b, const int *ldb, int *info);
  void F77_NAME(dtrmm)(
      const char *side, const char *uplo, const char *transa,
      const char *diag, const int *m, const int *n,
      const double *alpha, const double *a, const int *lda,
      double *b, const int *ldb);
  void F77_NAME(dger)(
        const int *m, const int *n, const double *alpha,
        const double *x, const int *incx,
        const double *y, const int *incy,
        double *a, const int *lda);
  void F77_NAME(dsyr2)(
      const char *uplo, const int *n, const double *alpha,
      const double *x, const int *incx,
      const double *y, const int *incy,
      double *a, const int *lda);
  void F77_NAME(dsbmv)(
      const char *uplo, const int *n, const int *k,
      const double *alpha, const double *a, const int *lda,
      const double *x, const int *incx,
      const double *beta, double *y, const int *incy);
  void F77_NAME(dpbtrs)(
      const char* uplo, const int* n,
      const int* kd, const int* nrhs,
      const double* ab, const int* ldab,
      double* b, const int* ldb, int* info);
}

/* avoid warnings due to different definitions of function */
namespace lapack {
  void dgetrf
    (const int*, const int*, double*, const int*, int*, int*);
  void dpbtrf
    (const char*, const int*, const int*, double*, const int*, int*);
  void dpotri(char*, int*, double*, int*, int*);
}

#endif
