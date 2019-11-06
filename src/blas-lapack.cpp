#include "blas-lapack.h"

#include <Rconfig.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#ifndef FCLEN
#define FCLEN
#endif
#ifndef FCONE
#define FCONE
#endif

void dtrsm(
    const char *side, const char *uplo,
    const char *transa, const char *diag,
    const int *m, const int *n, const double *alpha,
    const double *a, const int *lda,
    double *b, const int *ldb){
  F77_CALL(dtrsm)(
      side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb
      FCONE FCONE FCONE FCONE);
}
void dsyr(
    const char *uplo, const int *n, const double *alpha,
    const double *x, const int *incx,
    double *a, const int *lda){
  F77_CALL(dsyr)(
      uplo, n, alpha, x, incx, a, lda
      FCONE);
}
void dpotrs(
    const char* uplo, const int* n,
    const int* nrhs, const double* a, const int* lda,
    double* b, const int* ldb, int* info){
  F77_CALL(dpotrs)(
      uplo, n, nrhs, a, lda, b, ldb, info
      FCONE);
}
void dger(
    const int *m, const int *n, const double *alpha,
    const double *x, const int *incx,
    const double *y, const int *incy,
    double *a, const int *lda){
  F77_CALL(dger)(
      m, n, alpha, x, incx, y, incy, a, lda);
}
void daxpy(
    const int *n, const double *da,
    const double *dx, const int *incx,
    double *dy, const int *incy){
  F77_CALL(daxpy)(
      n, da, dx, incx, dy, incy);
}
void dgetrs(
    const char* trans, const int* n, const int* nrhs,
    const double* a, const int* lda, const int* ipiv,
    double* b, const int* ldb, int* info){
  F77_CALL(dgetrs)(
      trans, n, nrhs, a, lda, ipiv, b, ldb, info
      FCONE);
}
void dtrmm(
    const char *side, const char *uplo, const char *transa,
    const char *diag, const int *m, const int *n,
    const double *alpha, const double *a, const int *lda,
    double *b, const int *ldb){
  F77_CALL(dtrmm)(
      side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb
      FCONE FCONE FCONE FCONE);
}
void dsyr2(
    const char *uplo, const int *n, const double *alpha,
    const double *x, const int *incx,
    const double *y, const int *incy,
    double *a, const int *lda){
  F77_CALL(dsyr2)(
      uplo, n, alpha, x, incx, y, incy, a, lda
      FCONE);
}
void dsbmv(
    const char *uplo, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda,
    const double *x, const int *incx,
    const double *beta, double *y, const int *incy){
  F77_CALL(dsbmv)(
      uplo, n, k, alpha, a, lda, x, incx, beta, y, incy
      FCONE);
}
void dpbtrs(
    const char* uplo, const int* n,
    const int* kd, const int* nrhs,
    const double* ab, const int* ldab,
    double* b, const int* ldb, int* info){
  F77_CALL(dpbtrs)(
      uplo, n, kd, nrhs, ab, ldab, b, ldb, info
      FCONE);
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
    F77_CALL(::dpbtrf)(uplo, n, kd, ab, ldab, info
                       FCONE);
  }

  void dpotri
    (char* uplo, int *n, double *a, int *lda, int *info){
    F77_CALL(::dpotri)(uplo, n, a, lda, info
                       FCONE);
  }
}

