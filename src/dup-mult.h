#ifndef DUP_MULT_H
#define DUP_MULT_H
#include "arma.h"

/* computes either the transpose of the duplication matrix times the second
 * argument and adds it to the first argument or the second argument times the
 * duplication matrix and adds it to the first argument. The fourth argument
 * is the original dimension (before using the vec operator)*/
void D_mult(arma::mat&, const arma::mat&, const bool is_left,
            const unsigned int);

/* same as above but only for left multiplication. Allows for multiplication
 * by a scalar and different leading dimension of the output matrix */
void D_mult_left
  (const unsigned int, const unsigned int, const double,
   double * const, const unsigned int, const double * const);

/* similar to the above but for multiplcation on the right side */
void D_mult_right
  (const unsigned int, const unsigned int, const double,
   double * const, const unsigned int, const double * const);

#endif
