#ifndef DUP_MULT_H
#define DUP_MULT_H
#include "arma.h"

/* computes either the transpose of the duplication matrix times the second
 * argument and adds it to the first argument or the second argument times the
 * duplication matrix and adds it to the first argument. The fourth argument
 * is the original dimension (before using the vec operator)*/
void D_mult(arma::mat&, const arma::mat&, const bool is_left,
            const unsigned int);

#endif
