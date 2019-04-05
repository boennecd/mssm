#include "kernels.h"
#include "utils.h"

mvariate::mvariate(const arma::uword dim):
  norm_const_log(-(double)dim / 2. * std::log(2. * M_PI)),
  norm_const(std::exp(norm_const_log)) { }

double mvariate::operator()(const double dist) const
{
  return norm_const_log - dist / 2.;
}

double mvariate::operator()(
    const double *x, const double *y, const arma::uword N,
    const double x_log_w) const
{
  const double dist = norm_square(x, y, N);
  return this->operator()(dist) + x_log_w;
}
