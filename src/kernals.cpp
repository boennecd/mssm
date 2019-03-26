#include "kernals.hpp"

mvariate::mvariate(const arma::uword &dim):
  norm_const_log(-(double)dim / 2. * std::log(2. * M_PI)),
  norm_const(std::exp(norm_const_log)) { }

double mvariate::operator()(const double dist, const bool is_log) const
  {
    if(is_log)
      return norm_const_log - dist / 2.;
    return norm_const * std::exp(-dist / 2.);
  }
