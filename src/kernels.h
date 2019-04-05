#ifndef KERNELS_H
#define KERNELS_H
#include "arma.h"

class trans_obj {
public:
  virtual double operator()(const double) const = 0;
  virtual double operator()(
      const double *, const double *, const arma::uword,
      const double) const = 0;
};

class mvariate : public trans_obj {
  const double norm_const_log;
  const double norm_const;
public:
  mvariate(const arma::uword);

  double operator()(const double) const final;
  double operator()(
      const double *, const double *, const arma::uword,
      const double) const final;
};

#endif
