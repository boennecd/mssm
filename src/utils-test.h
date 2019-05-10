#ifndef UTILS_TEST_H
#define UTILS_TEST_H
#include "arma.h"
#include <testthat.h>

template
  <arma::uword n_elem,
   template <arma::uword> class arma_T = arma::vec::fixed, std::size_t N,
   typename T>
  arma_T<n_elem> create_vec(const T(&data)[N])
{
  static_assert(N == n_elem, "invalid 'data'");

  typedef typename arma_T<n_elem>::value_type value_T;
  value_T data_pass[N];
  for(unsigned i = 0; i < N; ++i)
    data_pass[i] = data[i];

  return arma_T<n_elem>(data_pass);
}

template
  <arma::uword n_rows, arma::uword n_cols,
   template <arma::uword, arma::uword> class arma_T = arma::mat::fixed,
   std::size_t N, typename T>
  arma_T<n_rows, n_cols> create_mat(const T(&data)[N])
{
  static_assert(N == n_rows * n_cols, "invalid 'data'");

  typedef typename arma_T<n_rows, n_cols>::value_type value_T;
  value_T data_pass[N];
  for(unsigned i = 0; i < N; ++i)
    data_pass[i] = data[i];

  return arma_T<n_rows, n_cols>(data_pass);
}

template<class T1, class T2>
bool is_all_equal(T1 first1, T1 end1, T2 first2, T2 end2){
  auto d1 = std::distance(first1, end1);
  if(d1 != std::distance(first2, end2))
    throw std::invalid_argument("The length of iterators do not match");
  std::vector<std::size_t> idx(d1);
  std::iota(idx.begin(), idx.end(), 0L);
  return std::all_of(
    idx.begin(), idx.end(),
    [&](std::size_t i){
      return *(first1 + i) == *(first2 + i);
    });
}

template<class XT, class YT>
bool is_all_equal(XT &X, YT &Y){
  return is_all_equal(
    std::begin(X), std::end(X), std::begin(Y), std::end(Y));
}

template<class T1, class T2>
bool is_all_aprx_equal
  (const T1 first1, const T1 end1, const T2 first2, const T2 end2,
   const double eps = 1e-12){
  auto d1 = std::distance(first1, end1);
  if(d1 != std::distance(first2, end2))
    throw std::invalid_argument("The length of iterators do not match");
  std::vector<std::size_t> idx(d1);
  std::iota(idx.begin(), idx.end(), 0L);
  return std::all_of(
    idx.begin(), idx.end(),
    [&](std::size_t i){
      const double d1 = *(first1 + i), d2 = *(first2 + i);
      const double diff = std::abs(d1 - d2), d1_abs = std::abs(d1);
      return (d1_abs < eps) ? diff < eps : diff / d1_abs < eps;
    });
}

template<class XT, class YT>
bool is_all_aprx_equal(XT &X, YT &Y, const double eps = 1e-12){
  return is_all_aprx_equal(
    std::begin(X), std::end(X), std::begin(Y), std::end(Y), eps);
}

#endif
