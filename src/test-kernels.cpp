#include "kernels.h"
#include <testthat.h>
#include <array>

context("Test mvariate") {
  test_that("Test mvariate gives correct results in 3D") {
    std::array<double, 3> x = {-3, 2, 1};
    std::array<double, 3> y = {-1, 0, 2};

    std::unique_ptr<trans_obj> kernel(new mvariate(3L));

    /* print(sum(dnorm(c(2, 2, 1), log = TRUE)), digits = 16) */
    constexpr double expect = -7.256815599614018;
    expect_true(std::abs(kernel->operator()(
        x.data(), y.data(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(kernel->operator()(
        x.data(), y.data(), 3L, 1) - (expect + 1.)) < 1e-8);
    expect_true(std::abs(kernel->operator()(
     2. * 2. + 2. * 2. + 1.) - expect) < 1e-8);
  }
}
