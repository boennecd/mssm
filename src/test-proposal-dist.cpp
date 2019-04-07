#include "proposal_dist.h"
#include "utils-test.h"
#include <testthat.h>
#include <array>

/* R code used below
		func <- function(y, X, cfix, Z, w, Q, mu, family){
		  library(mvtnorm)
		  off <- X %*% cfix
		  func <- function(crng){
		    mea <- drop(family$linkinv(off + Z %*% crng))
		    ll <- switch(family$family,
				 "binomial" = sum(w * dbinom(y, 1L, mea, log = TRUE)),
				 stop("not implemented"))
		    -(ll + dmvnorm(crng, mu, Q, log = TRUE))
		  }

		  opt_out <- optim(numeric(length(mu)), func, control = list(reltol = 1e-16))

		  library(numDeriv)
		  neg_H <- hessian(func, opt_out$par)

		  . <- function(x)
		    cat("\n{", paste0(x, collapse = ", "), "}\n")
		  .(opt_out$value)
		  .(opt_out$par)
		  .(signif(solve(neg_H), 16))
		}
*/

/* The first arguments are input while the latter arguments is the inverse
 * negative Hessian and mode */
template<class dist_T>
void test_func
  (arma::vec Y, arma::mat X, arma::vec cfix, arma::mat Z, arma::vec w,
   arma::vec disp, arma::mat Q, arma::vec mu, arma::vec mode,
   arma::mat Neg_Inv_Hes)
{
  dist_T family(Y, X, cfix, Z, disp, &w);
  mv_norm prior(Q, mu);

  /* assume zero vector is an ok starting point */
  arma::vec start(mu.n_elem, arma::fill::zeros);

  double nu = -1., covar_fac = 1., ftol_rel = 1e-16;
  {
    auto out = mode_approximation(
      { &prior, &family }, start, nu, covar_fac, ftol_rel);
    expect_true(!out.any_errors);

    /* should be a multivariate normal distribution */
    mv_norm *ptr = dynamic_cast<mv_norm*>(out.proposal.get());
    expect_true(ptr);
    if(ptr){
      expect_true(is_all_aprx_equal(ptr->mean(), mode,        1e-5));
      arma::mat vcov = ptr->vCov();
      expect_true(is_all_aprx_equal(vcov       , Neg_Inv_Hes, 1e-5));
    }
  }

  nu = 4.;
  {
    auto out = mode_approximation(
    { &prior, &family }, start, nu, covar_fac, ftol_rel);
    expect_true(!out.any_errors);

    /* should be a multivariate t-distribution */
    mv_tdist *ptr = dynamic_cast<mv_tdist*>(out.proposal.get());
    expect_true(ptr);
    if(ptr){
      expect_true(is_all_aprx_equal(ptr->mean(), mode,        1e-5));
      arma::mat vcov = ptr->vCov();
      expect_true(is_all_aprx_equal(vcov       , Neg_Inv_Hes, 1e-5));
    }
  }

  nu = -1;
  covar_fac = 1.2;
  {
    auto out = mode_approximation(
    { &prior, &family }, start, nu, covar_fac, ftol_rel);
    expect_true(!out.any_errors);

    /* should be a multivariate normal distribution */
    mv_norm *ptr = dynamic_cast<mv_norm*>(out.proposal.get());
    expect_true(ptr);
    if(ptr){
      expect_true(is_all_aprx_equal(ptr->mean(), mode,         1e-5));
      arma::mat vcov = ptr->vCov(), other_scaled = covar_fac * Neg_Inv_Hes;
      expect_true(is_all_aprx_equal(vcov       , other_scaled, 1e-5));
    }
  }
}

context("Test mvariate") {
  test_that("Test mode_approximation with binomial_logit") {
    /*  R code
     y <- c(1, 1, 1, 0, 0)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = binomial("logit"))
     */
    test_func<binomial_logit>(
      create_vec<5L>({1, 1, 1, 0, 0}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<0L>({ }),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({0.142785768678946, -0.264814543365718}),
      create_mat<2L, 2L>({1.72420197619348, -0.816978073329191, -0.816978073329191, 1.00305631517286})
    );
  }
}
