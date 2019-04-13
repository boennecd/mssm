#include "proposal_dist.h"
#include "utils-test.h"
#include <testthat.h>
#include <array>

/* R code used below
		func <- function(y, X, cfix, Z, w, Q, mu, family, disp = numeric()){
		  library(mvtnorm)

      ll <- function(mea)
        switch(family$family,
          "binomial" = sum(w * dbinom(y, 1L, mea, log = TRUE)),
          "poisson"  = sum(w * dpois(y, mea, log = TRUE)),
          "Gamma"    = sum(w * dgamma(y, 1/disp, scale = mea * disp, log = TRUE)),
          "gaussian" = sum(w * dnorm(y, mea, sd = sqrt(disp), log = TRUE)),
          stop("not implemented"))

		  off <- X %*% cfix
		  func <- function(crng){
        eta <-  drop(off + Z %*% crng)
		    mea <- family$linkinv(eta)
		    -(ll(mea) + dmvnorm(crng, mu, Q, log = TRUE))
		  }

		  opt_out <- optim(numeric(length(mu)), func, control = list(reltol = 1e-16))

		  library(numDeriv)
		  neg_H <- hessian(func, opt_out$par)

		  . <- function(x)
		    cat("\n{", paste0(x, collapse = ", "), "}\n")
		  .(opt_out$value)
		  .(opt_out$par)
		  .(signif(solve(neg_H), 16))

      # we also test the Hessian and Gradient while w.r.t. the fixed coefficients
      # while we are it
       func2 <- function(cfix){
         eta <-  drop(X %*% cfix + Z %*% mu)
         mea <- family$linkinv(eta)
         ll(mea)
       }

       .(jacobian(func2, cfix))
       .(hessian(func2, cfix))
		}
*/

/* The first arguments are input while the latter arguments is the inverse
 * negative Hessian and mode */
template<class dist_T>
void test_func
  (arma::vec Y, arma::mat X, arma::vec cfix, arma::mat Z, arma::vec w,
   arma::vec disp, arma::mat Q, arma::vec mu, arma::vec mode,
   arma::mat Neg_Inv_Hes, arma::vec d_beta, arma::mat dd_beta)
{
  dist_T family(Y, X, cfix, Z, &w, disp);
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

  expect_true(family.state_stat_dim(log_densty) == 0L);
  expect_true(family.state_stat_dim(gradient)   == 0L);
  expect_true(family.state_stat_dim(Hessian)    == 0L);

  const arma::uword dim = cfix.n_elem;
  expect_true(family.obs_stat_dim(log_densty) == 0L);
  expect_true(family.obs_stat_dim(gradient)   == dim);
  expect_true(family.obs_stat_dim(Hessian)    == dim * (1L + dim));
  {
    /* gradient only */
    arma::vec gr(dim, arma::fill::zeros);
    family.comp_stats_state_only(mu, gr.memptr(), gradient);

    expect_true(is_all_aprx_equal(gr, d_beta, 1e-5));

    /* add something to start with */
    gr.fill(1.);
    family.comp_stats_state_only(mu, gr.memptr(), gradient);

    arma::vec d_beta_p1 = d_beta + 1;
    expect_true(is_all_aprx_equal(gr, d_beta_p1, 1e-5));
  }

  {
    /* gradient and Hessian */
    std::vector<double> mem(dim * (dim + 1L));
    arma::vec gr(mem.data(), dim, false);
    arma::mat H (mem.data() + dim, dim, dim, false);
    family.comp_stats_state_only(mu, mem.data(), Hessian);

    expect_true(is_all_aprx_equal(gr, d_beta, 1e-5));
    expect_true(is_all_aprx_equal(H, dd_beta, 1e-5));

    /* add something to start with */
    gr.fill(1.);
    H.fill(1.);
    family.comp_stats_state_only(mu, mem.data(), Hessian);

    arma::vec d_beta_p1 = d_beta + 1;
    arma::mat dd_beta_p1 = dd_beta + 1;
    expect_true(is_all_aprx_equal(gr, d_beta_p1, 1e-5));
    expect_true(is_all_aprx_equal(H, dd_beta_p1, 1e-5));
  }
}

context("Test mode_approximation") {
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
      create_mat<2L, 2L>({1.72420197619348, -0.816978073329191, -0.816978073329191, 1.00305631517286}),
      create_vec<2L>({ -0.674831912733073, 0.683308426878151 }),
      create_mat<2L, 2L>({ -0.319733453625652, -0.166245033819841, -0.166245033819841, -0.324366892661865 })
    );
  }

  test_that("Test mode_approximation with poisson_log") {
    /*  R code
     y <- c(0, 0, 0, 2, 2)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = poisson("log"))
     */
    test_func<poisson_log>(
      create_vec<5L>({0, 0, 0, 2, 2}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<0L>({ }),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({-2.0406171043706, 1.14695374876214}),
      create_mat<2L, 2L>({1.08005976284406, -0.481444987243251, -0.481444987243251, 0.360699945039141}),
      create_vec<2L>({ -0.943108369825444, -2.31155959374901 }),
      create_mat<2L, 2L>({ -2.26155575102543, -0.848023953278823, -0.848023953278823, -1.53249092852642 })
    );
  }

  test_that("Test mode_approximation with Gamma_log") {
    /*  R code
     y <- c(1.114409, 0.002153, 0.678375, 0.153124, 2.203468)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = Gamma("log"), disp = 2)
     */
    test_func<Gamma_log>(
      create_vec<5L>({1.114409, 0.002153, 0.678375, 0.153124, 2.203468}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<1L>({2.}),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({-1.26002820140565, 0.732019629477068}),
      create_mat<2L, 2L>({1.92516619370979, -1.01468459759359, -1.01468459759359, 0.863496987384669}),
      create_vec<2L>({ -0.122805738950818, -0.559741464729249 }),
      create_mat<2L, 2L>({ -0.693518992224658, -0.246511212167728, -0.246511212167728, -0.383002495203044 })
    );
  }
  test_that("Test mode_approximation with gaussian_identity") {
    /*  R code
     y <- c(1.1, 0.14, 1.7, 0.13, -0.052)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = gaussian(), disp = 2)
     */
    test_func<gaussian_identity>(
      create_vec<5L>({1.1, 0.14, 1.7, 0.13, -0.052}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<1L>({2.}),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({-0.253907839645345, 0.672656052306334}),
      create_mat<2L, 2L>({1.40856099145504, -0.757380380162023, -0.757380380162023, 0.732549663623882}),
      create_vec<2L>({-0.192584000222586, 1.10758180019252 }),
      create_mat<2L, 2L>({-0.688533999977635, -0.342309999980215, -0.342309999980215, -0.658976000031672 })
    );
  }
}
