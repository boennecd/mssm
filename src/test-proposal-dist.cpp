#include "proposal_dist.h"
#include "utils-test.h"
#include <testthat.h>
#include <array>

/* R code used below
		func <- function(y, X, cfix, Z, w, Q, mu, family, offs, disp = numeric()){
		  library(mvtnorm)

      ll <- function(mea)
        switch(family$family,
          "binomial" = sum(dbinom(y * w, w, mea, log = TRUE)),
          "poisson"  = sum(w * dpois(y, mea, log = TRUE)),
          "Gamma"    = sum(w * dgamma(y, 1/disp, scale = mea * disp, log = TRUE)),
          "gaussian" = sum(w * dnorm(y, mea, sd = sqrt(disp), log = TRUE)),
          stop("not implemented"))

		  off <- X %*% cfix + offs
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
         eta <-  drop(X %*% cfix + Z %*% mu + offs)
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
   arma::vec disp, arma::mat Q, arma::vec offs, arma::vec mu, arma::vec mode,
   arma::mat Neg_Inv_Hes, arma::vec d_beta, arma::mat dd_beta)
{
  dist_T family(Y, X, cfix, Z, &w, disp, offs);
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

  expect_true(family.state_stat_dim_grad(log_densty) == 0L);
  expect_true(family.state_stat_dim_grad(gradient)   == 0L);
  expect_true(family.state_stat_dim_grad(Hessian)    == 0L);

  expect_true(family.state_stat_dim_hess(log_densty) == 0L);
  expect_true(family.state_stat_dim_hess(gradient)   == 0L);
  expect_true(family.state_stat_dim_hess(Hessian)    == 0L);

  constexpr bool has_disp =
    std::is_base_of<exp_family_w_disp, dist_T>::value;
  const arma::uword dim =
    has_disp ? cfix.n_elem + 1L : cfix.n_elem;
  expect_true(family.obs_stat_dim(log_densty) == 0L);
  expect_true(family.obs_stat_dim(gradient)   == dim);
  expect_true(family.obs_stat_dim(Hessian)    == dim * (1L + dim));

  expect_true(family.obs_stat_dim_grad(log_densty) == 0L);
  expect_true(family.obs_stat_dim_grad(gradient)   == dim);
  expect_true(family.obs_stat_dim_grad(Hessian)    == dim);

  expect_true(family.obs_stat_dim_hess(log_densty) == 0L);
  expect_true(family.obs_stat_dim_hess(gradient)   == 0L);
  expect_true(family.obs_stat_dim_hess(Hessian)    == dim * dim);

  const arma::uword cend = has_disp ? dim - 2L : dim - 1L;

  {
    /* gradient only */
    arma::vec gr(dim, arma::fill::zeros);
    family.comp_stats_state_only(mu, gr.memptr(), gradient);

    arma::vec tv = gr.subvec(0L, cend);
    expect_true(is_all_aprx_equal(tv, d_beta, 1e-5));

    /* add something to start with */
    gr.fill(1.);
    family.comp_stats_state_only(mu, gr.memptr(), gradient);

    arma::vec d_beta_p1 = d_beta + 1;
    tv = gr.subvec(0L, cend);
    expect_true(is_all_aprx_equal(tv, d_beta_p1, 1e-5));
  }

  {
    /* gradient and Hessian */
    std::vector<double> mem(dim * (dim + 1L));
    arma::vec gr(mem.data(), dim, false);
    arma::mat H (mem.data() + dim, dim, dim, false);
    family.comp_stats_state_only(mu, mem.data(), Hessian);

    arma::vec tv = gr.subvec(0L, cend);
    expect_true(is_all_aprx_equal(tv, d_beta, 1e-5));
    arma::mat tmp = H.submat(0L, 0L, cend, cend);
    expect_true(is_all_aprx_equal(tmp, dd_beta, 1e-5));

    /* add something to start with */
    gr.fill(1.);
    H.fill(1.);
    family.comp_stats_state_only(mu, mem.data(), Hessian);

    arma::vec d_beta_p1 = d_beta + 1;
    arma::mat dd_beta_p1 = dd_beta + 1;

    tv = gr.subvec(0L, cend);
    expect_true(is_all_aprx_equal(tv, d_beta_p1, 1e-5));
    tmp = H.submat(0L, 0L, cend, cend);
    expect_true(is_all_aprx_equal(tmp, dd_beta_p1, 1e-5));
  }
}

context("Test mode_approximation") {
  test_that("Test mode_approximation with binomial_logit") {
    /*  R code
     y <- c(1, 1, 2, 0, 1)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     dput(y <- y / w)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = binomial("logit"), offs = offs)
     */
    test_func<binomial_logit>(
      create_vec<5L>({1., 1., 1., 0., 0.333333333333333}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      arma::vec(),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ -0.108347603592228, -0.0385530861670482 }),
      create_mat<2L, 2L>({ 1.77015319261325, -0.806846139429892, -0.806846139429893, 1.01166064584202 }),
      create_vec<2L>({ -0.353816462390114, 0.307381274122112 }),
      create_mat<2L, 2L>({ -0.299857413431918, -0.147059696961003, -0.147059696961003, -0.268891531492241 })
    );
  }

  test_that("Test mode_approximation with binomial_cloglog") {
    /*  R code
    y <- c(1, 1, 2, 0, 1)
    X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
    cfix <- c(.5, -.3)
    Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
    w <- c(1, 1, 2, 1, 3)
    dput(y <- y / w)
    Q <- matrix(c(4, 2, 2, 6), 2L)
    offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
    mu <- c(-1, 1)
    func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = binomial("cloglog"), offs = offs)
    */
    test_func<binomial_cloglog>(
      create_vec<5L>({1., 1., 1., 0., 0.333333333333333}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      arma::vec(),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ 0.028263372927748, -0.607925854870125 }),
      create_mat<2L, 2L>({ 1.41054525834946, -0.692233490041432, -0.692233490041431, 0.629834595722303 }),
      create_vec<2L>({ -2.38003778071049, -0.816156416318663 }),
      create_mat<2L, 2L>({ -1.70821724733621, -0.541933608428215, -0.541933608428215, -0.833266341121643 })
    );
  }

  test_that("Test mode_approximation with binomial_probit") {
    /*  R code
     y <- c(1, 1, 2, 0, 2)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     dput(y <- y / w)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = binomial("probit"), offs = offs)
     */
    test_func<binomial_probit>(
      create_vec<5L>({1., 1., 1., 0., 0.666666666666667}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      arma::vec(),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ 0.296218806751709, -0.0169477214065822 }),
      create_mat<2L, 2L>({ 1.48694166880951, -0.709302975602664, -0.709302975602664, 0.656500390842832 }),
      create_vec<2L>({ -0.134981616682794, 0.143703900098055 }),
      create_mat<2L, 2L>({ -0.725861590804269, -0.334735039906621, -0.334735039906621, -0.581206708842952 })
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
     offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = poisson("log"), offs = offs)
     */
    test_func<poisson_log>(
      create_vec<5L>({0, 0, 0, 2, 2}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      arma::vec(),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({-2.56086700642316, 0.978107611335817}),
      create_mat<2L, 2L>({0.966379231940908, -0.437398143418101, -0.437398143418101, 0.344832867881286}),
      create_vec<2L>({ -2.64218474211349, -5.64688602543408 }),
      create_mat<2L, 2L>({ -2.92285534644817, -1.51299584380444, -1.51299584380444, -3.31235598336682 })
    );
  }

  test_that("Test mode_approximation with poisson_sqrt") {
    /*  R code
     y <- c(0, 0, 0, 2, 2)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = poisson("sqrt"), offs = offs)
     */
    test_func<poisson_sqrt>(
      create_vec<5L>({0, 0, 0, 2, 2}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      arma::vec(),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ -2.17992378042166, 1.87000894550256 }),
      create_mat<2L, 2L>({ 0.612979755959768, -0.34024026490344, -0.34024026490344, 0.231178493264105 }),
      create_vec<2L>({ 4.79988298131378, -2.5346406737559 }),
      create_mat<2L, 2L>({ -9.1893200905747, -2.33833903505377, -2.33833903505377, -3.10467527565322 })
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
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = Gamma("log"), disp = 2, offs = offs)
     */
    test_func<Gamma_log>(
      create_vec<5L>({1.114409, 0.002153, 0.678375, 0.153124, 2.203468}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<1L>({2.}),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ -1.60962153591909, 0.630777922429041 }),
      create_mat<2L, 2L>({ 2.01377124968931, -1.02674638861131, -1.02674638861131, 0.849247864352239 }),
      create_vec<2L>({ -0.387494472278193, -0.928645967289695 }),
      create_mat<2L, 2L>({ -0.566377762913443, -0.155317439114318, -0.155317439114318, -0.181321427067543 })
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
     offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = gaussian(), disp = 2, offs = offs)
     */
    test_func<gaussian_identity>(
      create_vec<5L>({1.1, 0.14, 1.7, 0.13, -0.052}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<1L>({2.}),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ -0.778708820184306, 0.462145589256333 }),
      create_mat<2L, 2L>({ 1.40856099086251, -0.757380379841547, -0.757380379841547, 0.732549663445067 }),
      create_vec<2L>({ -0.705383999958155, 0.138301799941341 }),
      create_mat<2L, 2L>({ -0.688534000038116, -0.342309999929884, -0.342309999929884, -0.658976000031672 })
    );
  }
  test_that("Test mode_approximation with gaussian_log") {
    /*  R code
    y <- c(1.1, 0.14, 1.7, 0.13, -0.052)
    X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
    cfix <- c(.5, -.3)
    Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
    w <- c(1, 1, 2, 1, 3)
    Q <- matrix(c(4, 2, 2, 6), 2L)
    offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
    mu <- c(-1, 1)
    func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = gaussian("log"), disp = 2, offs = offs)
    */
    test_func<gaussian_log>(
      create_vec<5L>({1.1, 0.14, 1.7, 0.13, -0.052}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<1L>({2.}),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ -1.41784588567423, -0.292580027441991  }),
      create_mat<2L, 2L>({ 1.73297468241502, -0.515323798975639, -0.515323798975639, 0.79979606241897 }),
      create_vec<2L>({ -6.18359370827735, -5.24595157430195 }),
      create_mat<2L, 2L>({ -6.02819782506261, -2.97995654774978, -2.97995654774978, -6.54275013797885 })
    );
  }
  test_that("Test mode_approximation with gaussian_inverse") {
    /*  R code
     y <- c(1.1, 0.14, 1.7, 0.13, -0.052)
     X <- matrix(c(0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072), ncol = 2L, byrow = TRUE)
     cfix <- c(.5, -.3)
     Z <- matrix(c(0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91), ncol = 2L, byrow = TRUE)
     w <- c(1, 1, 2, 1, 3)
     Q <- matrix(c(4, 2, 2, 6), 2L)
     offs <- c(0.6, 0.92, 0.9, 0.32, 0.11)
     mu <- c(-1, 1)
     func(y = y, X = X, cfix = cfix, Z = Z, w = w, Q = Q, mu = mu, family = gaussian("inverse"), disp = 2, offs = offs)
    */
    test_func<gaussian_inverse>(
      create_vec<5L>({1.1, 0.14, 1.7, 0.13, -0.052}),
      create_mat<2L, 5L>({0.51, 0.49, 0.38, 0.45, 0.078, 0.61, 0.14, 0.34, 0.56, 0.072}),
      create_vec<2L>({.5, -.3}),
      create_mat<2L, 5L>({0.19, 0.032, 0.96, 0.87, 0.65, 0.89, 0.12, 0.96, 0.51, 0.91}),
      create_vec<5L>({1, 1, 2, 1, 3}),
      create_vec<1L>({2.}),
      create_mat<2L, 2L>({4, 2, 2, 6}),
      create_vec<5L>({ 0.6, 0.92, 0.9, 0.32, 0.11 }),
      create_vec<2L>({-1, 1}),
      create_vec<2L>({ -0.456400960262501, 1.80085440669367  }),
      create_mat<2L, 2L>({ 1.9742943564148, -0.279320894386273, -0.279320894386273, 1.85454132619892 }),
      create_vec<2L>({ 2.75790426656431, 0.786721007649747 }),
      create_mat<2L, 2L>({ -7.03771120735055, -3.38323338229921, -3.38323338229921, -2.82184621474547 })
    );
  }
}
