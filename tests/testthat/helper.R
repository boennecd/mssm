#####
# override defaults
formals(expect_known_value)$update <- FALSE
formals(expect_known_output)$update <- FALSE

# vector of elements that we want to test on mssmFunc object
mssmFunc_ele_to_check <- c(
  "control", "family")

# vector of elements that we want to test on mssm object
mssm_ele_to_check <- c(
  "pf_output", "control", "family")

#####
# simulate or load data sets to use

# function to simulate state vector
.get_beta <- function(Q, Q0, F., n_periods){
  betas <- cbind(crossprod(chol(Q0),         rnorm(2L)                      ),
                 crossprod(chol(Q  ), matrix(rnorm((n_periods - 1L) * 2), 2)))
  betas <- t(betas)
  for(i in 2:nrow(betas))
    betas[i, ] <- betas[i, ] + F. %*% betas[i - 1L, ]

  betas
}

# function to simulate observations
.get_dat <- function(cfix, betas, sample_func, trans_func){
  n_periods <- nrow(betas)
  n_rng <- ncol(betas)
  n_fix <- length(cfix) - n_rng

  dat <- lapply(1:n_obs, function(id){
    x <- matrix(runif(n_periods * n_fix, -1, 1), nc = n_fix)
    z <- matrix(runif(n_periods * (n_rng - 1L), -1, 1), nc = n_rng - 1L)

    eta <- drop(cbind(1, x, z) %*% cfix + rowSums(cbind(1, z) * betas))
    y <- sample_func(n_periods, trans_func(eta))
    keep <- .5 > runif(n_periods)
    data.frame(y = y, x, Z = z, id = id, time_idx = 1:n_periods)[keep, ]
  })
  dat <- do.call(rbind, dat)
}

# # Poisson w/ log link
# n_periods <- 20L
# F. <- matrix(c(.5, .1, 0, .8), 2L)
# Q <- matrix(c(.5^2, .1, .1, .7^2), 2L)
# Q0 <- mssm:::.get_Q0(Q, F.)
# cfix <- c(-1, .2, .5)
# n_obs <- 20L
#
# set.seed(78727269)
# betas <- .get_beta(Q, Q0, F., n_periods)
# dat <- .get_dat(cfix, betas, sample_func = rpois, trans_func = exp)
#
# poisson_log <- list(data = dat, betas = betas, cfix = cfix, Q = Q, F. = F.)
# saveRDS(poisson_log, "poisson_log.RDS")
poisson_log <- readRDS("poisson_log.RDS")

# # Poisson w/ sqrt link
# n_periods <- 20L
# F. <- matrix(c(.5, .1, 0, .8), 2L)
# Q <- matrix(c(.2^2, -.1^2, -.1^2, .2^2), 2L)
# Q0 <- mssm:::.get_Q0(Q, F.)
# cfix <- c(4, .2, -2)
# n_obs <- 20L
#
# set.seed(78727269)
# betas <- .get_beta(Q, Q0, F., n_periods)
# dat <- .get_dat(
#   cfix, betas, sample_func = rpois, trans_func = trans_func = function(x) x * x)
#
# poisson_sqrt <- list(data = dat, betas = betas, cfix = cfix, Q = Q, F. = F.)
# saveRDS(poisson_sqrt, "poisson_sqrt.RDS")
poisson_sqrt <- readRDS("poisson_sqrt.RDS")

# # Binomial w/ logit
# n_periods <- 20L
# F. <- matrix(c(.5, .1, 0, .8), 2L)
# Q <-  matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
# Q0 <- mssm:::.get_Q0(Q, F.)
# cfix <- c(-1, .2, .5)
# n_obs <- 20L
#
# set.seed(78727269)
# betas <- .get_beta(Q, Q0, F., n_periods)
# dat <- .get_dat(cfix, betas,
#                 sample_func = function(n, mu) mu > runif(n),
#                 trans_func = function(x) 1 / (1 + exp(-x)))
#
# binomial_logit <- list(data = dat, betas = betas, cfix = cfix, Q = Q, F. = F.)
# saveRDS(binomial_logit, "binomial_logit.RDS")
binomial_logit <- readRDS("binomial_logit.RDS")

# # Binomial w/ cloglog
# n_periods <- 20L
# F. <- matrix(c(.5, .1, 0, .8), 2L)
# Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
# Q0 <- mssm:::.get_Q0(Q, F.)
# cfix <- c(0, .2, -1)
# n_obs <- 20L
#
# set.seed(78727269)
# betas <- .get_beta(Q, Q0, F., n_periods)
# dat <- .get_dat(cfix, betas,
#                 sample_func = function(n, mu) mu > runif(n),
#                 trans_func = function(x) -expm1(-exp(x)))
#
# binomial_cloglog <- list(data = dat, betas = betas, cfix = cfix, Q = Q, F. = F.)
# saveRDS(binomial_cloglog, "binomial_cloglog.RDS")
binomial_cloglog <- readRDS("binomial_cloglog.RDS")

# # Binomial w/ probit
# n_periods <- 20L
# F. <- matrix(c(.5, .1, 0, .8), 2L)
# Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
# Q0 <- mssm:::.get_Q0(Q, F.)
# cfix <- c(0, .2, -1)
# n_obs <- 20L
#
# set.seed(78727269)
# betas <- .get_beta(Q, Q0, F., n_periods)
# dat <- .get_dat(cfix, betas,
#                 sample_func = function(n, mu) mu > runif(n),
#                 trans_func = pnorm)
#
# binomial_probit <- list(data = dat, betas = betas, cfix = cfix, Q = Q, F. = F.)
# saveRDS(binomial_probit, "binomial_probit.RDS")
binomial_probit <- readRDS("binomial_probit.RDS")

# # Gamma w/ log
# n_periods <- 20L
# F. <- matrix(c(.9, -.3, 0, .5), 2L)
# Q <- matrix(c(.5^2, -.1^2, -.1^2, .9^2), 2L)
# Q0 <- mssm:::.get_Q0(Q, F.)
# cfix <- c(-1, .2, .5)
# n_obs <- 20L
#
# set.seed(78727269)
# betas <- .get_beta(Q, Q0, F., n_periods)
# disp <- 2.
# dat <- .get_dat(cfix, betas,
#                 sample_func = function(n, mu)
#                   rgamma(n = n, shape = 1/disp, scale = mu * disp),
#                 trans_func = function(x) exp(x))
#
# Gamma_log <- list(data = dat, betas = betas, cfix = cfix, Q = Q, F. = F.,
#                   disp = disp)
# saveRDS(Gamma_log, "Gamma_log.RDS")
Gamma_log <- readRDS("Gamma_log.RDS")
