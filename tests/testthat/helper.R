#####
# override defaults
formals(expect_known_value)$update <- FALSE
formals(expect_known_output)$update <- FALSE

# vector of elements that we want to test on mssmFunc object
mssmFunc_ele_to_check <- c(
  "terms_fixed", "terms_random", "control", "family")

# vector of elements that we want to test on mssm object
mssm_ele_to_check <- c(
  "pf_output", "terms_fixed", "terms_random", "control", "family")

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

# # Poisson data w/ log link
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
