.get_beta <- function(Q, Q0, F., n_periods){
  betas <- cbind(crossprod(chol(Q0),         rnorm(2L)                      ),
                 crossprod(chol(Q  ), matrix(rnorm((n_periods - 1L) * 2), 2)))
  betas <- t(betas)
  for(i in 2:nrow(betas))
    betas[i, ] <- betas[i, ] + F. %*% betas[i - 1L, ]

  betas
}

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

adam <- function(
  object, n_it = 150L, mp = .9, vp = .999, disp = numeric(),
  lr = 1e-2, avg_start = n_it + 1L, cfix, F., Q,  verbose = FALSE)
{
  # make checks
  stopifnot(
    inherits(object, "mssmFunc"), object$control$what == "gradient", n_it > 0L,
    lr > 0.)
  n_fix <- nrow(object$X)
  n_rng <- nrow(object$Z)

  # objects for estimates at each iteration and log-likelihood approximations
  ests <- matrix(
    NA_real_, n_it + 1L, n_fix + n_rng * n_rng + n_rng * (n_rng  + 1L) / 2L)
  ests[1L, ] <- c(cfix, F., Q[lower.tri(Q, diag = TRUE)])
  lls <- rep(NA_real_, n_it)

  # indices of the different components
  idx_fix <- 1:n_fix
  idx_F   <- 1:(n_rng * n_rng) + n_fix
  idx_Q   <- 1:(n_rng * (n_rng  + 1L) / 2L) + n_fix + n_rng * n_rng

  # we only want the lower part of `Q` so we make the following map for the
  # gradient
  library(matrixcalc) # TODO: get rid of this
  gr_map <- matrix(
    0., nrow = ncol(ests), ncol = length(cfix) + length(F.) + length(Q))
  gr_map[idx_fix, idx_fix] <- diag(length(idx_fix))
  gr_map[idx_F  , idx_F] <- diag(length(idx_F))
  dup_mat <- duplication.matrix(ncol(Q))
  gr_map[idx_Q  , -c(idx_fix, idx_F)] <- t(dup_mat)

  # function to set the parameters
  set_parems <- function(i){
    # select whether or not to average
    idx <- if(i > avg_start) avg_start:i else i

    # set new parameters
    cfix <<-             colMeans(ests[idx, idx_fix, drop = FALSE])
    F.[] <<-             colMeans(ests[idx, idx_F  , drop = FALSE])
    Q[]  <<- dup_mat %*% colMeans(ests[idx, idx_Q  , drop = FALSE])

  }

  # run gradient decent
  max_half <- 25L
  m <- NULL
  v <- NULL
  failed <- FALSE
  for(i in 1:n_it + 1L){
    # get gradient. First, run the particle filter
    filter_out <- object$pf_filter(
      cfix = cfix, disp = disp, F. = F., Q = Q, seed = NULL)
    lls[i - 1L] <- c(logLik(filter_out))

    # then get the gradient associated with each particle and the log
    # normalized weight of the particles
    grads <- tail(filter_out$pf_output, 1L)[[1L]]$stats
    ws    <- tail(filter_out$pf_output, 1L)[[1L]]$ws_normalized

    # compute the gradient and take a small step
    grad <- colSums(t(grads) * drop(exp(ws)))
    if(any(!is.finite(grad))){
      warning("non-finite gradient")
      failed <- TRUE
      break

    }

    m <- if(is.null(m)) (1 - mp) * grad   else mp * m + (1 - mp) * grad
    v <- if(is.null(v)) (1 - vp) * grad^2 else vp * v + (1 - vp) * grad^2
    mh <- m / (1 - mp^(i - 1))
    vh <- v / (1 - vp^(i - 1))
    de <- mh / sqrt(vh + 1e-8)

    lr_i <- lr
    k <- 0L
    while(k < max_half){
      ests[i, ] <- ests[i - 1L, ] + lr_i * gr_map %*% de
      set_parems(i)

      # check that Q is positive definite and the system is stationary
      c1 <- all(abs(eigen(F.)$values) < 1)
      c2 <- all(eigen(Q)$values > 0)
      if(c1 && c2)
        break

      # decrease learning rate
      lr_i <- lr_i * .5
      k <- k + 1L
    }

    # check if we failed to find a value within our constraints
    if(k == max_half){
      warning("failed to find solution within constraints")
      failed <- TRUE
      break
    }

    # print information if requested
    if(verbose){
      cat(sprintf(
        "\nIt %5d: log-likelihood (current, max) %12.2f, %12.2f\n",
        i - 1L, logLik(filter_out), max(lls, na.rm = TRUE)),
        rep("-", 66), "\n", sep = "")
      cat("cfix\n")
      print(cfix)
      cat("F\n")
      print(F.)
      cat("Q\n")
      print(Q)
      cat(sprintf("Gradient norm: %10.4f\n", norm(t(grad))))
      print(get_ess(filter_out))

    }
  }

  list(estimates = ests, logLik = lls, F. = F., Q = Q, cfix = cfix,
       failed = failed)
}

#####
# poisson w/ log link
n_periods <- 100L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.5^2, .1, .1, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 20L

set.seed(78727270)
betas <- .get_beta(Q, Q0, F., n_periods)
dat <- .get_dat(cfix, betas, sample_func = rpois, trans_func = exp)

ll_func <- mssm(
  fixed = y ~ x + Z, family = poisson(), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = 1e-2, which_sampler = "bootstrap"))
sta <- coef(glm(y ~ x  + Z, poisson(), dat))
system.time(
  res <- adam(
    ll_func, F. = diag(.5, 2), Q = diag(1, 2), cfix = sta, verbose = TRUE,
    n_it = 200L, lr = .01))

plot(res$logLik)
plot(tail(res$logLik, 150))

#####
# poisson w/ sqrt
n_periods <- 200L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.2^2, -.1^2, -.1^2, .2^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(4, .2, -2)
n_obs <- 200L

set.seed(78727271)
betas <- .get_beta(Q, Q0, F., n_periods)
matplot(betas, type = "l")
dat <- .get_dat(cfix, betas, sample_func = rpois, trans_func = function(x) x * x)

ll_func <- mssm(
  fixed = y ~ x + Z, family = poisson("sqrt"), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .1, covar_fac = 2))
(sta <- coef(glm(y ~ x  + Z, poisson("sqrt"), dat, start = cfix)))
system.time(
  res <- adam(
    ll_func, F. = diag(.2, 2), Q = diag(.01^2, 2), cfix = sta, verbose = TRUE,
    n_it = 200L, lr = .01))

system.time(
  res2 <- adam(
    ll_func, F. = res$F., Q = res$Q, cfix = res$cfix, verbose = TRUE,
    n_it = 150L, lr = .001))

plot(res2$logLik)
plot(tail(res2$logLik, 100))

o <- ll_func$pf_filter(F. = F., Q = Q, cfix = cfix, disp = numeric())
logLik(o)

#####
# binomial w/ logit
n_periods <- 100L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 100L

set.seed(78727270)
betas <- .get_beta(Q, Q0, F., n_periods)
dat <- .get_dat(
  cfix, betas,
  sample_func = function(n, mu) mu > runif(n),
  trans_func = function(x) 1 / (1 + exp(-x)))

ll_func <- mssm(
  fixed = y ~ x + Z, family = binomial(), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 2L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .1, covar_fac = 2))
sta <- coef(glm(y ~ x  + Z, binomial(), dat))
system.time(
  res <- adam(
    ll_func, F. = diag(.5, 2), Q = diag(1, 2), cfix = sta, verbose = TRUE,
    n_it = 200L, lr = .005))

res$F.
res$Q
plot(res$logLik)
plot(tail(res$logLik, 100))
matplot(res$estimates, col = 1:10)
abline(h = c(cfix, F., Q), lty = 2, col = 1:10)

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., Q = res$Q, disp = numeric())
logLik(o)
plot(get_ess(o))
po <- plot(o)
matplot(betas, type = "l", lty = 1)
matplot(t(po$means), type = "l", lty = 2, add = TRUE)

#####
# binomial w/ cloglog
n_periods <- 100L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(0, .2, -1)
n_obs <- 200L

set.seed(78727274)
betas <- .get_beta(Q, Q0, F., n_periods)
dat <- .get_dat(
  cfix, betas,
  sample_func = function(n, mu) mu > runif(n),
  trans_func = function(x) -expm1(-exp(x)))

ll_func <- mssm(
  fixed = y ~ x + Z, family = binomial("cloglog"), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .1, covar_fac = 2))
sta <- coef(glm_fit <- glm(y ~ x  + Z, binomial("cloglog"), dat))
logLik(glm_fit)
system.time(
  res <- adam(
    ll_func, F. = diag(.5, 2), Q = diag(.2^2, 2), cfix = sta, verbose = TRUE,
    n_it = 200L, lr = .01))

logLik(pf_out <- ll_func$pf_filter(cfix = cfix, F. = F., Q = Q, disp = numeric()))

o <- plot(pf_out)

matplot(cbind(o$means[2, ], o$lbs[2, ], o$ubs[2, ], betas[,2]), type = "l", lty = c(1, 2, 2, 3), col = "black")

plot(res$logLik)
plot(tail(res$logLik, 50))

#####
# binomial w/ probit
n_periods <- 100L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(0, .2, -1)
n_obs <- 200L

set.seed(78727276)
betas <- .get_beta(Q, Q0, F., n_periods)
dat <- .get_dat(
  cfix, betas,
  sample_func = function(n, mu) mu > runif(n),
  trans_func = pnorm)

ll_func <- mssm(
  fixed = y ~ x + Z, family = binomial("probit"), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    covar_fac = 2))
sta <- coef(glm_fit <- glm(y ~ x  + Z, binomial("probit"), dat))
logLik(glm_fit)
system.time(
  res <- adam(
    ll_func, F. = F., Q = Q, cfix = cfix, verbose = TRUE,
    n_it = 200L, lr = .01))

logLik(pf_out <- ll_func$pf_filter(cfix = cfix, F. = F., Q = Q, disp = numeric()))

o <- plot(pf_out)

matplot(cbind(o$means[2, ], o$lbs[2, ], o$ubs[2, ], betas[,2]), type = "l", lty = c(1, 2, 2, 3), col = "black")

plot(res$logLik)
plot(tail(res$logLik, 50))

#####
# gamma w/ log link
n_periods <- 400L
F. <- matrix(c(.9, -.3, 0, .5), 2L)
Q <- matrix(c(.5^2, -.1^2, -.1^2, .9^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 200L

set.seed(78727271)
betas <- .get_beta(Q, Q0, F., n_periods)
matplot(betas, type = "l")
disp <- 2.
dat <- .get_dat(
  cfix, betas,
  sample_func = function(n, mu)
    rgamma(n = n, shape = 1/disp, scale = mu * disp),
  trans_func = function(x) exp(x))

ll_func <- mssm(
  fixed = y ~ x + Z, family = Gamma("log"), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 500L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .1, covar_fac = 2))

(sta <- coef(glm_fit <- glm(y ~ x  + Z, Gamma("log"), dat)))
logLik(glm_fit)
system.time(
  res <- adam(
    ll_func, F. = diag(.5, 2), Q = diag(1, 2), cfix = sta, verbose = TRUE,
    n_it = 200L, lr = .11, disp = disp))
plot(res$logLik)
plot(tail(res$logLik, 100))

res2 <- adam(
  ll_func, F. = F., Q = Q, cfix = cfix, verbose = TRUE,
  n_it = 100L, lr = .01, disp = disp)

logLik(ll_func$pf_filter(cfix = cfix, disp = disp, F. = F., Q = Q))
