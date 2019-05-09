.get_beta <- function(Q, Q0, F., n_periods){
  betas <- cbind(crossprod(chol(Q0),         rnorm(2L)                      ),
                 crossprod(chol(Q  ), matrix(rnorm((n_periods - 1L) * 2), 2)))
  betas <- t(betas)
  for(i in 2:nrow(betas))
    betas[i, ] <- betas[i, ] + F. %*% betas[i - 1L, ]

  betas
}

.get_dat <- function(cfix, betas, sample_func, trans_func, foffset = NULL){
  n_periods <- nrow(betas)
  n_rng <- ncol(betas)
  n_fix <- length(cfix) - n_rng

  dat <- lapply(1:n_obs, function(id){
    x <- matrix(runif(n_periods * n_fix, -1, 1), nc = n_fix)
    z <- matrix(runif(n_periods * (n_rng - 1L), -1, 1), nc = n_rng - 1L)

    offs <- if(!is.null(foffset))
      foffset(n_periods) else 0.

    eta <- drop(cbind(1, x, z) %*% cfix + rowSums(cbind(1, z) * betas)) +
      offs

    y <- sample_func(n_periods, trans_func(eta))
    keep <- .5 > runif(n_periods)
    data.frame(y = y, x, Z = z, id = id, time_idx = 1:n_periods,
               offs = offs)[keep, ]
  })
  dat <- do.call(rbind, dat)
}

adam <- function(
  object, n_it = 150L, mp = .9, vp = .999, lr = .01, cfix, F., Q,
  verbose = FALSE, disp = numeric())
{
  # make checks
  stopifnot(
    inherits(object, "mssmFunc"), object$control$what == "gradient", n_it > 0L,
    lr > 0., mp > 0, mp < 1, vp > 0, vp < 1)
  n_fix <- nrow(object$X)
  n_rng <- nrow(object$Z)

  # objects for estimates at each iteration, gradient norms, and
  # log-likelihood approximations
  has_dispersion <-
    any(sapply(c("^Gamma", "^gaussian"), grepl, x = object$family))
  n_params <-
    has_dispersion + n_fix + n_rng * n_rng + n_rng * (n_rng  + 1L) / 2L
  ests <- matrix(NA_real_, n_it + 1L, n_params)
  ests[1L, ] <- c(
    cfix,  if(has_dispersion) disp else NULL, F.,
    Q[lower.tri(Q, diag = TRUE)])
  grad_norm <- lls <- rep(NA_real_, n_it)

  # indices of the different components
  idx_fix   <- 1:n_fix
  if(has_dispersion)
    idx_dip <- n_fix + 1L
  idx_F     <- 1:(n_rng * n_rng) + n_fix + has_dispersion
  idx_Q     <- 1:(n_rng * (n_rng  + 1L) / 2L) + n_fix + n_rng * n_rng +
    has_dispersion

  # function to set the parameters
  library(matrixcalc) # TODO: get rid of this
  dup_mat <- duplication.matrix(ncol(Q))
  set_parems <- function(i){
    cfix   <<-             ests[i, idx_fix]
    if(has_dispersion)
      disp <<-             ests[i, idx_dip]
    F.[]   <<-             ests[i, idx_F  ]
    Q[]    <<- dup_mat %*% ests[i, idx_Q  ]

  }

  # run algorithm
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
    grad_norm[i - 1L] <- norm(t(grad))

    m <- if(is.null(m)) (1 - mp) * grad   else mp * m + (1 - mp) * grad
    v <- if(is.null(v)) (1 - vp) * grad^2 else vp * v + (1 - vp) * grad^2
    mh <- m / (1 - mp^(i - 1))
    vh <- v / (1 - vp^(i - 1))
    de <- mh / sqrt(vh + 1e-8)

    lr_i <- lr
    k <- 0L
    while(k < max_half){
      ests[i, ] <- ests[i - 1L, ] + lr_i * de
      set_parems(i)

      # check that Q is positive definite, the dispersion parameter is
      # positive, and the system is stationary
      c1 <- all(abs(eigen(F.)$values) < 1)
      c2 <- all(eigen(Q)$values > 0)
      c3 <- !has_dispersion || disp > 0.
      if(c1 && c2 && c3)
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
      if(has_dispersion)
        cat(sprintf("Dispersion: %20.8f\n", disp))
      cat("F\n")
      print(F.)
      cat("Q\n")
      print(Q)
      cat(sprintf("Gradient norm: %10.4f\n", grad_norm[i - 1L]))
      print(get_ess(filter_out))

    }
  }

  list(estimates = ests, logLik = lls, F. = F., Q = Q, cfix = cfix,
       disp = disp, failed = failed, grad_norm = grad_norm)
}

get_grad_n_obs_info <- function(object){
  stopifnot(inherits(object, "mssm"))

  # get dimension of the components
  n_rng   <- nrow(object$Z)
  dim_fix <- nrow(object$X)
  dim_rng <- n_rng * n_rng + ((n_rng + 1L) * n_rng) / 2L
  has_dispersion <-
    any(sapply(c("^Gamma", "^gaussian"), grepl, x = object$family))

  # get quantities for each particle
  quants <- tail(object$pf_output, 1L)[[1L]]$stats
  ws     <- tail(object$pf_output, 1L)[[1L]]$ws_normalized

  # get mean estimates
  meas <- colSums(t(quants) * drop(exp(ws)))

  # separate out the different components. Start with the gradient
  idx <- dim_fix + dim_rng + has_dispersion
  grad <- meas[1:idx]
  dim_names <- names(grad)

  # then the observed information matrix
  hess <- matrix(
    meas[-(1:idx)], dim_fix + dim_rng + has_dispersion,
    dimnames = list(dim_names, dim_names))

  list(grad = grad, hess = hess)
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
    which_ll_cp = "KD", aprx_eps = 1e-2))
sta <- coef(glm_fit <- glm(y ~ x  + Z, poisson(), dat))
la <- ll_func$Laplace(
  cfix = sta, disp = numeric(), F. = diag(.5, 2), Q = diag(1, 2), trace = 1L)
logLik(glm_fit)

system.time(
  res <- adam(
    ll_func, F. = la$F., Q = la$Q, cfix = la$cfix,
    verbose = TRUE, n_it = 50L, lr = .01))
la$Laplace

plot(res$logLik)

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., Q = res$Q, disp = numeric(), N_part = 10000L,
  what = "Hessian", seed = NULL)

he <- get_grad_n_obs_info(o)
res$cfix
sqrt(diag(solve(-he$hess)))
c(res$F., res$Q)

#####
# poisson w/ log link and offsets
n_periods <- 100L
F. <- matrix(c(.5, .3, 0, .8), 2L)
Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 20L

set.seed(78727270)
betas <- .get_beta(Q, Q0, F., n_periods)
dat <- .get_dat(cfix, betas, sample_func = rpois, trans_func = exp,
                foffset = function(n) rnorm(n))

(sta <- coef(glm_fit <- glm(y ~ x  + Z, poisson(), dat, offset = offs)))

ll_func <- mssm(
  fixed = y ~ x + Z, family = poisson(), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = 1e-2))

logLik(glm_fit)
logLik(ll_func$pf_filter(cfix = sta, disp = numeric(), F. = diag(.001, 2),
                         Q = diag(1e-8, 2)))

system.time(
  res <- adam(
    ll_func, F. = diag(.5, 2), Q = diag(1, 2), cfix = sta, verbose = TRUE,
    n_it = 200L, lr = .01))

plot(res$logLik)
res$F.
res$Q

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., disp = numeric(), Q = res$Q, what = "Hessian",
  N_part = 2000L)
o <- get_grad_n_obs_info(o)
res$cfix
sqrt(diag(solve(-o$ddg)))
c(res$F., res$Q)
sqrt(diag(solve(-o$ddf)))

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

la <- ll_func$Laplace(
  cfix = sta, disp = numeric(), F. = diag(.5, 2), Q = diag(1, 2), trace = 1L)
logLik(glm_fit)

system.time(
  res <- adam(
    ll_func, F. = la$F., Q = la$Q, cfix = la$cfix,
    verbose = TRUE, n_it = 100L, lr = .001))
print(la$Laplace$logLik, digits = 6)

plot(res$logLik)

o <- ll_func$pf_filter(F. = F., Q = Q, cfix = cfix, disp = numeric())
logLik(o)

#####
# binomial w/ logit
n_periods <- 400L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 100L

set.seed(78727271)
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
sta <- coef(glm_fit <- glm(y ~ x  + Z, binomial(), dat))
la <- ll_func$Laplace(
  cfix = sta, disp = numeric(), F. = diag(.5, 2), Q = diag(1, 2), trace = 1L)
logLik(glm_fit)

system.time(
  res <- adam(
    ll_func, F. = la$Laplace$F., Q = la$Laplace$Q, cfix = la$Laplace$cfix,
    verbose = TRUE, n_it = 50L, lr = .005))

res$F.
la$Laplace$F.
res$Q
la$Laplace$Q
res$cfix
la$Laplace$cfix

plot(res$logLik)
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
# binomial w/ logit and grouped data
n_periods <- 1000L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 100L

set.seed(78727223)
betas <- .get_beta(Q, Q0, F., n_periods)
dat <- local({
  n_periods <- nrow(betas)
  n_rng <- ncol(betas)
  n_fix <- length(cfix) - n_rng

  dat <- lapply(1:n_obs, function(id){
    size <- sample.int(9L, n_periods, replace = TRUE) + 1L
    x <- matrix(runif(n_periods * n_fix, -1, 1), nc = n_fix)
    z <- matrix(runif(n_periods * (n_rng - 1L), -1, 1), nc = n_rng - 1L)

    eta <- drop(cbind(1, x, z) %*% cfix + rowSums(cbind(1, z) * betas))
    y <- rbinom(n_periods, size = size, prob = (1 + exp(-eta))^(-1))

    keep <- .5 > runif(n_periods)
    data.frame(
      y = y, x, Z = z, id = id, time_idx = 1:n_periods, size = size)[keep, ]
  })
  dat <- do.call(rbind, dat)
})

ll_func <- mssm(
  fixed = I(y/size) ~ x + Z, family = binomial("logit"), data = dat,
  weights = size, random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .1, covar_fac = 2))

(sta <- coef(
  glm_fit <- glm(I(y/size) ~ x  + Z, binomial("logit"), weights = size, dat)))
la <- ll_func$Laplace(cfix = sta, disp = numeric(), F. = diag(.5, 2),
                      Q = diag(1, 2), trace = 1L)

system.time(
  res <- adam(
    ll_func, F. = la$Laplace$F., Q = la$Laplace$Q, cfix = la$Laplace$cfix,
    verbose = TRUE, n_it = 50L, lr = .01))

plot(res$logLik)
la$Laplace$logLik
la$Laplace$F.
res$F
la$Laplace$Q

la2 <- ll_func$Laplace(cfix = res$cfix, disp = numeric(), F. = res$F.,
                      Q = res$Q, trace = 1L)
print(la2$logLik, digits = 6)
print(tail(res$logLik), digits = 6)

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., Q = res$Q, disp = numeric(), seed = NULL,
  what = "Hessian", N_part = 5000L)

he <- get_grad_n_obs_info(o)
sqrt(diag(solve(-he$hess)))
res$cfix
res$F.
res$Q

#####
# binomial w/ cloglog
n_periods <- 400L
F. <- matrix(c(.5, .1, 0, .8), 2L)
Q <- matrix(c(.5^2, -.5^2, -.5^2, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(0, .2, -1)
n_obs <- 50L

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
la <- ll_func$Laplace(cfix = sta, disp = numeric(), F. = diag(.5, 2), Q = diag(1, 2), trace = 1L)
logLik(glm_fit)

system.time(
  res <- adam(
    ll_func, F. = la$F., Q = la$Q, cfix = la$cfix, verbose = TRUE,
    n_it = 50L, lr = .01))

plot(res$logLik)
lines(smooth.spline(seq_along(res$logLik), res$logLik))
print(la$logLik, digits = 6)

o <- ll_func$pf_filter(
  cfix = res$cfix, disp = numeric(), Q = res$Q, F. = res$F, what = "Hessian",
  N_part = 10000L)

out <- get_grad_n_obs_info(o)
sqrt(diag(solve(-out$hess)))

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
    covar_fac = 2, which_ll_cp = "KD", aprx_eps = .01))
sta <- coef(glm_fit <- glm(y ~ x  + Z, binomial("probit"), dat))
lap <- ll_func$Laplace(
  cfix = sta, disp = numeric(), F. = diag(.5, 2), Q = diag(1, 2), trace = 1L)

logLik(glm_fit)
system.time(
  res <- adam(
    ll_func, F. = lap$F., Q = lap$Q, cfix = lap$cfix, verbose = TRUE,
    n_it = 50L, lr = .01))

plot(res$logLik)
print(lap$logLik, digits = 6)
print(res$logLik, 6)

res$F.
lap$F.

res$Q
lap$Q

res$cfix
lap$cfix

o <- ll_func$pf_filter(
  cfix = res$cfix, disp = numeric(), Q = res$Q, F. = res$F, what = "Hessian",
  N_part = 10000L)
out <- get_grad_n_obs_info(o)
sqrt(diag(solve(-out$hess)))

#####
# gamma w/ log link
n_periods <- 400L
F. <- matrix(c(.9, -.3, 0, .5), 2L)
Q <- matrix(c(.5^2, -.1^2, -.1^2, .9^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 50L

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
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .1, covar_fac = 2))

(sta <- coef(glm_fit <- glm(y ~ x  + Z, Gamma("log"), dat)))

lpa <- ll_func$Laplace(
  cfix = sta, disp = disp, F. = diag(.5, 2), Q = diag(1, 2), trace = 1L)
logLik(glm_fit)
logLik(lpa)

system.time(
  res <- adam(
    ll_func, F. = lpa$F., Q = lpa$Q, cfix = lpa$cfix, verbose = TRUE,
    n_it = 50L, lr = .001, disp = lpa$dispersion))
plot(res$logLik)

plot(res$logLik)
plot(res$grad_norm)

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., Q = res$Q, disp = 3, seed = NULL,
  what = "Hessian", N_part = 5000L)

of <- get_grad_n_obs_info(o)

sqrt(diag(solve(-of$hess)))
res$cfix
res$disp
res$F.
res$Q

#####
# Gaussian w/ identity link -- could just use Kalman fitler
n_periods <- 400L
F. <- matrix(c(.5, .2, -.1, .8), 2L)
Q <- matrix(c(.5^2, .1, .1, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 100L
disp <- 3

set.seed(78727273)

betas <- .get_beta(Q, Q0, F., n_periods)
dat <- .get_dat(cfix, betas, sample_func = function(n, mu) rnorm(n, mu, sqrt(disp)),
                trans_func = identity)

ll_func <- mssm(
  fixed = y ~ x + Z, family = gaussian(), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 200L, what = "gradient",
    ftol_abs = 1e-6, ftol_abs_inner = 1e-6, which_ll_cp = "KD",
    aprx_eps = .1))

sta <- coef(glm_fit <- glm(y ~ x  + Z, gaussian(), dat))
lpa <- ll_func$Laplace(
  cfix = sta, disp = 1, F. = diag(.5, 2), Q = diag(1, 2), trace = 1L)
logLik(glm_fit)
lpa$logLik
lpa$dispersion


system.time(
  res <- adam(
    ll_func, cfix = lpa$cfix, disp = disp, F. = lpa$F., Q = lpa$Q,
    verbose = TRUE, n_it = 100L, lr = .01))

plot(res$logLik)
mean(res$logLik)
lpa$logLik

F.
res$F.
lpa$F.

Q
res$Q
lpa$Q

cfix
res$cfix
lpa$cfix

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., Q = res$Q, disp = 3, seed = NULL,
  what = "Hessian", N_part = 5000L)



#####
# Gaussian w/ log link
n_periods <- 400L
F. <- matrix(c(.5, 0, 0, .8), 2L)
Q <- matrix(c(.5^2, .1, .1, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(-1, .2, .5)
n_obs <- 100L

set.seed(78727277)

betas <- .get_beta(Q, Q0, F., n_periods)
matplot(betas, type = "l")
dat <- .get_dat(cfix, betas, sample_func = function(n, mu) rnorm(n, mu, sqrt(3)),
                trans_func = exp)

ll_func <- mssm(
  fixed = y ~ x + Z, family = gaussian("log"), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .01, ftol_abs = 1e-8, ftol_abs_inner = 1e-8))
sta <- coef(glm_fit <- glm(y ~ x  + Z, gaussian("log"), dat, start = cfix))

logLik(glm_fit)
disp <- summary(glm_fit)$dispersion

system.time(
  lpa <- ll_func$Laplace(
    F. = diag(.01, 2), Q = diag(1e-8, 2), cfix = sta, trace = 1L, disp = disp))
lpa
cov2cor(Q)

system.time(
  res <- adam(
    ll_func, cfix = lpa$cfix, disp = lpa$disp, F. = lpa$F., Q = lpa$Q,
    verbose = TRUE, n_it = 100L, lr = .01))
plot(res$logLik)

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., Q = res$Q, disp = 3, seed = NULL,
  what = "Hessian", N_part = 5000L)

he <- get_grad_n_obs_info(o)
res$cfix
he$grad
sqrt(diag(solve(-he$hess)))

plot(o)

#####
# Gaussian w/ inverse link
n_periods <- 100L
F. <- matrix(c(.5, 0, 0, .8), 2L)
Q <- matrix(c(.5^2, .1, .1, .7^2), 2L)
Q0 <- mssm:::.get_Q0(Q, F.)
cfix <- c(10, .2, .5)
n_obs <- 1000L
disp <- .5^2

set.seed(78727279)

betas <- .get_beta(Q, Q0, F., n_periods)
matplot(betas, type = "l")
dat <- .get_dat(cfix, betas,
                sample_func = function(n, mu) rnorm(n, mu, sqrt(disp)),
                trans_func = function(z) 1 / z)

ll_func <- mssm(
  fixed = y ~ x + Z, family = gaussian("inverse"), data = dat,
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 1000L, what = "gradient",
    which_ll_cp = "KD", aprx_eps = .01))
sta <- coef(glm_fit <- glm(y ~ x  + Z, gaussian("inverse"), dat, start = cfix))

logLik(glm_fit)
disp <- summary(glm_fit)$dispersion
logLik(ll_func$pf_filter(cfix = sta, disp = disp, F. = diag(.001, 2),
                         Q = diag(1e-8, 2), seed = NULL))

system.time(
  res <- adam(
    ll_func, F. = F., Q = Q, cfix = cfix, verbose = TRUE,
    n_it = 100L, lr = .01, disp = disp))

plot(res$logLik)

o <- ll_func$pf_filter(
  cfix = res$cfix, F. = res$F., Q = res$Q, disp = disp, seed = NULL,
  what = "Hessian", N_part = 5000L)

he <- get_grad_n_obs_info(o)
res$cfix
sqrt(diag(solve(-he$ddg)))
sqrt(diag(vcov(glm_fit)))
c(res$F., res$Q)
sqrt(diag(solve(-he$ddf)))

plot(o)
