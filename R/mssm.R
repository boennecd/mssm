#' @title Get Multivariate State Space Model Function
#' @description
#' Returns an object which can be used to run a particle filter.
#'
#' @param fixed \code{\link{formula}} with outcome variable on the left hand
#' side and covariates with fixed effects on the right hand side.
#' @param family family for the observed outcome given the state variables
#' and covariates.
#' @param data \code{\link{data.frame}} or environment containing the variables
#' in \code{fixed} and \code{random}.
#' @param random \code{\link{formula}} for covariates with a random effect.
#' Left hand side is ignored.
#' @param weights optional prior weights.
#' @param offsets optional a priori known component in the linear predictor.
#' @param ti integer vector with time indices matching with each observation of
#' \code{fixed} and \code{random}.
#' @param control list with arguments passed to \code{\link{mssm_control}}.
#'
#' @return
#' An object of class \code{mssmFunc} with the following elements
#' \item{pf_filter}{function to perform particle filtering. See
#' \link{mssm-pf}.}
#' \item{Laplace}{function to perform parameter estimation with a Laplace
#' approximation. See \link{mssm-Laplace}.}
#' \item{terms_fixed}{\code{\link{terms.object}} for the covariates with
#' fixed effects.}
#' \item{terms_random}{\code{\link{terms.object}} for the covariates with
#' random effects.}
#' \item{y}{vector with outcomes.}
#' \item{X}{covariates with fixed effects.}
#' \item{Z}{covariates with random effects.}
#' \item{ti}{time indices for each observation.}
#' \item{weights}{prior weights for each observation.}
#' \item{offsets}{a priori known component in the linear predictor for
#' each observation.}
#' \item{call}{the matched call.}
#' \item{family}{character describing the conditional distribution of the
#' outcomes.}
#'
#' @seealso
#' The README of the package contains examples of how to use this function.
#' See \url{https://github.com/boennecd/mssm}.
#'
#' @importFrom stats model.frame model.matrix model.response terms
#' @export
mssm <- function(
  fixed, family, data, random, weights, offsets, ti, control = mssm_control())
{
  # TODO: do more checks
  stopifnot(
    inherits(fixed, "formula"), inherits(random, "formula"),
    is.data.frame(data) || is.environment(data))

  # get design matrices and outcome
  mf_X <- model.frame(fixed , data)
  mf_Z <- model.frame(random, data)
  N <- nrow(mf_X)
  stopifnot(N == nrow(mf_Z))

  y <- model.response(mf_X)
  stopifnot(length(y) == N)
  X <- t(model.matrix(terms(mf_X), mf_X))
  Z <- t(model.matrix(terms(mf_Z), mf_Z))
  stopifnot(ncol(X) == ncol(Z))

  # get weights, offsets, and time indices
  weights <- if(missing(weights))
    rep(1., N)
  else
    # TODO: test
    eval(substitute(weights), data)

  offsets <- if(missing(offsets))
    rep(0, N) else
      # TODO: test
      eval(substitute(offsets), data)
  ti <- eval(substitute(ti), data)

  stopifnot(
    is.numeric(weights), length(weights) == ncol(X),
    is.numeric(offsets), length(offsets) == ncol(X),
    is.integer(ti),      length(ti)      == ncol(X))

  # TODO: maybe do this in a smart way
  time_indices <- range(ti)
  time_indices = lapply(
    time_indices[1]:time_indices[2], function(t.)
      which(ti == t.))
  time_indices_elems <- unlist(time_indices)
  time_indices_len <- sapply(time_indices, length)

  # get control object
  stopifnot(is.list(control))
  control <- do.call(mssm_control, control)

  # setup other needed arguments
  stopifnot(inherits(family, "family"))
  fam <- paste0(family$family, "_", family$link)

  # make output list
  output_list <- list(
    terms_fixed = terms(mf_X), terms_random = terms(mf_Z),
    y = y, X = X, Z = Z, ti = ti, weights = weights, offsets = offsets,
    call = match.call(), control = control, family = fam)

  # assign function to validate input
  chech_input <- function(cfix, disp, F., Q, Q0, mu0, trace = 0L, seed, what,
                          N_part){
    p <- nrow(Z)
    stopifnot(
      is.numeric(cfix), length(cfix) == nrow(X),
      is.numeric(disp),
      is.numeric(F.), is.matrix(F.), nrow(F.) == p, ncol(F.) == p,
      is.numeric(Q ), is.matrix(Q ), nrow(Q ) == p, ncol(Q ) == p,
      is.numeric(Q0), is.matrix(Q0), nrow(Q0) == p, ncol(Q0) == p,
      is.numeric(mu0), length(mu0) == p,
      is.integer(trace),
      is.null(seed) || is.numeric(seed))
    .is_valid_N_part(N_part)
    .is_valid_what(what)
  }

  # assign function to run the particle filter
  out_func <- function(cfix, disp, F., Q, Q0, mu0, trace = 0L, seed, what,
                       N_part){
    p <- nrow(Z)
    if(missing(Q0))
      Q0 <- .get_Q0(Q, F.)
    if(missing(mu0))
      mu0 <- numeric(nrow(Q0))

    chech_input(cfix, disp, F., Q, Q0, mu0, trace, seed, what, N_part)

    if(!is.null(seed))
      set.seed(seed)
    out <- pf_filter(
      Y = y, cfix = cfix, ws = weights, offsets = offsets, disp = disp, X = X,
      Z = Z,
      time_indices_elems = time_indices_elems - 1L, # zero index
      time_indices_len = time_indices_len, F = F., Q = Q, Q0 = Q0,
      fam = fam, mu0 = mu0, n_threads = control$n_threads, nu = control$nu,
      covar_fac = control$covar_fac, ftol_rel = control$ftol_rel,
      N_part = N_part, what = what,
      which_sampler = control$which_sampler, which_ll_cp = control$which_ll_cp,
      trace, KD_N_max = control$KD_N_max, aprx_eps = control$aprx_eps)

    # set dimension names
    di <- .get_dimnames(output_list)
    if(what == "gradient")
      rownames(out[[length(out)]]$stats) <- di$grad
    else if(what == "Hessian")
      rownames(out[[length(out)]]$stats) <- c(
        di$grad, c(outer(di$grad, di$grad, paste, sep = "*")))

    structure(c(list(pf_output = out), output_list), class = "mssm")
  }

  # assign function to use Laplace approximation to estimate parameters
  Laplace <- function(cfix, disp, F., Q, Q0, mu0, trace = 0L){
    p <- nrow(Z)
    if(missing(Q0))
      Q0 <- .get_Q0(Q, F.)
    if(missing(mu0))
      mu0 <- numeric(nrow(Q0))

    what <- "log_density"
    N_part <- 2L
    seed <- NULL

    chech_input(cfix, disp, F., Q, Q0, mu0, trace, seed, what, N_part)

    out <- run_Laplace_aprx(
      Y = y, cfix = cfix, ws = weights, offsets = offsets, disp = disp, X = X,
      Z = Z,
      time_indices_elems = time_indices_elems - 1L, # zero index
      time_indices_len = time_indices_len, F = F., Q = Q, Q0 = Q0,
      fam = fam, mu0 = mu0, n_threads = control$n_threads, nu = control$nu,
      covar_fac = control$covar_fac, ftol_rel = control$ftol_rel,
      N_part = N_part, what = what, trace, KD_N_max = control$KD_N_max,
      aprx_eps = control$aprx_eps,

      ftol_abs = control$ftol_abs, la_ftol_rel = control$la_ftol_rel,
      ftol_abs_inner = control$ftol_abs_inner,
      la_ftol_rel_inner = control$la_ftol_rel_inner,
      maxeval = control$maxeval, maxeval_inner = control$maxeval_inner)
    out$cfix <- drop(out$cfix)

    # set dimension names
    di <- .get_dimnames(output_list)
    dimnames(out$F.) <- dimnames(out$Q) <- di$QF
    names(out$cfix) <- di$cfix[seq_along(out$cfix)]

    structure(c(out, output_list), class = "mssmLaplace")
  }

  # set defaults
  idx_set <- c("seed", "what", "N_part")
  formals(out_func)[idx_set] <- control[idx_set]

  structure(
    c(list(pf_filter = out_func, Laplace = Laplace),
      output_list), class = "mssmFunc")
}

.is.num.le1 <- function(x)
  is.numeric(x) && length(x) == 1L

.is.int.le1 <- function(x)
  is.integer(x) && length(x) == 1L

# returns list with dimension names for various objects
.get_dimnames <- function(output_list){
  fix_names <- rownames(output_list$X)
  if(any(sapply(c("^Gamma", "^gaussian"), grepl, x = output_list$family)))
    fix_names <- c(fix_names, "dispersion")

  rng_names <- rownames(output_list$Z)
  ma_ele <- outer(rng_names, rng_names, paste, sep = ".")
  grad <- c(
    paste0(fix_names),
    paste0("F:", c(ma_ele)),
    paste0("Q:", ma_ele[lower.tri(ma_ele, diag = TRUE)]))

  list(
    cfix = fix_names, QF = list(rng_names, rng_names), grad = grad)
}

#' @title Particle Filter Function for Multivariate State Space Model
#' @name mssm-pf
#' @description
#' Function returned from \code{\link{mssm}} which can be used to perform
#' particle filtering given values for the parameters in the model.
#'
#' @param cfix coefficient for fixed effects.
#' @param disp additional parameters for the family (e.g., a dispersion
#' parameter).
#' @param F. matrix in the transition density of the state vector.
#' @param Q covariance matrix in the transition density of the state vector.
#' @param Q0 optional covariance matrix at the first time point. Default is
#' the covariance matrix in the time invariant distribution.
#' @param mu0 optional mean at the first time point. Default is
#' the zero vector.
#' @param trace integer controlling whether information should be printed
#' during particle filtering. Zero yields no information.
#' @param seed integer to pass to \code{\link{set.seed}}. The seed is not set
#' if the argument is \code{NULL}.
#' @param what,N_part same as in \code{\link{mssm_control}}.
#'
#' @return
#' An object of class \code{mssm} with the following elements
#' \item{pf_output}{A list with an element for each time period. Each element
#' is a list with
#' \code{particles}: the sampled particles,
#' \code{stats}: additional object that is requested to be computed with
#' each particle,
#' \code{ws:} unnormalized log particle weights for the filtering distribution,
#' and
#' \code{ws_normalized:} normalized log particle weights for the filtering
#' distribution.}
#'
#' Remaining elements are the same as returned by \code{\link{mssm}}.
#'
#' If gradient approximation is requested then the first elements of
#' \code{stats} are w.r.t. the fixed coefficients, the next elements are
#' w.r.t. the matrix in the map from the previous state vector to the mean
#' of the next, and the last element is w.r.t. the covariance matrix.
#' Only the lower triangular matrix is kept for the covariance
#' matrix. See the examples in the README at
#' \url{https://github.com/boennecd/mssm}. There will be an additional
#' element for the dispersion parameter if the family has a dispersion
#' parameter.
#'
#' If an approximation of the observed information matrix is requested then
#' it's components are given after the gradient elements in the
#' \code{stats} object.
#'
#' @seealso
#' \code{\link{mssm}}.
NULL

#' @title Parameter Estimation with Laplace Approximation for Multivariate
#' State Space Model
#' @name mssm-Laplace
#' @description
#' Function returned from \code{\link{mssm}} which can be used to perform
#' parameter estimation with a Laplace approximation.
#'
#' @param cfix starting values for coefficient for fixed effects.
#' @param disp starting value for additional parameters for the family
#' (e.g., a dispersion parameter).
#' @param F. starting values for matrix in the transition density of the state
#' vector.
#' @param Q starting values for covariance matrix in the transition density
#' of the state vector.
#' @param Q0 un-used.
#' @param mu0 un-used.
#' @param trace integer controlling whether information should be printed
#' during parameter estimation. Zero yields no information.
#'
#' @return
#' An object of class \code{mssmLaplace} with the following elements
#' \item{F.}{estimate of \code{F.}.}
#' \item{Q}{estimate of \code{Q}.}
#' \item{cfix}{estimate of \code{cfix}.}
#' \item{logLik}{approximate log-likelihood at estimates.}
#' \item{n_it}{number of Laplace approximations.}
#' \item{code}{returned code from \code{nlopt}.}
#' \item{dispersion}{estimated dispersion parameter \code{nlopt}.}
#'
#' Remaining elements are the same as returned by \code{\link{mssm}}.
#'
#' @seealso
#' \code{\link{mssm}}.
NULL

#' @title Auxiliary for Controlling Multivariate State Space Model Fitting
#' @description
#' Auxiliary function for \code{\link{mssm}}.
#'
#' @param N_part integer greater than zero for the number of particles to use.
#' @param n_threads integer greater than zero for the number of threads to use.
#' @param covar_fac positive numeric scalar used to scale the covariance
#' matrix in the proposal distribution.
#' @param ftol_rel positive numeric scalar with convergence threshold passed
#' to \code{\link{nloptr}} if the mode approximation method is used for the
#' proposal distribution.
#' @param nu degrees of freedom to use for the multivariate \eqn{t}-distribution
#' used as the proposal distribution. A multivariate normal distribution is
#' used if \code{nu <= 2}.
#' @param what character indicating what to approximate. \code{"log_density"}
#' implies only the log-likelihood. \code{"gradient"} also yields a gradient
#' approximation. \code{"Hessian"} also yields an approximation of the
#' observed information matrix.
#' @param which_sampler character indicating what type of proposal
#' distribution to use. \code{"mode_aprx"} yields a Taylor approximation at
#' the mode. \code{"bootstrap"} yields a proposal distribution similar to the
#' common bootstrap filter.
#' @param which_ll_cp character indicating what type of computation should be
#' performed in each iteration of the particle filter. \code{"no_aprx"} yields
#' no approximation. \code{"KD"} yields an approximation using a dual k-d tree
#' method.
#' @param seed integer with seed to pass to \code{\link{set.seed}}.
#' @param KD_N_max integer greater than zero with the maximum number of
#' particles to be in each leaf of the two k-d trees if the dual k-d trees
#' method is used.
#' @param aprx_eps positive numeric scalar with the maximum error if the
#' dual k-d tree method is used.
#' @param ftol_abs,ftol_abs_inner,la_ftol_rel,la_ftol_rel_inner,maxeval,maxeval_inner
#' scalars passed to \code{nlopt} when in estimation with Laplace approximation.
#' The \code{_inner} denotes the value passed in the mode estimation.
#'
#'
#' @seealso
#' See README of the package for details of the dual k-d tree method
#' at \url{https://github.com/boennecd/mssm}.
#'
#' @export
mssm_control <- function(
  N_part = 1000L, n_threads = 1L, covar_fac = 1.2, ftol_rel = 1e-6, nu = 8.,
  what = "log_density", which_sampler = "mode_aprx", which_ll_cp = "no_aprx",
  seed = 1L, KD_N_max = 10L, aprx_eps = 1e-3, ftol_abs = 1e-4,
  ftol_abs_inner = 1e-4, la_ftol_rel = -1., la_ftol_rel_inner = -1.,
  maxeval = 10000L, maxeval_inner = 10000L){
  stopifnot(
    .is.num.le1(n_threads), n_threads > 0L,
    .is.num.le1(covar_fac), covar_fac > 0.,
    .is.num.le1(ftol_rel), ftol_rel > 0.,
    .is.num.le1(nu), nu > 2. || nu == -1.,

    is.character(which_sampler), length(which_sampler) == 1L,
    which_sampler %in% c("mode_aprx", "bootstrap"),

    is.character(which_ll_cp), length(which_ll_cp) == 1L,
    which_ll_cp %in% c("no_aprx", "KD"),

    is.numeric(seed),
    .is.int.le1(KD_N_max), KD_N_max > 1L,
    .is.num.le1(aprx_eps), aprx_eps > 0.,

    .is.num.le1(ftol_abs), .is.num.le1(la_ftol_rel),
    ftol_abs > 0. || la_ftol_rel > 0.,

    .is.num.le1(ftol_abs_inner), .is.num.le1(la_ftol_rel_inner),
    ftol_abs_inner > 0. || la_ftol_rel_inner > 0.,

    .is.int.le1(maxeval), maxeval > 0L,
    .is.int.le1(maxeval_inner), maxeval_inner > 0L)
  .is_valid_N_part(N_part)
  .is_valid_what(what)

  list(
    N_part = N_part, n_threads = n_threads, covar_fac = covar_fac,
    ftol_rel = ftol_rel, what = what, which_sampler = which_sampler,
    which_ll_cp = which_ll_cp, nu = nu, seed = seed, KD_N_max = KD_N_max,
    aprx_eps = aprx_eps, ftol_abs = ftol_abs, la_ftol_rel = la_ftol_rel,
    ftol_abs_inner = ftol_abs_inner, la_ftol_rel_inner = la_ftol_rel_inner,
    maxeval = maxeval, maxeval_inner = maxeval_inner)
}

.is_valid_N_part <- function(N_part)
  stopifnot(is.integer(N_part), length(N_part) == 1L, N_part > 0L)

.is_valid_what <- function(what)
  stopifnot(
    is.character(what), length(what) == 1L,
    what %in% c("log_density", "gradient", "Hessian"))

#' @title Approximate Log-likelihood for a mssm Object
#' @description
#' Function to extract the log-likelihood from a \code{mssm} or
#' \code{mssmLaplace} object.
#'
#' @param object an object of class \code{mssm} or \code{mssmLaplace}.
#' @param ... un-used.
#'
#' @return
#' A \code{logLik} object. The \code{log_lik_terms} attribute contains
#' the log-likelihood contributions from each time point.
#'
#' The degrees of freedom assumes that all parameters are free. The number of
#' observations may be invalid for some models (e.g., discrete survival
#' analysis).
#'
#' @method logLik mssm
#' @export
logLik.mssm <- function(object, ...){
  stopifnot(inherits(object, "mssm"))
  ll <- sum(log_lik_terms <- sapply(object$pf_output, function(x){
    x <- x$ws
    ma <- max(x)
    log(sum(exp(x - ma))) + ma - log(length(x))
  }))
  df <- .get_df(object)
  nobs <- .get_nobs(object)
  # TODO: test output
  structure(ll, nobs = nobs, df = df, class = "logLik",
            log_lik_terms = log_lik_terms)
}

.get_df <- function(object){
  stopifnot(inherits(object, c("mssm", "mssmLaplace")))
  # assumes that all parameters are free
  n_rng <- nrow(object$Z)
  n_fix <- nrow(object$X)

  out <- n_fix + n_rng * n_rng + (n_rng * (n_rng + 1L)) / 2L
  if(any(sapply(c("^Gamma", "^gaussian"), grepl, x = object$family)))
    out <- out + 1L
  out
}

.get_nobs <- function(object){
  stopifnot(inherits(object, c("mssm", "mssmLaplace")))
  ncol(object$X)
}

#' @rdname logLik.mssm
#' @method logLik mssmLaplace
#' @export
logLik.mssmLaplace <- function(object, ...){
  stopifnot(inherits(object, "mssmLaplace"))
  df <- .get_df(object)
  nobs <- .get_nobs(object)
  # TODO: test output
  structure(object$logLik, nobs = nobs, df = df,
            class = "logLik")
}

#' @title Plot Predicted State Variables for mssm Object.
#' @description
#' Plots the predicted mean and pointwise prediction interval of the state
#' variables for the filtering distribution.
#'
#' @param x an object of class \code{mssm}.
#' @param y un-used.
#' @param qs two-dimensional numeric vector with bounds of the prediction
#' interval.
#' @param do_plot \code{TRUE} to create a plot with the mean and quantiles.
#' @param ... un-used.
#'
#' @return
#' List with means and quantiles.
#'
#' @importFrom graphics plot lines par
#' @method plot mssm
#' @export
plot.mssm <- function(x, y, qs = c(.05, .95), do_plot = TRUE, ...){
  stopifnot(inherits(x, "mssm"), qs[2] > qs[1], all(qs > 0, qs < 1),
            is.logical(do_plot))

  particles <- lapply(x$pf_output, "[[", "particles")
  ws <- lapply(x$pf_output, "[[", "ws_normalized")

  # get means
  filter_ests <- mapply(function(ws, ps){
    colSums(t(ps) * drop(exp(ws)))
  }, ws = ws, ps = particles)
  if(!is.matrix(filter_ests))
    filter_ests <- t(filter_ests)

  # get quantiles
  quants <- mapply(function(ws, ps){
    out <- apply(ps, 1, function(x){
      ord <- order(x)
      ws <- exp(ws[ord])
      x <- x[ord]

      ws <- cumsum(ws)
      x[c(min(which(ws > qs[1])), min(which(ws > qs[2])))]
    })

    apply(out, 1L, list)
  }, ws = ws, ps = particles, SIMPLIFY = FALSE)
  lbs <- t(do.call(rbind, sapply(quants, "[[", 1L)))
  ubs <- t(do.call(rbind, sapply(quants, "[[", 2L)))

  # plot
  idx_time <- .get_time_index(x)
  colnames(lbs)<- colnames(ubs) <- colnames(filter_ests) <- idx_time
  rownames(lbs)<- rownames(ubs) <- rownames(filter_ests) <- rownames(x$Z)
  if(do_plot){
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    par(mar = c(5, 4, 1, 1))
    for(i in 1:nrow(filter_ests)){
      plot(idx_time, filter_ests[i, ], ylab = rownames(filter_ests)[i],
           xlab = "time", type = "l", ylim = range(lbs[i, ], ubs[i, ],
                                                   filter_ests[i, ]))
      lines(idx_time, lbs[i, ], lty = 2)
      lines(idx_time, ubs[i, ], lty = 2)
    }
  }

  # TODO: test output
  invisible(list(means = filter_ests, lbs = lbs, ubs = ubs))
}

.get_time_index <- function(object){
  stopifnot(inherits(object, "mssm"))
  with(object, min(ti):max(ti))
}
