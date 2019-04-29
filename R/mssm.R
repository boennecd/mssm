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

  # assign function to compute the requested objects
  out_func <- function(cfix, disp, F., Q, Q0, mu0, trace = 0L, seed){
    p <- nrow(Z)
    stopifnot(
      is.numeric(cfix), length(cfix) == nrow(X),
      is.numeric(disp),
      is.numeric(F.), is.matrix(F.), nrow(F.) == p, ncol(F.) == p,
      is.numeric(Q ), is.matrix(Q ), nrow(Q ) == p, ncol(Q ) == p,
      is.integer(trace),
      is.null(seed) || is.numeric(seed))

    if(missing(Q0))
      Q0 <- .get_Q0(Q, F.)
    if(missing(mu0))
      mu0 <- numeric(nrow(Q0))

    stopifnot(
      is.numeric(Q0 ), is.matrix(Q0), nrow(Q0) == p, ncol(Q0) == p,
      is.numeric(mu0), length(mu0) == p)

    if(!is.null(seed))
      set.seed(seed)
    out <- pf_filter(
      Y = y, cfix = cfix, ws = weights, offsets = offsets, disp = disp, X = X,
      Z = Z,
      time_indices_elems = time_indices_elems - 1L, # zero index
      time_indices_len = time_indices_len, F = F., Q = Q, Q0 = Q0,
      fam = fam, mu0 = mu0, n_threads = control$n_threads, nu = control$nu,
      covar_fac = control$covar_fac, ftol_rel = control$ftol_rel,
      N_part = control$N_part, what = control$what,
      which_sampler = control$which_sampler, which_ll_cp = control$which_ll_cp,
      trace, KD_N_max = control$KD_N_max, aprx_eps = control$aprx_eps)

    structure(c(list(pf_output = out), output_list), class = "mssm")
  }
  formals(out_func)$seed <- control$seed

  structure(
    c(list(pf_filter = out_func), output_list), class = "mssmFunc")
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
#' \item{remaining elements}{same as return by \code{\link{mssm}}.}
#'
#' If gradient approximation is requested then the first elements of
#' \code{stats} are w.r.t. the fixed coefficients, the next elements are
#' w.r.t. the matrix in the map from the previous state vector to the mean
#' of the next, and the last element is w.r.t. the covariance matrix.
#' Only the lower triangular matrix is kept for the covariance
#' matrix. See the examples in the README at
#' \url{https://github.com/boennecd/mssm}.
#'
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
#' approximation.
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
#'
#' @seealso
#' See README of the package for details of the dual k-d tree method
#' at \url{https://github.com/boennecd/mssm}.
#'
#' @export
mssm_control <- function(
  N_part = 1000L, n_threads = 1L, covar_fac = 1.2, ftol_rel = 1e-6, nu = 8.,
  what = "log_density", which_sampler = "mode_aprx", which_ll_cp = "no_aprx",
  seed = 1L, KD_N_max = 10L, aprx_eps = 1e-3){
  stopifnot(
    is.integer(N_part), length(N_part) == 1L, N_part > 0L,
    is.integer(n_threads), length(n_threads) == 1L, n_threads > 0L,
    is.numeric(covar_fac), length(covar_fac) == 1L, covar_fac > 0.,
    is.numeric(ftol_rel), length(ftol_rel) == 1L, ftol_rel > 0.,
    is.numeric(nu), length(nu) == 1L, nu > 2. || nu == -1.,

    is.character(what), length(what) == 1L,
    what %in% c("log_density", "gradient"),

    is.character(which_sampler), length(which_sampler) == 1L,
    which_sampler %in% c("mode_aprx", "bootstrap"),

    is.character(which_ll_cp), length(which_ll_cp) == 1L,
    which_ll_cp %in% c("no_aprx", "KD"),

    is.numeric(seed),
    is.integer(KD_N_max), length(KD_N_max) == 1L, KD_N_max > 1L,
    is.numeric(aprx_eps), length(aprx_eps) == 1L, aprx_eps > 0.)

  list(
    N_part = N_part, n_threads = n_threads, covar_fac = covar_fac,
    ftol_rel = ftol_rel, what = what, which_sampler = which_sampler,
    which_ll_cp = which_ll_cp, nu = nu, seed = seed, KD_N_max = KD_N_max,
    aprx_eps = aprx_eps)
}

.get_Q0 <- function(Qmat, Fmat){
  stopifnot(is.matrix(Qmat), is.numeric(Qmat),
            is.matrix(Fmat), is.numeric(Fmat),
            all(dim(Qmat) == dim(Fmat)))

  eg  <- eigen(Fmat)
  las <- eg$values
  if(any(abs(las) >= 1))
    stop("Divergent series")
  U   <- eg$vectors
  T. <- solve(U, t(solve(U, Qmat)))
  Z   <- T. / (1 - tcrossprod(las))
  out <- tcrossprod(U %*% Z, U)
  if(is.complex(out)){
    if(all(abs(Im(out)) < .Machine$double.eps^(1/2)))
      return(Re(out))

    stop("Q_0 has imaginary part")
  }

  out
}

#' @title Approximate Log-likelihood for a mssm Object
#' @description
#' Function to extract the log-likelihood from a \code{mssm} object.
#'
#' @param object an object of class \code{mssm}.
#' @param ... un-used.
#'
#' @return
#' A \code{logLik} object. The \code{log_lik_terms} attribute contains
#' the log-likelihood contributions from each time point.
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
  # TODO: set nobs and df
  # TODO: test output
  structure(ll, nobs = NA_integer_, df = NA_integer_, class = "logLik",
            log_lik_terms = log_lik_terms)
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
