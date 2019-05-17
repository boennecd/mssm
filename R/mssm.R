#' @title Get Multivariate State Space Model Functions
#' @description
#' Returns an object with a function that can be used to run a particle filter,
#' a function to perform parameter estimation using a Laplace approximation,
#' and a function to perform smoothing of particle weights.
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
#' \item{smoother}{function to compute smoothing weights for an \code{mssm}
#' object returned by the \code{pf_filter} function. See \link{mssm-smoother}.}
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
#' @examples
#' if(require(Ecdat)){
#'   # load data and fit glm to get starting values
#'   . <- print
#'   data("Gasoline", package = "Ecdat")
#'   glm_fit <- glm(lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
#'                  Gamma("log"), Gasoline)
#'
#'   # get object to perform estimation
#'   library(mssm)
#'   ll_func <- mssm(
#'   fixed = formula(glm_fit), random = ~ 1, family = Gamma("log"),
#'     data = Gasoline, ti = year, control = mssm_control(
#'       N_part = 1000L, n_threads = 1L))
#'   .(ll_func)
#'
#'   # fit model with time-varying intercept with Laplace approximation
#'   disp <- summary(glm_fit)$dispersion
#'   laplace <- ll_func$Laplace(
#'     cfix = coef(glm_fit), disp = disp, F. = diag(.5, 1), Q = diag(1))
#'   .(laplace)
#'
#'   # compare w/ glm
#'   .(logLik(laplace))
#'   .(logLik(glm_fit))
#'   .(rbind(laplace = laplace$cfix, glm = coef(glm_fit)))
#'
#'   # run particle filter
#'   pf <- ll_func$pf_filter(
#'     cfix = laplace$cfix, disp = laplace$disp, F. = laplace$F., Q = laplace$Q)
#'   .(pf)
#'
#'   # compare approximate log-likelihoods
#'   .(logLik(pf))
#'   .(logLik(laplace))
#'
#'   # predicted values from filtering (does not appear random...)
#'   plot(pf)
#'
#'   # plot predicted values from smoothing distribution
#'   pf <- ll_func$smoother(pf)
#'   plot(pf, which_weights = "smooth")
#' }
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

    # set dimension names
    dimnames(F.) <- dimnames(Q) <- di$QF
    if(length(cfix) > 0)
      names(cfix) <- di$cfix[seq_along(cfix)]

    structure(c(
      list(pf_output = out), list(cfix = cfix, disp = disp, F. = F.,
                                  Q = Q, Q0 = Q0, mu0 = mu0, N_part = N_part),
      output_list), class = "mssm")
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
    if(length(out$cfix) > 0)
      names(out$cfix) <- di$cfix[seq_along(out$cfix)]

    structure(c(out, output_list), class = "mssmLaplace")
  }

  # assign function to perform smoothing
  smoother <- function(object){
    stopifnot(inherits(object, "mssm"))

    out <- smoother_cpp(
      Y = y, cfix = object$cfix, ws = weights, offsets = offsets,
      disp = object$disp, X = X, Z = Z,
      time_indices_elems = time_indices_elems - 1L, # zero index
      time_indices_len = time_indices_len, F = object$F., Q = object$Q,
      Q0 = object$Q0, fam = fam, mu0 = object$mu0,
      n_threads = control$n_threads, nu = control$nu,
      covar_fac = control$covar_fac, ftol_rel = control$ftol_rel,
      N_part = object$N_part, what = "log_density", trace = 0L,
      KD_N_max = control$KD_N_max, aprx_eps = control$aprx_eps,
      which_ll_cp = control$which_ll_cp, pf_output = object$pf_output)

    out <- mapply(
      function(x, y) c(y, list(ws_normalized_smooth = x)),
      x = out, y = object$pf_output, SIMPLIFY = FALSE)

    object$pf_output <- out
    object
  }

  # set defaults
  idx_set <- c("seed", "what", "N_part")
  formals(out_func)[idx_set] <- control[idx_set]

  structure(
    c(list(pf_filter = out_func, Laplace = Laplace, smoother = smoother),
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
#' @param cfix values for for coefficient for the fixed effects.
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
#' If the Hessian is requested then the \eqn{\tilde \beta_n^{(i)}}s
#' in Poyiadjis et al. (2011) are returned after the gradient elements. These
#' can be used to approximate the observed information matrix. That is,
#' using that the approximation of the observed information matrix is
#'
#' \deqn{\tilde S_n\tilde S_n^\top - \sum_{i = 1}^n
#'    \tilde W_n^{(i)}(\tilde\alpha_n^{(i)}\tilde\alpha_n^{(i)\top} +
#'    \tilde \beta_n^{(i)}),
#'    \qquad \tilde S_n = \sum_{i=1}^n \tilde W_n^{(i)}\tilde\alpha_n^{(i)}}
#'
#' as in Poyiadjis et al. (2011). See the README for an example.
#'
#' @references
#' Poyiadjis, G., Doucet, A. and Singh, S. S. (2011) Particle Approximations of
#' the Score and Observed Information Matrix in State Space Models with
#' Application to Parameter Estimation. \emph{Biometrika}, \strong{98(1)},
#' 65--80.
#'
#' @seealso
#' \code{\link{mssm}}.
#'
#' @examples
#' if(require(Ecdat)){
#'   # load data and get object to perform particle filtering
#'   data("Gasoline", package = "Ecdat")
#'
#'   library(mssm)
#'   ll_func <- mssm(
#'     fixed = lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
#'     random = ~ 1, family = Gamma("log"), data = Gasoline, ti = year,
#'     control = mssm_control(N_part = 1000L, n_threads = 1L))
#'
#'   # run particle filter
#'   cfix <- c(0.612, -0.015, 0.214, 0.048, -0.013, -0.016, -0.022, 0.047,
#'             -0.046, 0.007, -0.001, 0.008, -0.117, 0.075, 0.048, -0.054, 0.017,
#'             0.228, 0.077, -0.056, -0.139)
#'   pf <- ll_func$pf_filter(
#'     cfix = cfix, Q = as.matrix(2.163e-05), F. = as.matrix(0.9792),
#'     disp = 0.000291)
#'   print(pf)
#' }
NULL

#' @title Parameter Estimation with Laplace Approximation for Multivariate
#' State Space Model
#' @name mssm-Laplace
#' @description
#' Function returned from \code{\link{mssm}} which can be used to perform
#' parameter estimation with a Laplace approximation.
#'
#' @param cfix starting values for coefficient for the fixed effects.
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
#' \item{disp}{estimated dispersion parameter.}
#'
#' Remaining elements are the same as returned by \code{\link{mssm}}.
#'
#' @seealso
#' \code{\link{mssm}}.
#'
#' @examples
#' if(require(Ecdat)){
#'   # load data and fit glm to get starting values
#'   data("Gasoline", package = "Ecdat")
#'   glm_fit <- glm(lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
#'                  Gamma("log"), Gasoline)
#'
#'   # get object to perform estimation
#'   library(mssm)
#'   ll_func <- mssm(
#'     fixed = formula(glm_fit), random = ~ 1, family = Gamma("log"),
#'     data = Gasoline, ti = year, control = mssm_control(
#'       N_part = 1000L, n_threads = 1L))
#'
#'   # fit model with time-varying intercept with Laplace approximation
#'   disp <- summary(glm_fit)$dispersion
#'   laplace <- ll_func$Laplace(
#'     cfix = coef(glm_fit), disp = disp, F. = diag(.5, 1), Q = diag(1))
#'   print(laplace)
#' }
NULL

#' @title Computes Smoothed Particle Weights for Multivariate State Space
#' Model
#' @name mssm-smoother
#' @description
#' Computes smoothed weights using the backward smoothing formula for a
#' \code{mssm} object. The k-d dual tree approximation is also used if it used
#' for the \code{mssm} object.
#'
#' @param object an object of class \code{mssm} from \link{mssm-pf}.
#'
#' @return
#' Same as \link{mssm-pf} but where the \code{pf_output}'s list elements
#' has an additional element called \code{ws_normalized_smooth}. This
#' contains the normalized log smoothing weights.
#'
#' @seealso
#' \code{\link{mssm}}.
#'
#' @examples
#' if(require(Ecdat)){
#'   # load data and get object to perform particle filtering
#'   data("Gasoline", package = "Ecdat")
#'
#'   library(mssm)
#'   ll_func <- mssm(
#'     fixed = lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
#'     random = ~ 1, family = Gamma("log"), data = Gasoline, ti = year,
#'     control = mssm_control(N_part = 1000L, n_threads = 1L))
#'
#'   # run particle filter
#'   cfix <- c(0.612, -0.015, 0.214, 0.048, -0.013, -0.016, -0.022, 0.047,
#'             -0.046, 0.007, -0.001, 0.008, -0.117, 0.075, 0.048, -0.054, 0.017,
#'             0.228, 0.077, -0.056, -0.139)
#'   pf <- ll_func$pf_filter(
#'     cfix = cfix, Q = as.matrix(2.163e-05), F. = as.matrix(0.9792),
#'     disp = 0.000291)
#'
#'   print(is.null(pf$pf_output[[1L]]$ws_normalized_smooth))
#'   pf <- ll_func$smoother(pf)
#'   print(is.null(pf$pf_output[[1L]]$ws_normalized_smooth))
#' }
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
#' @param nu degrees of freedom to use for the multivariate
#' \eqn{t}-distribution that is used as the proposal distribution. A
#' multivariate normal distribution is used if \code{nu <= 2}.
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
#' particles to include in each leaf of the two k-d trees if the dual k-d trees
#' method is used.
#' @param aprx_eps positive numeric scalar with the maximum error if the
#' dual k-d tree method is used.
#' @param ftol_abs,ftol_abs_inner,la_ftol_rel,la_ftol_rel_inner,maxeval,maxeval_inner
#' scalars passed to \code{nlopt} when estimating parameters with a Laplace
#' approximation. The \code{_inner} denotes the values passed in the inner
#' mode estimation. The mode estimation is done with a custom Newton–Raphson
#' method
#'
#' @seealso
#' \code{\link{mssm}}.
#'
#' See the README of the package for details of the dual k-d tree method
#' at \url{https://github.com/boennecd/mssm}.
#'
#' @examples
#' library(mssm)
#' str(mssm_control())
#' str(mssm_control(N_part = 2000L))
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
#' @examples
#' if(require(Ecdat)){
#'  # load data and fit glm to get starting values
#'  data("Gasoline", package = "Ecdat")
#'  glm_fit <- glm(lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
#'                 Gamma("log"), Gasoline)
#'
#'  # get object to perform estimation
#'  library(mssm)
#'  ll_func <- mssm(
#'    fixed = formula(glm_fit), random = ~ 1, family = Gamma("log"),
#'    data = Gasoline, ti = year, control = mssm_control(
#'      N_part = 1000L, n_threads = 1L))
#'
#'  # fit model with time-varying intercept with Laplace approximation
#'  disp <- summary(glm_fit)$dispersion
#'  laplace <- ll_func$Laplace(
#'    cfix = coef(glm_fit), disp = disp, F. = diag(.5, 1), Q = diag(1))
#'
#'  # run particle filter
#'  pf <- ll_func$pf_filter(
#'    cfix = laplace$cfix, disp = laplace$disp, F. = laplace$F., Q = laplace$Q)
#'
#'  # compare approximate log-likelihoods
#'  print(logLik(pf))
#'  print(logLik(laplace))
#'}
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
  stopifnot(inherits(object, c("mssm", "mssmLaplace", "mssmFunc")))
  # assumes that all parameters are free
  n_rng <- nrow(object$Z)
  n_fix <- nrow(object$X)

  out <- n_fix + n_rng * n_rng + (n_rng * (n_rng + 1L)) / 2L
  if(any(sapply(c("^Gamma", "^gaussian"), grepl, x = object$family)))
    out <- out + 1L
  out
}

.get_nobs <- function(object){
  stopifnot(inherits(object, c("mssm", "mssmLaplace", "mssmFunc")))
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

.get_time_index <- function(object){
  stopifnot(inherits(object, "mssm"))
  with(object, min(ti):max(ti))
}
