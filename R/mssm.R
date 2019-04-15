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

  # get weights, offsets, and time indices
  weights <- if(missing(weights))
    rep(1., N) else
      # TODO: test
      eval(substitute(weights), data)
  offsets <- if(missing(offsets))
    rep(0, N) else
      # TODO: test
      eval(substitute(offsets), data)

  ti <- eval(substitute(ti), data)
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
    # TODO: check args
    if(missing(Q0))
      Q0 <- .get_Q0(Q, F.)
    if(missing(mu0))
      mu0 <- numeric(nrow(Q0))

    set.seed(seed) # TODO: check that results are reproducible
    out <- pf_filter(
      Y = y, cfix = cfix, ws = weights, offsets = offsets, disp = disp, X = X,
      Z = Z,
      time_indices_elems = time_indices_elems - 1L, # zero index
      time_indices_len = time_indices_len, F = F., Q = Q, Q0 = Q0,
      fam = fam, mu0 = mu0, n_threads = control$n_threads, nu = control$nu,
      covar_fac = control$covar_fac, ftol_rel = control$ftol_rel,
      N_part = control$N_part, what = control$what,
      which_sampler = control$which_sampler, which_ll_cp = control$which_ll_cp,
      trace)

    # TODO: test output
    structure(c(list(pf_output = out), output_list), class = "mssm")
  }
  formals(out_func)$seed <- control$seed

  # TODO: test output
  structure(
    c(list(pf_filter = out_func), output_list), class = "mssmFunc")
}

#' @export
mssm_control <- function(
  N_part = 1000L, n_threads = 1L, covar_fac = 1.2, ftol_rel = 1e-6, nu = 8.,
  what = "log_density", which_sampler = "mode_aprx", which_ll_cp = "no_aprx",
  seed = 1L){
  # TODO: check input arguments
  # TODO: test output
  list(
    N_part = N_part, n_threads = n_threads, covar_fac = covar_fac,
    ftol_rel = ftol_rel, what = what, which_sampler = which_sampler,
    which_ll_cp = which_ll_cp, nu = nu, seed = seed)
}

.get_Q0 <- function(Qmat, Fmat){
  eg  <- eigen(Fmat)
  las <- eg$values
  if(any(abs(las) >= 1))
    stop("Divergent series")
  U   <- eg$vectors
  T. <- solve(U, t(solve(U, Qmat)))
  Z   <- T. / (1 - tcrossprod(las))
  out <- tcrossprod(U %*% Z, U)
  if(is.complex(out)){
    if(all(abs(Im(out)) < .Machine$double.eps^(3/4)))
      return(Re(out))

    stop("Q_0 has imaginary part")
  }

  out
}

#' @method logLik mssm
#' @export
logLik.mssm <- function(object){
  stopifnot(inherits(object, "mssm"))
  ll <- sum(sapply(object$pf_output, function(x) mean(x$ws)))
  # TODO: set nobs and df
  # TODO: test output
  structure(ll, nobs = NA_integer_, df = NA_integer_, class = "logLik")
}

#' @method plot mssm
#' @export
plot.mssm <- function(x, y, qs = c(.05, .95), ...){
  stopifnot(inherits(x, "mssm"), qs[2] > qs[1], all(qs > 0, qs < 1))

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

  # TODO: test output
  invisible(list(means = filter_ests, lbs = lbs, ubs = ubs))
}

.get_time_index <- function(object){
  stopifnot(inherits(object, "mssm"))
  with(object, min(ti):max(ti))
}
