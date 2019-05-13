#' @title Plot Predicted State Variables for mssm Object.
#' @description
#' Plots the predicted mean and pointwise prediction interval of the state
#' variables for the filtering distribution or smoothing distribution.
#'
#' @param x an object of class \code{mssm}.
#' @param y un-used.
#' @param qs two-dimensional numeric vector with bounds of the prediction
#' interval.
#' @param do_plot \code{TRUE} to create a plot with the mean and quantiles.
#' @param which_weights character of which weights to use. Either
#' \code{"filter"} for filter weights or \code{"smooth"} for smooth for
#' smooth weights. The latter requires that \code{smooth} element has
#' been used.
#' @param ... un-used.
#'
#' @return
#' List with means and quantiles.
#'
#' @importFrom graphics plot lines par
#' @method plot mssm
#' @export
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
#'   # plot predicted values and prediction intervals
#'   plot(pf)
#'   plot(pf, qs = c(.01, .99))
#'   pf <- ll_func$smoother(pf)
#'   plot(pf, which_weights = "smooth")
#' }
plot.mssm <- function(x, y, qs = c(.05, .95), do_plot = TRUE,
                      which_weights = c("filter", "smooth"), ...){
  stopifnot(inherits(x, "mssm"), qs[2] > qs[1], all(qs > 0, qs < 1),
            is.logical(do_plot))
  which_weights <- which_weights[1L]
  stopifnot(which_weights %in% c("filter", "smooth"),
            !which_weights == "smooth" ||
              !is.null(x$pf_output[[1L]]$ws_normalized_smooth))

  particles <- lapply(x$pf_output, "[[", "particles")
  ws <- lapply(
    x$pf_output, "[[", if(which_weights == "smooth")
      "ws_normalized_smooth" else "ws_normalized")

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
           xlab = "time", type = "l",
           ylim = range(lbs[i, ], ubs[i, ], filter_ests[i, ]))
      lines(idx_time, lbs[i, ], lty = 2)
      lines(idx_time, ubs[i, ], lty = 2)
    }
  }

  # TODO: test output
  invisible(list(means = filter_ests, lbs = lbs, ubs = ubs))
}

#' @title Plot Effective Sample Sizes
#' @description
#' Plots the effective sample sizes.
#'
#' @param x an object of class \code{mssmEss}.
#' @param y un-used.
#' @param ... un-used.
#'
#' @return
#' The plotted x-values, y-values, and maximum possible effective sample
#' size.
#'
#' @examples
#' if(require(Ecdat)){
#'   # load data and fit glm to get some parameters to use in an illustration
#'   data("Gasoline", package = "Ecdat")
#'   glm_fit <- glm(lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
#'                  Gamma("log"), Gasoline)
#'
#'   # get object to run particle filter
#'   library(mssm)
#'   ll_func <- mssm(
#'     fixed = formula(glm_fit), random = ~ 1, family = Gamma("log"),
#'     data = Gasoline, ti = year, control = mssm_control(
#'       N_part = 1000L, n_threads = 1L))
#'
#'   # run particle filter
#'   pf <- ll_func$pf_filter(
#'     cfix = coef(glm_fit), disp = summary(glm_fit)$dispersion,
#'     F. = as.matrix(.0001), Q = as.matrix(.0001^2))
#'
#'   # plot effective samples sizes
#'   plot(get_ess(pf))
#' }
#'
#' @importFrom graphics plot abline par
#' @method plot mssmEss
#' @export
plot.mssmEss <- function(x, y, ...){
  stopifnot(inherits(x, "mssmEss"))

  x_vals <- as.integer(names(x))
  par_old <- par(no.readonly = TRUE)
  on.exit(par(par_old))
  par(mar = c(5, 4, 1, 1))
  ylim <- c(0, attr(x, "n_max") * 1.04)
  plot(x_vals, x, ylim = ylim, type = "h", yaxs = "i", xlab = "Time",
       ylab = "Effective sample size")
  abline(h = attr(x, "n_max"), lty = 2)

  # TODO: test output
  invisible(list(x = x_vals, y = x, ylim = ylim))
}
