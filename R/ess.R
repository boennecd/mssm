#' @title Effective Sample size of a mssm Object
#' @description
#' Extracts the effective sample size at each time point from a \code{mssm}
#' object.
#'
#' @param object an object of class \code{mssm}.
#'
#' @return
#' An object of class \code{mssmEss} with the effective sample sizes.
#'
#' @export
get_ess <- function(object){
  stopifnot(inherits(object, "mssm"))

  ws <- lapply(object$pf_output, "[[", "ws_normalized")
  ess_inv <- sapply(ws, function(x) sum(exp(2 * x)))

  # TODO: test output
  structure(1 / ess_inv, names = .get_time_index(object), class = "mssmEss",
            n_max = object$control$N_part)
}

#' @importFrom stats sd
#' @method print mssmEss
#' @export
print.mssmEss <- function(x, ...){
  stopifnot(inherits(x, "mssmEss"))

  to_print <- c(mean(x), sd(x), min(x), max(x))

  cat(
    sprintf("Effective sample sizes\n  Mean %10.1f\n  sd %12.1f\n  Min %11.1f\n  Max %11.1f\n",
            to_print[1], to_print[2], to_print[3], to_print[4]))

  # TODO: test output
  invisible(to_print)
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
