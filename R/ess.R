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
