#' @title Effective Sample Sizes of a mssm Object
#' @description
#' Extracts the effective sample size at each time point from a \code{mssm}
#' object.
#'
#' @param object an object of class \code{mssm}.
#'
#' @return
#' An object of class \code{mssmEss} with the effective sample sizes.
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
#'   # summary statistics for effective sample sizes
#'   print(ess <- get_ess(pf))
#' }
#' @export
get_ess <- function(object){
  stopifnot(inherits(object, "mssm"))

  ws <- lapply(object$pf_output, "[[", "ws_normalized")
  ess_inv <- sapply(ws, function(x) sum(exp(2 * x)))

  # TODO: test output
  structure(1 / ess_inv, names = .get_time_index(object), class = "mssmEss",
            n_max = object$control$N_part)
}
