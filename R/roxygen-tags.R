#' @useDynLib mssm, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @import nloptr
NULL

#' Multivariate State Space Models
#'
#' @name mssm-package
#' @docType package
#'
#' @description
#' This package contains particle filter methods for multivariate observed
#' outcomes and low dimensional state vectors. The methods are intended to
#' scale well in the dimension of the observed outcomes. The main function in
#' the package is the \code{\link{mssm}} function.
#'
#' The package is still under development and the API and results of the
#' methods may change.
NULL
