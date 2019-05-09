.print_call <- function(x){
  cat("\nCall:  ", paste(deparse(x$call), sep = "\n", collapse = "\n"),
      "\n", sep = "")

  cat("\nFamily is ",
      gsub("^(.+)(_)(.+)$", "'\\1' with link '\\3'", x$family),
      ".\n", sep = "")
}

.print_df_obs <- function(x)
  cat(
    sprintf("Number of parameters   %10d\n", .get_df(x)),
    sprintf("Number of observations %10d\n", .get_nobs(x)), sep = "")

#' @importFrom stats cov2cor
.print_params <- function(x, is_estimates){
  xtra_sing <- if(is_estimates) "estimate "  else ""
  xtra_plur <- if(is_estimates) "estimates " else ""
  formals(cat)$sep <- ""

  cat(
    "State vector is assumed to be X_t ~ N(F * X_(t-1), Q).\n\nF ",
    xtra_sing, "is\n")
  print(x$F.)

  cat("\nQ's standard deviations ", xtra_plur, "are\n")
  print(sqrt(diag(x$Q)))

  if((p <- NCOL(x$Q)) > 1L){
    cat("\nQ's correlation matrix ", xtra_plur, "is (lower triangle is shown)\n")
    Q <- cov2cor(x$Q)
    Q[upper.tri(Q, diag = TRUE)] <- NA_real_
    print(Q[-1, -p, drop = FALSE], na.print = "")

  }

  if(length(x$disp) > 0L)
    cat("\nDispersion parameter ", xtra_sing, "is " , x$disp, "\n")

  cat("\nFixed coefficients ", xtra_plur, "are\n")
  print(x$cfix)

  cat("\n")
}

#' @method print mssmFunc
#' @export
print.mssmFunc <- function(x, ...){
  stopifnot(inherits(x, "mssmFunc"))

  .print_call(x)
  cat("\n")
  .print_df_obs(x)

  invisible(x)
}

#' @importFrom stats logLik
#' @method print mssmLaplace
#' @export
print.mssmLaplace <- function(x, digits = 3, ...){
  stopifnot(inherits(x, "mssmLaplace"))

  old_digits <- getOption("digits")
  on.exit(options(digits = old_digits))
  options(digits = digits)

  .print_call(x)
  cat("Parameters are estimated with a Laplace approximation.\n")
  .print_params(x, TRUE)
  cat("Log-likelihood approximation is", c(logLik(x)), "\n")
  .print_df_obs(x)

  invisible(x)
}

#' @importFrom stats logLik
#' @method print mssm
#' @export
print.mssm <- function(x, digits = 3, ...){
  stopifnot(inherits(x, "mssm"))

  old_digits <- getOption("digits")
  on.exit(options(digits = old_digits))
  options(digits = digits)

  .print_call(x)
  .print_params(x, FALSE)
  cat(
    "Log-likelihood approximation is ", c(logLik(x)), "\n",
    x$N_part, " particles are used and summary statistics for effective sample sizes are\n",
    sep = "")
  .print.mssmEss(get_ess(x), prefix = FALSE)

  cat("\n")
  .print_df_obs(x)

  invisible(x)
}

.print.mssmEss <- function(x, prefix){
  stopifnot(inherits(x, "mssmEss"))

  to_print <- c(mean(x), sd(x), min(x), max(x))
  prefix <- if(prefix) "Effective sample sizes\n" else ""

  cat(
    prefix,
    sprintf("  Mean %10.1f\n  sd %12.1f\n  Min %11.1f\n  Max %11.1f\n",
            to_print[1], to_print[2], to_print[3], to_print[4]),
    sep = "")

  invisible(to_print)
}

#' @importFrom stats sd
#' @method print mssmEss
#' @export
print.mssmEss <- function(x, ...){
  # TODO: test output
  .print.mssmEss(x, prefix = TRUE)
}
