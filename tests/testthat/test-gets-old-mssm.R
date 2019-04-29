context("Test versus old results for 'mssm' methods")

get_test_expr <- function(data, label, family){
  substitute({
  ctrl <- mssm_control(N_part = 100L, n_threads = 2L, seed = 26545947)
  disp <- if(is.null(dat$disp)) numeric() else dat$disp

  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = family,
    data = dat$data, ti = time_idx, control = ctrl)

  expect_known_value(
    func[mssmFunc_ele_to_check], paste0("mssmFunc-", label, ".RDS"),
    label = label)

  func_out <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)

  prep_for_test <- function(obj){
    N <- length(obj$pf_output)
    ta <- obj$pf_output[[N]]
    if(length(ta$stats) > 0L){
      # add gradient and set stats to NA
      gr <- colSums(t(ta$stats) * drop(exp(ta$ws_normalized)))
      obj$pf_output[[N]]$stats <- ta$stats[0L, ]

    } else
      gr <- NULL

    obj$pf_output <- lapply(obj$pf_output, function(x)
      lapply(x[names(x) != "gr"], function(z){
        if(NROW(z) < NCOL(z))
          z <- t(z)
        rbind(head(z, 2L), tail(z, 2L))
      }))
    obj$pf_output[[N]]$gr <- gr
    obj
  }
  func_out_org <- func_out
  func_out <- prep_for_test(func_out)
  expect_known_value(
    func_out[mssm_ele_to_check], paste0("mssm-", label, ".RDS"),
    label = label)

  ctrl$what <- "gradient"
  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = family,
    data = dat$data, ti = time_idx, control = ctrl)

  func_out_grad <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)

  expect_equal(
    lapply(func_out_grad$pf_output, "[", "ws"),
    lapply(func_out_org $pf_output, "[", "ws"),
    label = label)

  expect_known_value(
    lapply(func_out_grad$pf_output, "[[", "stats"),
    paste0("mssm-gradient-", label, ".RDS"),
    label = label)

  #####
  # w/ k-d tree method
  ctrl <- mssm_control(N_part = 100L, n_threads = 2L, seed = 26545947,
                       what = "log_density", which_ll_cp = "KD")

  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = family,
    data = dat$data, ti = time_idx, control = ctrl)
  func_out <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)
  func_out <- prep_for_test(func_out)
  expect_known_value(
    func_out[mssm_ele_to_check], label = label,
    paste0("mssm-", label, "-kd.RDS"))

  #####
  # w/ larger epsilon
  ctrl$aprx_eps <- .1
  ctrl$what <- "gradient"

  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = family,
    data = dat$data, ti = time_idx, control = ctrl)
  func_out <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)
  func_out <- prep_for_test(func_out)
  expect_known_value(
    func_out[mssm_ele_to_check], label = label,
    paste0("mssm-", label, "-kd-large-eps.RDS"))
  }, list(dat = substitute(data), label = label, family = substitute(family),
          disp = substitute(disp)))
}

test_that(
  "get the same with 'poisson_log'",
  eval(get_test_expr(poisson_log, "poisson-log", poisson())))

test_that(
  "get the same with 'poisson_sqrt'",
  eval(get_test_expr(poisson_sqrt, "poisson-sqrt", poisson("sqrt"))))

test_that(
  "get the same with 'binomial_logit'",
  eval(get_test_expr(binomial_logit, "binomial-logit", binomial())))

test_that(
  "get the same with 'binomial_cloglog'",
  eval(get_test_expr(binomial_cloglog, "binomial-cloglog", binomial("cloglog"))))

test_that(
  "get the same with 'binomial_probit'",
  eval(get_test_expr(binomial_probit, "binomial-probit", binomial("probit"))))

test_that(
  "get the same with 'Gamma_log'",
  eval(get_test_expr(Gamma_log, "Gamma-log", Gamma("log"))))
