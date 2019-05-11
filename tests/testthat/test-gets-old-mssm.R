context("Test versus old results for 'mssm' methods")

prep_for_test_mssmFunc <- function(obj){
  obj$control$n_threads <- NULL
  obj
}

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
  obj$control$n_threads <- NULL
  obj
}

get_test_expr <- function(data, label, family, alway_hess = FALSE, n_threads){
  substitute({
  ctrl <- mssm_control(N_part = 100L, n_threads = n_threads, seed = 26545947,
                       maxeval = 1L)
  disp <- if(is.null(dat$disp)) numeric() else dat$disp

  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = family,
    data = dat$data, ti = time_idx, control = ctrl)
  expect_s3_class(func, "mssmFunc")

  expect_known_value(
    prep_for_test_mssmFunc(func[mssmFunc_ele_to_check]),
    paste0("mssmFunc-", label, ".RDS"), label = label)

  func_out <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)
  expect_s3_class(func_out, "mssm")

  func_out_org <- func_out
  func_out <- prep_for_test(func_out)
  expect_known_value(
    func_out[mssm_ele_to_check], paste0("mssm-", label, ".RDS"),
    label = label)

  # test only the first smoothed particle weights
  smoother_no_approx <- func$smoother(func_out_org)
  expect_s3_class(smoother_no_approx, "mssm")
  expect_known_value(
    sapply(head(smoother_no_approx$pf_output, 1L), "[[",
           "ws_normalized_smooth"), paste0("mssm-smooth-", label, ".RDS"),
    label = label)

  # take a few iterations of laplace approximation
  lpa <- func$Laplace(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q, disp = disp)
  expect_s3_class(lpa, "mssmLaplace")

  lpa$control["n_threads"] <- NULL
  expect_known_value(
    lpa[mssmLaplace_to_check], tolerance = 1e-5,
    paste0("mssmLaplace-", label, ".RDS"),
    label = label)

  # make gradient approximation
  ctrl$what <- "gradient"
  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = family,
    data = dat$data, ti = time_idx, control = ctrl)
  expect_s3_class(func, "mssmFunc")

  func_out_grad <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)
  expect_s3_class(func_out_grad, "mssm")

  expect_equal(
    lapply(func_out_grad$pf_output, "[", "ws"),
    lapply(func_out_org $pf_output, "[", "ws"),
    label = label)

  expect_known_value(
    lapply(func_out_grad$pf_output, "[[", "stats"),
    paste0("mssm-gradient-", label, ".RDS"),
    label = label)

  if(alway_hess || dir.exists("local-tests-res")){
    f <- paste0("mssm-hess-", label, ".RDS")
    if(!alway_hess)
      f <- file.path("local-tests-res", f)

    func <- mssm(
      fixed = y ~ x + Z, random = ~ Z, family = family,
      data = dat$data, ti = time_idx,
      control = mssm_control(
        N_part = 50L, n_threads = n_threads, seed = 26545947,
        what = "Hessian"))
    expect_s3_class(func, "mssmFunc")

    func_out_hess <- func$pf_filter(
      cfix = dat$cfix, F. = dat$F., Q = dat$Q,
      disp = disp)
    func_out_hess$control["n_threads"] <- NULL
    expect_s3_class(func_out_hess, "mssm")

    expect_known_value(
      func_out_hess[mssm_ele_to_check], f, label = label)
  }

  #####
  # w/ k-d tree method
  ctrl <- mssm_control(N_part = 100L, n_threads = 1L, seed = 26545947,
                       what = "log_density", which_ll_cp = "KD")

  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = family,
    data = dat$data, ti = time_idx, control = ctrl)
  expect_s3_class(func, "mssmFunc")

  func_out <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)
  expect_s3_class(func_out, "mssm")

  # should be very similar
  smoother_eps_small <- func$smoother(func_out)
  expect_s3_class(smoother_eps_small, "mssm")
  expect_equal(smoother_eps_small$pf_output, smoother_no_approx$pf_output,
               tolerance = 1e-4)

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
  expect_s3_class(func, "mssmFunc")

  func_out_org <- func_out <- func$pf_filter(
    cfix = dat$cfix, F. = dat$F., Q = dat$Q,
    disp = disp)
  expect_s3_class(func_out, "mssm")

  func_out <- prep_for_test(func_out)
  expect_known_value(
    func_out[mssm_ele_to_check], label = label,
    paste0("mssm-", label, "-kd-large-eps.RDS"))

  # test only the first smoothed particle weights
  smoother_eps_large <- func$smoother(func_out_org)
  expect_s3_class(smoother_eps_large, "mssm")
  expect_known_value(
    sapply(head(smoother_eps_large$pf_output, 1L), "[[",
           "ws_normalized_smooth"), paste0("mssm-smooth-", label, "-kd-.RDS"),
    label = label)

  if(alway_hess || dir.exists("local-tests-res")){
    f <- paste0("mssm-hess-", label, "-kd-large-eps.RDS")
    if(!alway_hess)
      f <- file.path("local-tests-res", f)

    ctrl$what <- "Hessian"
    func <- mssm(
      fixed = y ~ x + Z, random = ~ Z, family = family,
      data = dat$data, ti = time_idx,
      control = ctrl)
    expect_s3_class(func, "mssmFunc")

    func_out_hess <- func$pf_filter(
      cfix = dat$cfix, F. = dat$F., Q = dat$Q,
      disp = disp)
    expect_s3_class(func_out_hess, "mssm")

    func_out_hess$control["n_threads"] <- NULL
    expect_known_value(
      func_out_hess[mssm_ele_to_check], f, label = label)

    # test that we get the same as with the gradient call
    t1 <- tail(func_out_hess$pf_output, 1L)[[1L]]
    t2 <- tail(func_out_org$pf_output , 1L)[[1L]]

    t1$stats <- t1$stats[1:nrow(t2$stats), ]
    expect_equal(t1, t2)
  }
  }, list(dat = substitute(data), label = label, family = substitute(family),
          disp = substitute(disp), alway_hess = alway_hess,
          n_threads = n_threads))
}

test_that(
  "get the same with 'poisson_log'", {
    eval(get_test_expr(
      poisson_log, "poisson-log", poisson(), alway_hess = TRUE,
      n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(
      poisson_log, "poisson-log", poisson(), alway_hess = TRUE,
      n_threads = 4L))
  })

test_that(
  "get the same with 'poisson_sqrt'", {

    eval(get_test_expr(poisson_sqrt, "poisson-sqrt", poisson("sqrt"),
                       n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(poisson_sqrt, "poisson-sqrt", poisson("sqrt"),
                       n_threads = 4L))
  })

test_that(
  "get the same with 'binomial_logit'", {

    eval(get_test_expr(binomial_logit, "binomial-logit", binomial(),
                       n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(binomial_logit, "binomial-logit", binomial(),
                       n_threads = 4L))
  })

test_that(
  "get the same with 'binomial_cloglog'", {

    eval(get_test_expr(
      binomial_cloglog, "binomial-cloglog", binomial("cloglog"),
      n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(
      binomial_cloglog, "binomial-cloglog", binomial("cloglog"),
      n_threads = 4L))
  })

test_that(
  "get the same with 'binomial_probit'", {

    eval(get_test_expr(
      binomial_probit, "binomial-probit", binomial("probit"),
      n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(
      binomial_probit, "binomial-probit", binomial("probit"),
      n_threads = 4L))
  })

test_that(
  "get the same with 'Gamma_log'", {

    eval(get_test_expr(Gamma_log, "Gamma-log", Gamma("log"),
                       n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(Gamma_log, "Gamma-log", Gamma("log"),
                       n_threads = 4L))
  })

test_that(
  "get the same with 'gaussian_identity'", {
    eval(get_test_expr(gaussian_identity, "gaussian-identity", gaussian(),
                       n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(gaussian_identity, "gaussian-identity", gaussian(),
                       n_threads = 4L))
  })

test_that(
  "get the same with 'gaussian_log'", {

    eval(get_test_expr(gaussian_log, "gaussian-log", gaussian("log"),
                       n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(gaussian_log, "gaussian-log", gaussian("log"),
                       n_threads = 4L))
  })

test_that(
  "get the same with 'gaussian_inverse'", {

    eval(get_test_expr(
      gaussian_inverse, "gaussian-inverse", gaussian("inverse"),
      n_threads = 1L))

    skip_on_cran()
    eval(get_test_expr(
      gaussian_inverse, "gaussian-inverse", gaussian("inverse"),
      n_threads = 4L))
  })

test_that("gets the same with Poisson data with offsets", {
  ctrl <- mssm_control(N_part = 100L, n_threads = 1L, seed = 26545947)
  poisson_log$data$offs <-
    (1:nrow(poisson_log$data)) / nrow(poisson_log$data) - .5

  ll_func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = poisson("log"),
    offsets = offs, data = poisson_log$data, ti = time_idx)

  out <- with(
    poisson_log, ll_func$pf_filter(cfix = cfix, disp = numeric(), F. = F., Q = Q))
  out <- prep_for_test(out)
  expect_known_value(out, "poisson-log-w-offsets.RDS")
})

test_that("gets the same with binomial data with weights", {
  ctrl <- mssm_control(N_part = 100L, n_threads = 1L, seed = 26545947)

  ll_func <- mssm(
    fixed = I(y/size) ~ x + Z, random = ~ Z, family = binomial(),
    weights = size, data = binomial_logit_grouped, ti = time_idx)

  out <- with(
    poisson_log, ll_func$pf_filter(cfix = cfix, disp = numeric(), F. = F., Q = Q))
  out <- prep_for_test(out)
  expect_known_value(out, "binomial-logit-grouped-w-weights.RDS")
})
