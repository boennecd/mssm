context("Testing manual page results")

for(n_threads in c(1L, 4L)){
  skip_on_cran()
  test_that("manual page for 'mssm' yields the same output", {
    skip_if_not_installed("Ecdat")
    if(n_threads > 1L)
      skip_on_cran()
    skip_if(Sys.getenv("ISTRAVIS", "No") != "No")

    # load data and fit glm model to get starting values
    data("Gasoline", package = "Ecdat")
    glm_fit <- glm(lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
                   Gamma("log"), Gasoline)

    # get object to perform estimation
    label <- paste("'n_threads': ", n_threads)
    library(mssm)
    ll_func <- mssm(
      fixed = formula(glm_fit), random = ~ 1, family = Gamma("log"),
      data = Gasoline, ti = year, control = mssm_control(
        N_part = 1000L, n_threads = n_threads))
    expect_known_output(
      ll_func, "mssm-man-ll_func.txt", print = TRUE, label = label)

    # fit model with time-varying intercept with Laplace approximation
    disp <- summary(glm_fit)$dispersion
    laplace <- ll_func$Laplace(
      cfix = coef(glm_fit), disp = disp, F. = diag(.5, 1), Q = diag(1))
    expect_known_output(
      laplace, "mssm-man-laplace.txt", print = TRUE, label = label)

    # compare w/ glm
    expect_known_output(
      logLik(laplace), "mssm-man-logLik-laplace.txt", print = TRUE,
      label = label)
    expect_known_value(
      rbind(laplace = laplace$cfix, glm = coef(glm_fit)),
      "mssm-man-rbind-laplace-glm.RDS", label = label)

    # run particle filter
    pf <- ll_func$pf_filter(
      cfix = laplace$cfix, disp = laplace$disp, F. = laplace$F., Q = laplace$Q)
    expect_known_output(
      pf, "mssm-man-pf.txt", print = TRUE, label = label)

    # compare approximate log-likelihood
    expect_known_output(
      logLik(pf), "mssm-man-logLik-pf.txt", print = TRUE, label = label)

    # predicted values from filtering (does not appear random...)
    pout <- plot(pf)
    expect_known_value(pout, "mssm-man-plot-pf.RDS", label = label)

    # plot predicted values from smoothing distribution
    pf <- ll_func$smoother(pf)
    sm <- plot(pf, which_weights = "smooth")
    expect_known_value(sm, "mssm-man-plot-pf-sm.RDS", label = label)
  })
}

for(n_threads in c(1L, 4L)){
  test_that("manual page for 'plot.mssm' yields the same output", {
    skip_if_not_installed("Ecdat")
    if(n_threads > 1L)
      skip_on_cran()

    # load data and get object to perform particle filtering
    data("Gasoline", package = "Ecdat")

    library(mssm)
    label <- paste("'n_threads': ", n_threads)
    ll_func <- mssm(
      fixed = lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
      random = ~ 1, family = Gamma("log"), data = Gasoline, ti = year,
      control = mssm_control(N_part = 1000L, n_threads = n_threads))

    # run particle filter
    cfix <- c(0.612, -0.015, 0.214, 0.048, -0.013, -0.016, -0.022, 0.047,
              -0.046, 0.007, -0.001, 0.008, -0.117, 0.075, 0.048, -0.054, 0.017,
              0.228, 0.077, -0.056, -0.139)
    pf <- ll_func$pf_filter(
      cfix = cfix, Q = as.matrix(2.163e-05), F. = as.matrix(0.9792),
      disp = 0.000291)

    # plot predicted values and prediction interval
    plain <- plot(pf)
    expect_known_value(plain, "plot-mssm-man-plain.RDS", label = label)

    wide <- plot(pf, qs = c(.01, .99))
    expect_known_value(wide, "plot-mssm-man-wide.RDS", label = label)
    pf <- ll_func$smoother(pf)
    smooth <- plot(pf, which_weights = "smooth")
    expect_known_value(smooth, "plot-mssm-man-smooth.RDS", label = label)
  })
}


for(n_threads in c(1L, 4L)){
  test_that("manual page for 'get_ess.mssm' and `plot.mssmEss` yields the same output", {
    skip_if_not_installed("Ecdat")
    if(n_threads > 1L)
      skip_on_cran()

    # load data and fit glm to get some parameters to use in an illustration
    data("Gasoline", package = "Ecdat")
    glm_fit <- glm(lgaspcar ~ factor(country) + lincomep + lrpmg + lcarpcap,
                   Gamma("log"), Gasoline)

    # get object to run particle filter
    library(mssm)
    ll_func <- mssm(
      fixed = formula(glm_fit), random = ~ 1, family = Gamma("log"),
      data = Gasoline, ti = year, control = mssm_control(
        N_part = 1000L, n_threads = 1L))

    # run particle filter
    pf <- ll_func$pf_filter(
      cfix = coef(glm_fit), disp = summary(glm_fit)$dispersion,
      F. = as.matrix(.0001), Q = as.matrix(.0001^2))

    # summary statistics for effective sample sizes
    expect_known_output((ess <- get_ess(pf)), "get_ess.txt", print = TRUE)
    expect_known_value(plot(ess), "get_ess.RDS")
  })
}
