context("Test versus old results for 'mssm' methods")

test_that("get the same with 'poisson_log'", {
  ctrl <- mssm_control(N_part = 100L, n_threads = 2L, seed = 26545947)

  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = poisson(),
    data = poisson_log$data, ti = time_idx, control = ctrl)

  expect_known_value(func[mssmFunc_ele_to_check], "mssmFunc-poisson-log.RDS")

  func_out <- func$pf_filter(
    cfix = poisson_log$cfix, F. = poisson_log$F., Q = poisson_log$Q,
    disp = numeric())

  expect_known_value(func_out[mssm_ele_to_check], "mssm-poisson-log.RDS")

  ctrl$what <- "gradient"
  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = poisson(),
    data = poisson_log$data, ti = time_idx, control = ctrl)

  func_out_grad <- func$pf_filter(
    cfix = poisson_log$cfix, F. = poisson_log$F., Q = poisson_log$Q,
    disp = numeric())

  expect_equal(
    lapply(func_out_grad$pf_output, "[", "ws"),
    lapply(func_out     $pf_output, "[", "ws"))

  expect_known_value(
    lapply(func_out_grad$pf_output, "[[", "stats"),
    "mssm-gradient-poisson-log.RDS")

  #####
  # w/ k-d tree method
  ctrl <- mssm_control(N_part = 100L, n_threads = 2L, seed = 26545947,
                       what = "gradient", which_ll_cp = "KD")

  func <- mssm(
    fixed = y ~ x + Z, random = ~ Z, family = poisson(),
    data = poisson_log$data, ti = time_idx, control = ctrl)
  func_out <- func$pf_filter(
    cfix = poisson_log$cfix, F. = poisson_log$F., Q = poisson_log$Q,
    disp = numeric())
  expect_known_value(func_out[mssm_ele_to_check], "mssm-poisson-log-kd.RDS")
})
