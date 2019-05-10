#include "dists.h"
#include <testthat.h>
#include <array>
#include "utils-test.h"

context("Test state distribution") {
  test_that("Test mvs_norm gives correct results in 3D") {
    auto x = create_vec<3L>({ -3, 2, 1 });
    auto y = create_vec<3L>({ -1, 0, 2 });

    std::unique_ptr<trans_obj> kernel(new mvs_norm(3L));

    /* print(sum(dnorm(c(2, 2, 1), log = TRUE)), digits = 16) */
    constexpr double expect = -7.256815599614018;
    expect_true(std::abs(kernel->operator()(
        x.begin(), y.begin(), 3L, 0.) - expect) < 1e-8);
    expect_true(std::abs(kernel->operator()(
        x.begin(), y.begin(), 3L, 1) - (expect + 1.)) < 1e-8);

    mvs_norm k2(y);
    expect_true(std::abs(k2.log_density_state(
        x, nullptr, nullptr, log_densty) - expect) < 1e-8);
  }

  test_that("Test mv_norm gives correct results in 3D") {
    /* R code
       x <- c(-3, 2, 3)
       y <- c(-1, 0, 2)
       Q <- matrix(c(2, 1, 1,
                     1, 1, 1,
                     1, 1, 3), 3L)

       library(mvtnorm)
       dput(dmvnorm(x, y, Q, log = TRUE))
    */

    auto x = create_vec<3L>({ -3, 2, 3 });
    auto y = create_vec<3L>({ -1, 0, 2 });
    auto Q = create_mat<3L, 3L>({ 2, 1, 1,
                                  1, 1, 1,
                                  1, 1, 3 });

    mv_norm di(Q);

    constexpr double expect = -13.353389189894;

    {
      arma::mat x1 = x, y1 = y;
      di.trans_X(x1);
      di.trans_Y(y1);

      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 3L, 0.) - expect) < 1e-8);
      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 3L, 1) - (expect + 1.)) < 1e-8);

      di.trans_inv_X(x1);
      expect_true(is_all_aprx_equal(x1, x));
      di.trans_inv_Y(y1);
      expect_true(is_all_aprx_equal(y1, y));
    }

    mv_norm di2(Q, y);

    expect_true(std::abs(di2.log_density_state(
        x, nullptr, nullptr, log_densty) - expect) < 1e-8);
    expect_true(std::abs(di2.log_prop_dens(
        x                                     ) - expect) < 1e-8);
  }

  test_that("Test mv_norm_reg gives correct results in 3D") {
    /* R code
    x <- c(-3, 2)
    y <- c(-1, 0)
    F. <- matrix(c(.8, .2, .1, .3), 2L)
    Q <- matrix(c(2, 1, 1, 1), 2L)

    library(mvtnorm)
    dput(dmvnorm(y, F. %*% x, Q, log = TRUE))
    F. %*% x
    */

    auto x = create_vec<2L>({ -3, 2});
    auto y = create_vec<2L>({ -1, 0,});
    auto Q = create_mat<2L, 2L>({ 2, 1, 1, 1});
    auto F = create_mat<2L, 2L>({.8, .2, .1, .3});

    mv_norm_reg di(F, Q);

    constexpr double expect = -2.55787706640935;

    {
      arma::mat x1 = x, y1 = y;
      di.trans_X(x1);
      di.trans_Y(y1);

      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 0.) - expect) < 1e-8);
      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 1) - (expect + 1.)) < 1e-8);

      di.trans_inv_X(x1);
      expect_true(is_all_aprx_equal(x1, x));
      di.trans_inv_Y(y1);
      expect_true(is_all_aprx_equal(y1, y));
    }

    auto mea = di.mean(x);
    auto expected = create_vec<2L>({-2.2, 0.});
    expect_true(is_all_aprx_equal(mea, expected));
  }

  test_that("Test mv_norm_reg::comp_stats_state_state gives correct results in 3D"){
    /* R code
# 3D example
    x <- c(-3, 2, 1)
    y <- c(-1, 0, 2)
    F. <- matrix(c(.8, .2, .3, .1, .6, .2, .1, .1, .5), 3L)
    Q <- matrix(c(3, 1, 1, 1, 2, 3, 1, 3, 7), 3L)
    library(mvtnorm)
    library(numDeriv)

# log-likelihood (which will not fail even if Q is not symmetric -- slightly
# odd)
    func <- function(fq){
    F.[] <- fq[1:9]
    Q[] <- fq[-(1:9)]
    rss <- y - F. %*% x
    rss <- crossprod(rss, solve(Q, rss))
    c(-log(2 * pi) * 3 / 2 - determinant(Q)$modulus / 2 - drop(rss) / 2)
    }
    n <- 3L
    nn <- n * n
    K <- matrix(0., n * n + n * (n + 1L) / 2L, 2L * n * n)
    K[  1:nn,    1:nn] <- diag(nn)
    library(matrixcalc)
    K[-(1:nn), -(1:nn)] <- t(duplication.matrix(n))

    dput(K %*% drop(jacobian(func, c(F., Q))))
    dput(tcrossprod(K %*% hessian(func, c(F., Q)), K))
    */

    auto x = create_vec<3L>({ -3, 2, 1 });
    auto y = create_vec<3L>({ -1, 0, 2});
    auto F = create_mat<3L, 3L>({ .8, .2, .3, .1, .6, .2, .1, .1, .5 });
    auto Q = create_mat<3L, 3L>({ 3, 1, 1, 1, 2, 3, 1, 3, 7 });

    mv_norm_reg di(F, Q);
    di.trans_X(x);
    di.trans_Y(y);

    {
      std::array<double, 0L> stat;
      expect_true(di.obs_stat_dim     (log_densty)   == 0L);
      expect_true(di.obs_stat_dim_grad(log_densty)   == 0L);
      expect_true(di.obs_stat_dim_hess(log_densty)   == 0L);

      expect_true(di.obs_stat_dim     (gradient)   == 0L);
      expect_true(di.obs_stat_dim_grad(gradient)   == 0L);
      expect_true(di.obs_stat_dim_hess(gradient)   == 0L);

      expect_true(di.obs_stat_dim     (Hessian)   == 0L);
      expect_true(di.obs_stat_dim_grad(Hessian)   == 0L);
      expect_true(di.obs_stat_dim_hess(Hessian)   == 0L);

      expect_true(di.state_stat_dim     (log_densty) == 0L);
      expect_true(di.state_stat_dim_grad(log_densty) == 0L);
      expect_true(di.state_stat_dim_hess(log_densty) == 0L);
      /* run to check it does not throw or access memory that it should not */
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), 1, stat.data(), log_densty);
    }
    {
      constexpr unsigned int
        dim = 3L,
          dimdim = dim * dim, dimlower = (dim * (dim + 1L)) / 2L,
          gdim = dimdim  + dimlower;
      expect_true(di.state_stat_dim     (gradient)   == gdim);
      expect_true(di.state_stat_dim_grad(gradient)   == gdim);
      expect_true(di.state_stat_dim_hess(gradient)   == 0L);
      expect_true(di.state_stat_dim     (Hessian)    == gdim * (gdim + 1L));
      expect_true(di.state_stat_dim_grad(Hessian)    == gdim );
      expect_true(di.state_stat_dim_hess(Hessian)    == gdim * gdim);

      std::array<double, 18L> stat;
      arma::mat d_F(stat.data(), dim, dim, false);
      d_F.zeros();
      arma::vec d_Q(stat.data() + dimdim, dimlower, false);
      d_Q.zeros();
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), 1, stat.data(), gradient);

      auto d_F_expect = create_mat<3L, 3L>({
        -2.57499999997225, 8.60000000021791, -4.17499999999886, 1.71666666608245,
        -5.73333333327812, 2.78333333319569, 0.858333333297878, -2.86666666645546,
        1.39166666672156 });
      auto d_Q_expect = create_vec<6L>({
        0.160034722201642, -2.12722222221256,
        1.1111805555525, 3.27555555551822, -3.32277777776056, 0.760034722214741 });

      expect_true(is_all_aprx_equal(d_F, d_F_expect, 1e-4));
      expect_true(is_all_aprx_equal(d_Q, d_Q_expect, 1e-4));

      /* mult by .5 weight instead */
      d_F.zeros();
      d_Q.zeros();
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), .5, stat.data(), gradient);

      arma::mat ep_F_half = d_F_expect * .5;
      arma::vec ep_Q_half = d_Q_expect * .5;

      expect_true(is_all_aprx_equal(d_F, ep_F_half, 1e-4));
      expect_true(is_all_aprx_equal(d_Q, ep_Q_half, 1e-4));

      /* add something already */
      d_F.fill(1.);
      d_Q.fill(1.);
      di.comp_stats_state_state(
        x.memptr(), y.memptr(), .5, stat.data(), gradient);

      arma::mat ep_F_p1 = .5 * d_F_expect + 1;
      arma::mat ep_Q_p1 = .5 * d_Q_expect + 1;

      expect_true(is_all_aprx_equal(d_F, ep_F_p1, 1e-4));
      expect_true(is_all_aprx_equal(d_Q, ep_Q_p1, 1e-4));

      /* check Hessian */
      auto dd_FQ_expect = create_mat<15L, 15L>({
        -3.75000000001299, 3.00000000003825, -0.749999999984016,
        2.50000000000753, -1.99999999999334, 0.500000000022459, 1.2500000000573,
        -0.999999999899219, 0.250000000022528, 1.07291666666149, -4.44166666665276,
        1.95416666666104, 2.86666654737098, -2.10833333187424, 0.34791666178881,
        3.00000000003825, -15.0000000004804, 6.00000000007797, -1.99999999990926,
        10.0000000000661, -3.99999999979404, -0.999999999449363, 5.0000000006222,
        -1.99999999993395, -0.858333333298505, 7.15833333343639, -3.10833333326402,
        -14.3333326781599, 12.6916666589777, -2.78333329134723, -0.749999999984016,
        6.00000000007797, -3.75000000000606, 0.499999999947989, -4.00000000000808,
        2.50000000004598, 0.250000000179224, -2.000000000098, 1.24999999998746,
        0.214583333336156, -2.4333333333167, 1.42083333334348, 5.7333330921276,
        -6.3666666622253, 1.73958330948347, 2.50000000000753, -1.99999999990926,
        0.499999999947989, -1.66666666699791, 1.33333333325643, -0.33333333378846,
        -0.833333333254639, 0.666666666654591, -0.166666666706317, -0.715277777775954,
        2.96111111121732, -1.30277777782467, -1.91111102779903, 1.40555555464681,
        -0.231944441102175, -1.99999999999334, 10.0000000000661, -4.00000000000808,
        1.33333333325643, -6.66666666668792, 2.66666666668927, 0.666666666762419,
        -3.33333333329152, 1.33333333334402, 0.572222222219239, -4.77222222220703,
        2.07222222221937, 9.55555518394817, -8.46111110465039, 1.85555553151206,
        0.500000000022459, -3.99999999979404, 2.50000000004598, -0.33333333378846,
        2.66666666668927, -1.66666666673728, -0.166666666466358, 1.33333333335919,
        -0.833333333329364, -0.143055555551966, 1.62222222226664, -0.947222222229056,
        -3.82222205388102, 4.24444444172141, -1.15972220520612, 1.2500000000573,
        -0.999999999449363, 0.250000000179224, -0.833333333254639, 0.666666666762419,
        -0.166666666466358, -0.416666668394122, 0.333333334131987, -0.0833333331035143,
        -0.35763888883512, 1.48055555565202, -0.65138888870871, -0.955555513928015,
        0.702777777367556, -0.115972220541768, -0.999999999899219, 5.0000000006222,
        -2.000000000098, 0.666666666654591, -3.33333333329152, 1.33333333335919,
        0.333333334131987, -1.66666666788549, 0.6666666666947, 0.286111111125734,
        -2.38611111099389, 1.03611111117898, 4.77777757210076, -4.23055555272995,
        0.927777764561319, 0.250000000022528, -1.99999999993395, 1.24999999998746,
        -0.166666666706317, 1.33333333334402, -0.833333333329364, -0.0833333331035143,
        0.6666666666947, -0.416666666660727, -0.0715277777777637, 0.811111111108802,
        -0.473611111106567, -1.91111102665252, 2.12222222086895, -0.579861102562081,
        1.07291666666149, -0.858333333298505, 0.214583333336156, -0.715277777775954,
        0.572222222219239, -0.143055555551966, -0.35763888883512, 0.286111111125734,
        -0.0715277777777637, -0.220167824073935, 1.13192129629208, -0.524386574036955,
        -0.764629352456024, 0.575439814142386, -0.0960705956328093, -4.44166666665276,
        7.15833333343639, -2.4333333333167, 2.96111111121732, -4.77222222220703,
        1.62222222226664, 1.48055555565202, -2.38611111099389, 0.811111111108802,
        1.13192129629208, -5.48678240737529, 2.45108796298785, 6.28462883549372,
        -5.28474533856578, 1.07324070656778, 1.95416666666104, -3.10833333326402,
        1.42083333334348, -1.30277777782467, 2.07222222221937, -0.947222222229056,
        -0.65138888870871, 1.03611111117898, -0.473611111106567, -0.524386574036955,
        2.45108796298785, -1.13247685187938, -2.74796234044746, 2.6051620314261,
        -0.624386541187941, 2.86666654737098, -14.3333326781599, 5.7333330921276,
        -1.91111102779903, 9.55555518394817, -3.82222205388102, -0.955555513928015,
        4.77777757210076, -1.91111102665252, -0.764629352456024, 6.28462883549372,
        -2.74796234044746, -12.3074068242617, 11.0164810901563, -2.43738377934538,
        -2.10833333187424, 12.6916666589777, -6.3666666622253, 1.40555555464681,
        -8.46111110465039, 4.24444444172141, 0.702777777367556, -4.23055555272995,
        2.12222222086895, 0.575439814142386, -5.28474533856578, 2.6051620314261,
        11.0164810901563, -10.832331485813, 2.6756481209151, 0.34791666178881,
        -2.78333329134723, 1.73958330948347, -0.231944441102175, 1.85555553151206,
        -1.15972220520612, -0.115972220541768, 0.927777764561319, -0.579861102562081,
        -0.0960705956328093, 1.07324070656778, -0.624386541187941, -2.43738377934538,
        2.6756481209151, -0.720167812719562 });

      std::array<double, gdim * (gdim + 1L)> stat_w_Hes;
      std::fill(stat_w_Hes.data(), stat_w_Hes.end(), 0.);
      d_F = arma::mat(stat_w_Hes.data()       , 3L  , 3L  , false);
      d_Q = arma::vec(stat_w_Hes.data() + 9L  , 6L        , false);
      arma::mat dd_FQ(stat_w_Hes.data() + gdim, gdim, gdim, false);

      di.comp_stats_state_state(
        x.memptr(), y.memptr(), 1, stat_w_Hes.data(), Hessian);

      expect_true(is_all_aprx_equal(d_F  , d_F_expect  , 1e-4));
      expect_true(is_all_aprx_equal(d_Q  , d_Q_expect  , 1e-4));
      expect_true(is_all_aprx_equal(dd_FQ, dd_FQ_expect, 1e-4));
    }
  }

  test_that("Test mv_tdist gives correct results in 3D") {
    /* R code
      x <- c(-3, 2, 3)
      y <- c(-1, 0, 1)
      Q <- matrix(c(2, 1, 1,
                    1, 1, 1,
                    1, 1, 3), 3L)
      nu <- 5

      library(mvtnorm)
      dput(dmvt(x, y, Q, df = nu, log = TRUE))
    */

    auto x = create_vec<3L>({-3, 2, 3});
    auto y = create_vec<3L>({-1, 0, 1});
    auto Q = create_mat<3L, 3L>({2, 1, 1,
                                 1, 1, 1,
                                 1, 1, 3});
    const double nu = 5;

    mv_tdist di(Q, nu);

    constexpr double expect = -9.40850033868649;

    {
      arma::mat x1 = x, y1 = y;
      di.trans_X(x1);
      di.trans_Y(y1);

      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 0.) - expect) < 1e-8);
      expect_true(std::abs(di(
          x1.begin(), y1.begin(), 2L, 1) - (expect + 1.)) < 1e-8);

      di.trans_inv_X(x1);
      expect_true(is_all_aprx_equal(x1, x));
      di.trans_inv_Y(y1);
      expect_true(is_all_aprx_equal(y1, y));
    }

    mv_tdist di2(Q, y, nu);
    expect_true(std::abs(di2.log_density_state(
        x, nullptr, nullptr, log_densty) - expect) < 1e-8);
    expect_true(std::abs(di2.log_prop_dens(
        x                                     ) - expect) < 1e-8);
  }
}

template<typename family>
inline void test_exp_fam_func
  (const arma::vec co, const arma::mat X, const arma::vec y,
   const arma::vec w, const arma::vec di, const arma::vec je,
   const arma::mat H){
  arma::vec state(1L, arma::fill::zeros);
  arma::mat Z(1L, X.n_cols, arma::fill::zeros);
  arma::vec offs(X.n_cols, arma::fill::zeros);

  family obj(y, X, co, Z, &w, di, offs);

  unsigned pp1 = co.n_elem + 1L, mem_size = pp1 * (1L + pp1);
  std::unique_ptr<double[]> mem(new double[mem_size]);
  std::fill(mem.get(), mem.get() + mem_size, 0.);

  arma::vec je_out(mem.get(), pp1, false);
  arma::mat H_out (mem.get() + pp1, pp1, pp1, false);

  obj.comp_stats_state_only(state, mem.get(), Hessian);

  expect_true(is_all_aprx_equal(je_out, je, 1e-5));
  expect_true(is_all_aprx_equal(H_out, H, 1e-5));
}

context("testing that derivatives are correct for exponential families with a dispersion paramter") {
  /* R code to generate test results
   test_func <- function(co, x, y, disp, ws, dfun, linkinv){
   gen <- function(dfun, linkinv){
   function(coefdisp){
   n <- length(coefdisp)
   disp <- coefdisp[n]
   co <- coefdisp[-n]
   sum(ws * dfun(mu = linkinv(co %*% x), disp = disp, y = y))
   }
   }

   obj <- gen(dfun, linkinv)

   library(numDeriv)
   print(ja <- jacobian(obj, c(co, disp), method.args=list(eps=1e-8)))
   dput(ja)
   print(he <- hessian(obj, c(co, disp), method.args=list(eps=1e-8)))
   dput(he)
   }
   */

  test_that("gaussian_identity gives correct derivatives"){
    /* R code
     dput(x <- matrix(seq(-2, 2.5, length.out = 10), 2))
     dput(y <- seq(-1, 1, length.out = 5))
     ws <- dput(c(0.159, 0.485, 0.083, 0.235, 0.038))
     disp <- 2
     test_func(
     co = c(-1, 1), x = x, y = y, disp = disp, ws = ws,
     dfun = function(mu, disp, y) dnorm(y, mu, sqrt(disp), log = TRUE),
     linkinv = identity)
     */

    test_exp_fam_func<gaussian_identity>(
      create_vec<2L>({-1, 1}),
      create_mat<2L, 5L>({-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5}),
      create_vec<5L>({-1., -0.5, 0., 0.5, 1.}),
      create_vec<5L>({0.159, 0.485, 0.083, 0.235, 0.038}),
      create_vec<1L>({2}),
      create_vec<3L>({
        0.499999999998834, 0.313500000006206, -0.140874999996791}),
        create_mat<3L, 3L>({
          -0.754000000002628, -0.630999999997086, -0.249999999998314,
          -0.630999999997086, -0.63300000000448, -0.156749999997746, -0.249999999998314,
          -0.156749999997746, 0.0158749999990248 }));
  }

  test_that("gaussian_log gives correct derivatives"){
    /* R code
     dput(x <- matrix(seq(-2, 2.5, length.out = 10), 2))
     dput(y <- seq(-1, 1, length.out = 5))
     ws <- dput(c(0.159, 0.485, 0.083, 0.235, 0.038))
     disp <- 2
     test_func(
     co = c(-1, 1), x = x, y = y, disp = disp, ws = ws,
     dfun = function(mu, disp, y) dnorm(y, mu, sqrt(disp), log = TRUE),
     linkinv = exp)
     */

    test_exp_fam_func<gaussian_log>(
      create_vec<2L>({-1, 1}),
      create_mat<2L, 5L>({-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5}),
      create_vec<5L>({-1., -0.5, 0., 0.5, 1.}),
      create_vec<5L>({0.159, 0.485, 0.083, 0.235, 0.038}),
      create_vec<1L>({2}),
      create_vec<3L>({
        1.29026524884408, 0.509298433582429, 0.238306586713173 }),
      create_mat<3L, 3L>({
        -4.60120462424153, -3.62172333483588, -0.645132624417262,
        -3.62172333483588, -3.3725106817756, -0.254649216792272, -0.645132624417262,
        -0.254649216792272, -0.363306586704076 }));
  }

  test_that("gaussian_inverse gives correct derivatives"){
    /* R code
     dput(x <- matrix(seq(-1, 1.25, length.out = 10), 2))
     dput(y <- seq(-1, 1, length.out = 5))
     ws <- dput(c(0.159, 0.485, 0.083, 0.235, 0.038))
     disp <- 2
     test_func(
     co = c(1, 2), x = x, y = y, disp = disp, ws = ws,
     dfun = function(mu, disp, y) dnorm(y, mu, sqrt(disp), log = TRUE),
     linkinv = function(x) 1/x)
     */

    test_exp_fam_func<gaussian_inverse>(
      create_vec<2L>({1, 2}),
      create_mat<2L, 5L>({-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1., 1.25}),
      create_vec<5L>({-1., -0.5, 0., 0.5, 1.}),
      create_vec<5L>({0.159, 0.485, 0.083, 0.235, 0.038}),
      create_vec<1L>({2}),
      create_vec<3L>({
        0.0518851282859568, 0.106203660345803, -0.183765280618064 }),
        create_mat<3L, 3L>({
          -0.118509081859038, -0.0596930366931129, -0.0259425641386291,
          -0.0596930366931129, -0.155862422431851, -0.0531018301711219,
          -0.0259425641386291, -0.0531018301711219, 0.0587652806109991 }));
  }

  test_that("Gamma_log gives correct derivatives"){
    /* R code
     dput(x <- matrix(seq(-1, 1.25, length.out = 10), 2))
     dput(y <- seq(1, 2, length.out = 5))
     ws <- dput(c(0.159, 0.485, 0.083, 0.235, 0.038))
     disp <- 2
     test_func(
     co = c(1, 3), x = x, y = y, disp = disp, ws = ws,
     dfun = function(mu, disp, y) dgamma(y, 1/disp, scale = mu * disp, log = TRUE),
     linkinv = exp)
     */

    test_exp_fam_func<Gamma_log>(
      create_vec<2L>({1, 3}),
      create_mat<2L, 5L>({-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1., 1.25}),
      create_vec<5L>({1., 1.25, 1.5, 1.75, 2.}),
      create_vec<5L>({0.159, 0.485, 0.083, 0.235, 0.038}),
      create_vec<1L>({2}),
      create_vec<3L>({
        -2.44943578027701, -1.78663053495889, 0.874587896858328 }),
        create_mat<3L, 3L>({
          -2.31844963169785, -1.67534068663101, 1.22471789012685,
          -1.67534068663101, -1.2291830528889, 0.893315267479427, 1.22471789012685,
          0.893315267479427, -1.0580130344079 }));
  }
}
