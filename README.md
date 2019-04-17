
Multivariate State Space Models
===============================

[![Build Status on Travis](https://travis-ci.org/boennecd/mssm.svg?branch=master,osx)](https://travis-ci.org/boennecd/mssm)

This package provides methods to estimate models of the form

![y\_{it} \\sim g(\\eta\_{it}),\\qquad i\\in I\_t](https://chart.googleapis.com/chart?cht=tx&chl=y_%7Bit%7D%20%5Csim%20g%28%5Ceta_%7Bit%7D%29%2C%5Cqquad%20i%5Cin%20I_t "y_{it} \sim g(\eta_{it}),\qquad i\in I_t")

![\\eta\_{it} = \\vec\\gamma^\\top\\vec x\_{it} +\\vec\\beta\_t^\\top\\vec z\_{it}](https://chart.googleapis.com/chart?cht=tx&chl=%5Ceta_%7Bit%7D%20%3D%20%5Cvec%5Cgamma%5E%5Ctop%5Cvec%20x_%7Bit%7D%20%2B%5Cvec%5Cbeta_t%5E%5Ctop%5Cvec%20z_%7Bit%7D "\eta_{it} = \vec\gamma^\top\vec x_{it} +\vec\beta_t^\top\vec z_{it}")

![\\vec\\beta\_t = F\\vec\\beta\_{t-1}+\\vec\\epsilon\_t, \\qquad \\vec\\epsilon\_t\\sim N(\\vec 0, Q)](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%5Cbeta_t%20%3D%20F%5Cvec%5Cbeta_%7Bt-1%7D%2B%5Cvec%5Cepsilon_t%2C%20%5Cqquad%20%5Cvec%5Cepsilon_t%5Csim%20N%28%5Cvec%200%2C%20Q%29 "\vec\beta_t = F\vec\beta_{t-1}+\vec\epsilon_t, \qquad \vec\epsilon_t\sim N(\vec 0, Q)")

where ![g](https://chart.googleapis.com/chart?cht=tx&chl=g "g") is simple distribution, we observe ![t=1,\\dots,T](https://chart.googleapis.com/chart?cht=tx&chl=t%3D1%2C%5Cdots%2CT "t=1,\dots,T") periods, and ![I\_t](https://chart.googleapis.com/chart?cht=tx&chl=I_t "I_t"), ![y\_{it}](https://chart.googleapis.com/chart?cht=tx&chl=y_%7Bit%7D "y_{it}"), ![\\vec x\_{it}](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%20x_%7Bit%7D "\vec x_{it}"), and ![\\vec z\_{it}](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%20z_%7Bit%7D "\vec z_{it}") are known. What is multivariate is ![\\vec y\_t = \\{y\_{it}\\}\_{i\\in I\_t}](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%20y_t%20%3D%20%5C%7By_%7Bit%7D%5C%7D_%7Bi%5Cin%20I_t%7D "\vec y_t = \{y_{it}\}_{i\in I_t}") (though, ![\\vec \\beta\_t](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%20%5Cbeta_t "\vec \beta_t") can also be multivariate) and this package is written to scale well in the dimension of ![| I\_t |](https://chart.googleapis.com/chart?cht=tx&chl=%7C%20I_t%20%7C "| I_t |"). The package uses independent particle filters as suggested by Lin et al. (2005). This particular type of filter can be used in the method suggested by Poyiadjis, Doucet, and Singh (2011). I will show an example of how to use the package through the rest of the document and highlight some implementation details.

The package is not on CRAN but you can be installed from Github using e.g.,

``` r
devtools::install_github("boennecd/mssm")
```

Table of Contents
-----------------

-   [Multivariate State Space Models](#multivariate-state-space-models)
    -   [Poisson Example](#poisson-example)
        -   [Log-Likelihood Approximations](#log-likelihood-approximations)
        -   [Parameter Estimation](#parameter-estimation)
        -   [Faster Approximation](#faster-approximation)
    -   [Fast Sum-Kernel Approximation](#fast-sum-kernel-approximation)
    -   [Function Definitions](#function-definitions)
-   [References](#references)

Poisson Example
---------------

We simulate data as follows.

``` r
# simulate path of state variables 
set.seed(78727269)
n_periods <- 110L
(F. <- matrix(c(.5, .1, 0, .8), 2L))
```

    ##      [,1] [,2]
    ## [1,]  0.5  0.0
    ## [2,]  0.1  0.8

``` r
(Q <- matrix(c(.5^2, .1, .1, .7^2), 2L))
```

    ##      [,1] [,2]
    ## [1,] 0.25 0.10
    ## [2,] 0.10 0.49

``` r
(Q_0 <- matrix(c(0.333, 0.194, 0.194, 1.46), 2L))
```

    ##       [,1]  [,2]
    ## [1,] 0.333 0.194
    ## [2,] 0.194 1.460

``` r
betas <- cbind(crossprod(chol(Q_0),        rnorm(2L)                      ), 
               crossprod(chol(Q  ), matrix(rnorm((n_periods - 1L) * 2), 2)))
betas <- t(betas)
for(i in 2:nrow(betas))
  betas[i, ] <- betas[i, ] + F. %*% betas[i - 1L, ]
par(mar = c(5, 4, 1, 1))
matplot(betas, lty = 1, type = "l")
```

![](./README-fig/simulate-1.png)

``` r
# sumulate observations
cfix <- c(-1, .2, .5, -1) # gamma
n_obs <- 100L
dat <- lapply(1:n_obs, function(id){
  x <- runif(n_periods, -1, 1)
  X <- cbind(X1 = x, X2 = runif(1, -1, 1))
  z <- runif(n_periods, -1, 1)
  
  eta <- drop(cbind(1, X, z) %*% cfix + rowSums(cbind(1, z) * betas))
  y <- rpois(n_periods, lambda = exp(eta))
  
  # randomly drop some
  keep <- .2 > runif(n_periods)
  
  data.frame(y = y, X, Z = z, id = id, time_idx = 1:n_periods)[keep, ]
})
dat <- do.call(rbind, dat)

# show some properties 
nrow(dat)
```

    ## [1] 2267

``` r
head(dat)
```

    ##    y       X1      X2       Z id time_idx
    ## 7  0 -0.09531 -0.4899  0.2972  1        7
    ## 15 0  0.97242 -0.4899 -0.3414  1       15
    ## 16 1 -0.77700 -0.4899  0.6973  1       16
    ## 20 1  0.24525 -0.4899 -0.3047  1       20
    ## 22 0 -0.20661 -0.4899  0.9579  1       22
    ## 25 0  0.90989 -0.4899  0.4456  1       25

``` r
table(dat$y)
```

    ## 
    ##    0    1    2    3    4    5    6    7    8    9   11   15   18 
    ## 1465  519  163   72   20   12    4    4    4    1    1    1    1

``` r
# quick smooth of number of events vs. time
par(mar = c(5, 4, 1, 1))
plot(smooth.spline(dat$time_idx, dat$y), type = "l", xlab = "Time", 
     ylab = "Number of events")
```

![](./README-fig/simulate-2.png)

``` r
# and split by those with `Z` above and below 0
with(dat, {
  z_large <- ifelse(Z > 0, "large", "small")
  smooths <- lapply(split(cbind(dat, z_large), z_large), function(x){
    plot(smooth.spline(x$time_idx, x$y), type = "l", xlab = "Time", 
     ylab = paste("Number of events -", unique(x$z_large)))
  })
})
```

![](./README-fig/simulate-3.png)![](./README-fig/simulate-4.png)

In the above, we simulate 110 (`n_periods`) with 100 (`n_obs`) individuals. Each individual has a fixed covaraite, `X2`, and two time-varying covariates, `X1` and `Z`. One of the time-varying covariates, `Z`, has a random slope. Further, the intercept is also random.

### Log-Likelihood Approximations

We start by estimating a generalized linear model without random effects.

``` r
glm_fit <- glm(y ~ X1 + X2 + Z, poisson(), dat)
summary(glm_fit)
```

    ## 
    ## Call:
    ## glm(formula = y ~ X1 + X2 + Z, family = poisson(), data = dat)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -2.285  -0.962  -0.670   0.379   6.226  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value   Pr(>|z|)    
    ## (Intercept)  -0.7345     0.0333  -22.07    < 2e-16 ***
    ## X1            0.2501     0.0487    5.14 0.00000028 ***
    ## X2            0.5395     0.0499   10.81    < 2e-16 ***
    ## Z            -1.1091     0.0530  -20.92    < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 3358.1  on 2266  degrees of freedom
    ## Residual deviance: 2731.7  on 2263  degrees of freedom
    ## AIC: 4589
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
logLik(glm_fit)
```

    ## 'log Lik.' -2291 (df=4)

Next, we make a log-likelihood approximation with the implemented particle at the true parameters with the `mssm` function.

``` r
library(mssm)
ll_func <- mssm(
  fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat, 
  # make it explict that there is an intercept (not needed)
  random = ~ 1 + Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 500L, what = "log_density"))

system.time(
  mssm_obj <- ll_func$pf_filter(
    cfix = cfix, disp = numeric(), F. = F., Q = Q))
```

    ##    user  system elapsed 
    ##   0.753   0.018   0.197

``` r
# returns the log-likelihood approximation
logLik(mssm_obj)
```

    ## 'log Lik.' -2008 (df=NA)

As expected, we get a much higher log-likelihood. We can e.g., compare this to a model where we use splines instead for each of the two random effects.

``` r
# compare with GLM with spline
library(splines)
logLik(glm(y ~ X1 + X2 + Z * ns(time_idx, df = 20, intercept = TRUE)- 1, 
           poisson(), dat))
```

    ## 'log Lik.' -2007 (df=42)

We can plot the predicted values of state variables from the filter distribution.

``` r
# plot estiamtes 
filter_means <- plot(mssm_obj)
```

![](./README-fig/plot_filter-1.png)![](./README-fig/plot_filter-2.png)

``` r
# plot with which also contains the true paths
for(i in 1:ncol(betas)){
  be <- betas[, i]
  me <- filter_means$means[i, ]
  lb <- filter_means$lbs[i, ]
  ub <- filter_means$ubs[i, ]
  
  #     dashed: true paths
  # continuous: predicted mean from filter distribution 
  #     dotted: prediction interval
  par(mar = c(5, 4, 1, 1))
  matplot(cbind(be, me, lb, ub), lty = c(2, 1, 3, 3), type = "l", 
          col = "black", ylab = rownames(filter_means$lbs)[i])
}
```

![](./README-fig/plot_filter-3.png)![](./README-fig/plot_filter-4.png)

We can get the effective sample size at each point in time with the `get_ess` function.

``` r
(ess <- get_ess(mssm_obj))
```

    ## Effective sample sizes
    ##   Mean      460.5
    ##   sd         13.0
    ##   Min       404.4
    ##   Max       478.5

``` r
plot(ess)
```

![](./README-fig/show_ess-1.png)

We can compare this what we get by using a so-called bootstrap (like) filter instead.

``` r
local({
  ll_boot <- mssm(
    fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat, 
    random = ~ Z, ti = time_idx, control = mssm_control(
      n_threads = 5L, N_part = 500L, what = "log_density", 
      which_sampler = "bootstrap"))
  
  print(system.time(
    boot_fit <- ll_boot$pf_filter(
      cfix = coef(glm_fit), disp = numeric(), F. = F., Q = Q)))
  
  plot(get_ess(boot_fit))
})
```

    ##    user  system elapsed 
    ##   0.660   0.010   0.172

![](./README-fig/comp_boot-1.png)

The above is not much faster (and maybe slower in this run) as the bulk of the computation is not in the sampling step. We can also compare the log-likelihood approximation with what we get if we choose parameters close to the GLM estimates.

``` r
mssm_glm <- ll_func$pf_filter(
  cfix = coef(glm_fit), disp = numeric(), F. = diag(1e-8, 2), 
  Q = diag(1e-4^2, 2))
logLik(mssm_glm)
```

    ## 'log Lik.' -2297 (df=NA)

TODO: I am not sure why we get a lower log-likelihood with the above.

### Parameter Estimation

We will need to estimate the parameters for real applications. We could do this e.g., with a Monte Carlo expectation-maximization algorithm or by using a Monte Carlo approximation of the gradient. Currently, the latter is only available and the user will have to write a custom function to perform the estimation. I will provide an example below. The `sgd` function is not a part of the package. Instead the package provides a way to approximate the gradient and allows the user to perform subsequent maximization (e.g., with constraints or penalties). The definition of the `sgd` is given at the end of this file as it is somewhat long.

``` r
# setup mssmFunc object to use
ll_func <- mssm(
  fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat, 
  random = ~ Z, ti = time_idx, control = mssm_control(
    n_threads = 5L, N_part = 200L, what = "gradient"))

# use stochastic gradient descent
system.time( 
  res <- sgd(
    ll_func, F. = diag(.5, 2), Q = diag(2, 2), cfix = coef(glm_fit)))
```

    ##    user  system elapsed 
    ## 201.825   1.315  46.881

A plot of the approximate log-likelihoods at each iteration is shown below along with the final estimates.

``` r
tail(res$logLik, 1L) # final log-likelihood approximation
```

    ## [1] -2004

``` r
par(mar = c(5, 4, 1, 1))
plot(     res$logLik       , type = "l")
```

![](./README-fig/show_use_sgd-1.png)

``` r
plot(tail(res$logLik, 100L), type = "l") # only the final iterations
```

![](./README-fig/show_use_sgd-2.png)

``` r
# final estimates
res$F. 
```

    ##          [,1]    [,2]
    ## [1,]  0.40283 0.06492
    ## [2,] -0.05569 0.83593

``` r
res$Q
```

    ##         [,1]    [,2]
    ## [1,] 0.29828 0.05731
    ## [2,] 0.05731 0.31031

``` r
res$cfix
```

    ## [1] -1.0480  0.2145  0.5192 -1.0929

### Faster Approximation

One drawback with the particle filter we use is that it has ![\\mathcal{O}(N^2)](https://chart.googleapis.com/chart?cht=tx&chl=%5Cmathcal%7BO%7D%28N%5E2%29 "\mathcal{O}(N^2)") computational complexity where ![N](https://chart.googleapis.com/chart?cht=tx&chl=N "N") is the number of particles. We can see this by changing the number of particles.

``` r
local({
  # assign function that returns a function that use given number of particles
  func <- function(N){
    ll_func <- mssm(
      fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat, 
      random = ~ Z, ti = time_idx, control = mssm_control(
        n_threads = 5L, N_part = N, what = "log_density"))
    function()
      ll_func$pf_filter(
        cfix = coef(glm_fit), disp = numeric(), F. = diag(1e-8, 2), 
        Q = diag(1e-4^2, 2))
      
  }
  
  f_100  <- func( 100)
  f_200  <- func( 200)
  f_400  <- func( 400)
  f_800  <- func( 800)
  f_1600 <- func(1600)
  
  # benchmark. Should grow ~ N^2
  microbenchmark::microbenchmark(
    `100` = f_100(), `200` = f_200(), `400` = f_400(), `800` = f_800(),
    `1600` = f_1600(), times = 3L)
})
```

    ## Unit: milliseconds
    ##  expr     min      lq    mean  median      uq     max neval
    ##   100   24.02   24.93   26.01   25.85   27.00   28.16     3
    ##   200   47.53   48.60   50.60   49.67   52.14   54.61     3
    ##   400  135.57  135.90  138.64  136.23  140.17  144.11     3
    ##   800  374.26  382.28  405.15  390.31  420.59  450.88     3
    ##  1600 1257.33 1271.76 1304.84 1286.20 1328.60 1370.99     3

A solution is to use the dual k-d tree method I cover later. The computational complexity is ![\\mathcal{O}(N \\log N)](https://chart.googleapis.com/chart?cht=tx&chl=%5Cmathcal%7BO%7D%28N%20%5Clog%20N%29 "\mathcal{O}(N \log N)") for this method which is somewhat indicated by the run times shown below.

``` r
local({
  # assign function that returns a function that use given number of particles
  func <- function(N){
    ll_func <- mssm(
      fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat, 
      random = ~ Z, ti = time_idx, control = mssm_control(
        n_threads = 5L, N_part = N, what = "log_density", 
        which_ll_cp = "KD", KD_N_max = 6L, aprx_eps = 1e-2))
    function()
      ll_func$pf_filter(
        cfix = coef(glm_fit), disp = numeric(), F. = diag(1e-8, 2), 
        Q = diag(1e-4^2, 2))
      
  }
  
  f_100   <- func(  100)
  f_200   <- func(  200)
  f_400   <- func(  400)
  f_800   <- func(  800)
  f_1600  <- func( 1600)
  f_51200 <- func(51200)
  
  # benchmark. Should grow ~ N log N
  microbenchmark::microbenchmark(
    `100` = f_100(), `200` = f_200(), `400` = f_400(), `800` = f_800(), 
    `1600` = f_1600(), `51200` = f_51200(), times = 3L)
})
```

    ## Unit: milliseconds
    ##   expr      min       lq     mean   median       uq      max neval
    ##    100    36.62    38.26    39.65    39.91    41.17    42.43     3
    ##    200    76.98    84.21    88.51    91.44    94.28    97.12     3
    ##    400   161.32   165.31   167.39   169.30   170.43   171.56     3
    ##    800   290.58   301.19   310.78   311.81   320.88   329.95     3
    ##   1600   530.80   578.54   598.78   626.28   632.77   639.25     3
    ##  51200 13169.70 13306.89 13389.39 13444.08 13499.23 13554.37     3

The `aprx_eps` controls the size of the error. To be precise about what this value does then we need to some notation for the complete likelihood (the likelihood where we observe ![\\vec\\beta\_1,\\dots,\\vec\\beta\_T](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%5Cbeta_1%2C%5Cdots%2C%5Cvec%5Cbeta_T "\vec\beta_1,\dots,\vec\beta_T")s). This is

![L = \\mu\_1(\\vec \\beta\_1)g\_1(\\vec y\_1 \\mid \\vec \\beta\_1)\\prod\_{t=2}^Tf(\\vec\\beta\_t \\mid\\vec\\beta\_{t-1})g\_t(y\_t\\mid\\beta\_t)](https://chart.googleapis.com/chart?cht=tx&chl=L%20%3D%20%5Cmu_1%28%5Cvec%20%5Cbeta_1%29g_1%28%5Cvec%20y_1%20%5Cmid%20%5Cvec%20%5Cbeta_1%29%5Cprod_%7Bt%3D2%7D%5ETf%28%5Cvec%5Cbeta_t%20%5Cmid%5Cvec%5Cbeta_%7Bt-1%7D%29g_t%28y_t%5Cmid%5Cbeta_t%29 "L = \mu_1(\vec \beta_1)g_1(\vec y_1 \mid \vec \beta_1)\prod_{t=2}^Tf(\vec\beta_t \mid\vec\beta_{t-1})g_t(y_t\mid\beta_t)")

where ![g\_t](https://chart.googleapis.com/chart?cht=tx&chl=g_t "g_t") is conditional distribution ![\\vec y\_t](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%20y_t "\vec y_t") given ![\\vec\\beta\_t](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%5Cbeta_t "\vec\beta_t"), ![f](https://chart.googleapis.com/chart?cht=tx&chl=f "f") is the conditional distribution of ![\\vec\\beta\_t](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%5Cbeta_t "\vec\beta_t") given ![\\vec\\beta\_{t-1}](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%5Cbeta_%7Bt-1%7D "\vec\beta_{t-1}"), and ![\\mu](https://chart.googleapis.com/chart?cht=tx&chl=%5Cmu "\mu") is the time-invariant distribution of ![\\vec\\beta\_t](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%5Cbeta_t "\vec\beta_t"). Let ![w\_t^{(j)}](https://chart.googleapis.com/chart?cht=tx&chl=w_t%5E%7B%28j%29%7D "w_t^{(j)}") be the weight of particle ![j](https://chart.googleapis.com/chart?cht=tx&chl=j "j") at time ![t](https://chart.googleapis.com/chart?cht=tx&chl=t "t") and ![\\vec \\beta\_t^{(j)}](https://chart.googleapis.com/chart?cht=tx&chl=%5Cvec%20%5Cbeta_t%5E%7B%28j%29%7D "\vec \beta_t^{(j)}") be the ![j](https://chart.googleapis.com/chart?cht=tx&chl=j "j")th particle at time ![t](https://chart.googleapis.com/chart?cht=tx&chl=t "t"). Then we ensure the error in our evaluation of terms ![w\_{t-1}^{(j)}f(\\vec\\beta\_t^{(i)} \\mid \\vec\\beta\_{t-1}^{(j)})](https://chart.googleapis.com/chart?cht=tx&chl=w_%7Bt-1%7D%5E%7B%28j%29%7Df%28%5Cvec%5Cbeta_t%5E%7B%28i%29%7D%20%5Cmid%20%5Cvec%5Cbeta_%7Bt-1%7D%5E%7B%28j%29%7D%29 "w_{t-1}^{(j)}f(\vec\beta_t^{(i)} \mid \vec\beta_{t-1}^{(j)})") never exceeds

![w\_{t-1} \\frac{u - l}{(u + l)/2}](https://chart.googleapis.com/chart?cht=tx&chl=w_%7Bt-1%7D%20%5Cfrac%7Bu%20-%20l%7D%7B%28u%20%2B%20l%29%2F2%7D "w_{t-1} \frac{u - l}{(u + l)/2}")

 where ![u](https://chart.googleapis.com/chart?cht=tx&chl=u "u") and ![l](https://chart.googleapis.com/chart?cht=tx&chl=l "l") are respectively an upper and lower bound of ![f(\\vec\\beta\_t^{(i)} \\mid \\vec\\beta\_{t-1}^{(j)})](https://chart.googleapis.com/chart?cht=tx&chl=f%28%5Cvec%5Cbeta_t%5E%7B%28i%29%7D%20%5Cmid%20%5Cvec%5Cbeta_%7Bt-1%7D%5E%7B%28j%29%7D%29 "f(\vec\beta_t^{(i)} \mid \vec\beta_{t-1}^{(j)})"). The question is how big the error is. Thus, we consider the error in the log-likelihood approximation at the true parameters.

``` r
ll_compare <- local({
  N_use <- 500L
  # we alter the seed in each run. First, the exact method
  ll_no_approx <- sapply(1:100, function(seed){
    ll_func <- mssm(
      fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat,
      random = ~ Z, ti = time_idx, control = mssm_control(
        n_threads = 5L, N_part = N_use, what = "log_density", 
        seed = seed))
    
    logLik(ll_func$pf_filter(
      cfix = cfix, disp = numeric(), F. = F., Q = Q))
  })
  
  # then the approximation
  ll_approx <- sapply(1:100, function(seed){
    ll_func <- mssm(
      fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat,
      random = ~ Z, ti = time_idx, control = mssm_control(
        n_threads = 5L, N_part = N_use, what = "log_density", 
        KD_N_max = 6L, aprx_eps = 1e-2, seed = seed, 
        which_ll_cp = "KD"))
    
    logLik(ll_func$pf_filter(
      cfix = cfix, disp = numeric(), F. = F., Q = Q))
  })
  
  list(ll_no_approx = ll_no_approx, ll_approx = ll_approx)
}) 
```

    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights

``` r
par(mar = c(5, 4, 1, 1))
hist(
  ll_compare$ll_no_approx, main = "", breaks = 20L, 
  xlab = "Log-likelihood approximation -- no aprox")
```

![](./README-fig/show_comp_arell_aprx-1.png)

``` r
hist(
  ll_compare$ll_approx   , main = "", breaks = 20L, 
  xlab = "Log-likelihood approximation -- aprox")
```

![](./README-fig/show_comp_arell_aprx-2.png)

The latter seems to have a small positive bias.

``` r
with(ll_compare, t.test(ll_no_approx, ll_approx))
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  ll_no_approx and ll_approx
    ## t = -1.9, df = 200, p-value = 0.06
    ## alternative hypothesis: true difference in means is not equal to 0
    ## 95 percent confidence interval:
    ##  -0.39189  0.01063
    ## sample estimates:
    ## mean of x mean of y 
    ##     -2008     -2008

The fact that it is small is nice because now we can get a much better approximation (in terms of variance) quickly of e.g., the log-likelihood as shown below.

``` r
ll_approx <- sapply(1:10, function(seed){
  N_use <- 100000L # many more particles
  
  ll_func <- mssm(
    fixed = y ~ X1 + X2 + Z, family = poisson(), data = dat,
    random = ~ Z, ti = time_idx, control = mssm_control(
      n_threads = 5L, N_part = N_use, what = "log_density", 
      KD_N_max = 100L, aprx_eps = 1e-2, seed = seed, 
      which_ll_cp = "KD"))
  
  logLik(ll_func$pf_filter(
    cfix = cfix, disp = numeric(), F. = F., Q = Q))
}) 
```

    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights
    ## Ignoring terms with -infinite normalized log weights

``` r
# approximate log-likelihood
sd(ll_approx)
```

    ## [1] 0.04254

``` r
mean(ll_approx)
```

    ## [1] -2008

``` r
# compare sd with 
sd(ll_compare$ll_no_approx)
```

    ## [1] 0.721

Fast Sum-Kernel Approximation
-----------------------------

This package contains a simple implementation of the dual-tree method like the one suggested by Gray and Moore (2003) and shown in Klaas et al. (2006). The problem we want to solve is the sum-kernel problem in Klaas et al. (2006). Particularly, we consider the situation where we have ![1,\\dots,N\_q](https://chart.googleapis.com/chart?cht=tx&chl=1%2C%5Cdots%2CN_q "1,\dots,N_q") query particles denoted by ![\\{\\vec Y\_i\\}\_{i=1,\\dots,N\_q}](https://chart.googleapis.com/chart?cht=tx&chl=%5C%7B%5Cvec%20Y_i%5C%7D_%7Bi%3D1%2C%5Cdots%2CN_q%7D "\{\vec Y_i\}_{i=1,\dots,N_q}") and ![1,\\dots,N\_s](https://chart.googleapis.com/chart?cht=tx&chl=1%2C%5Cdots%2CN_s "1,\dots,N_s") source particles denoted by ![\\{\\vec X\_j\\}\_{j=1,\\dots,N\_s}](https://chart.googleapis.com/chart?cht=tx&chl=%5C%7B%5Cvec%20X_j%5C%7D_%7Bj%3D1%2C%5Cdots%2CN_s%7D "\{\vec X_j\}_{j=1,\dots,N_s}"). For each query particle, we want to compute the weights

![W\_i = \\frac{\\tilde W\_i}{\\sum\_{k = 1}^{N\_q} \\tilde W\_i},\\qquad \\tilde W\_i = \\sum\_{j=1}^{N\_s} \\bar W\_j K(\\vec Y\_i, \\vec X\_j)](https://chart.googleapis.com/chart?cht=tx&chl=W_i%20%3D%20%5Cfrac%7B%5Ctilde%20W_i%7D%7B%5Csum_%7Bk%20%3D%201%7D%5E%7BN_q%7D%20%5Ctilde%20W_i%7D%2C%5Cqquad%20%5Ctilde%20W_i%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BN_s%7D%20%5Cbar%20W_j%20K%28%5Cvec%20Y_i%2C%20%5Cvec%20X_j%29 "W_i = \frac{\tilde W_i}{\sum_{k = 1}^{N_q} \tilde W_i},\qquad \tilde W_i = \sum_{j=1}^{N_s} \bar W_j K(\vec Y_i, \vec X_j)")

where each source particle has an associated weight ![\\bar W\_j](https://chart.googleapis.com/chart?cht=tx&chl=%5Cbar%20W_j "\bar W_j") and ![K](https://chart.googleapis.com/chart?cht=tx&chl=K "K") is a kernel. Computing the above is ![\\mathcal{O}(N\_sN\_q)](https://chart.googleapis.com/chart?cht=tx&chl=%5Cmathcal%7BO%7D%28N_sN_q%29 "\mathcal{O}(N_sN_q)") which is major bottleneck if ![N\_s](https://chart.googleapis.com/chart?cht=tx&chl=N_s "N_s") and ![N\_q](https://chart.googleapis.com/chart?cht=tx&chl=N_q "N_q") is large. However, one can use a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) for the query particles and source particles and exploit that:

-   ![W\_j K(\\vec Y\_i, \\vec X\_j)](https://chart.googleapis.com/chart?cht=tx&chl=W_j%20K%28%5Cvec%20Y_i%2C%20%5Cvec%20X_j%29 "W_j K(\vec Y_i, \vec X_j)") is almost zero for some pairs of nodes in the two k-d trees.
-   ![K(\\cdot, \\vec X\_j)](https://chart.googleapis.com/chart?cht=tx&chl=K%28%5Ccdot%2C%20%5Cvec%20X_j%29 "K(\cdot, \vec X_j)") is almost identical for some nodes in the k-d tree for the source particles.

Thus, a substantial amount of computation can be skipped or approximated with e.g., the centroid in the source node with only a minor loss of precision. The dual-tree approximation method is ![\\mathcal{O}(N\_s\\log N\_s)](https://chart.googleapis.com/chart?cht=tx&chl=%5Cmathcal%7BO%7D%28N_s%5Clog%20N_s%29 "\mathcal{O}(N_s\log N_s)") and ![\\mathcal{O}(N\_q\\log N\_q)](https://chart.googleapis.com/chart?cht=tx&chl=%5Cmathcal%7BO%7D%28N_q%5Clog%20N_q%29 "\mathcal{O}(N_q\log N_q)"). We start by defining a function to simulate the source and query particles (we will let the two sets be identical for simplicity). Further, we plot one draw of simulated points and illustrate the leafs in the k-d tree.

``` r
######
# define function to simulate data
mus <- matrix(c(-1, -1, 
                 1, 1, 
                -1, 1), 3L, byrow = FALSE)
mus <- mus * .75
Sig <- diag(c(.5^2, .25^2))

get_sims <- function(n_per_grp){
  # simulate X
  sims <- lapply(1:nrow(mus), function(i){
    mu <- mus[i, ]
    X <- matrix(rnorm(n_per_grp * 2L), nrow = 2L)
    X <- t(crossprod(chol(Sig), X) + mu)
    
    data.frame(X, grp = i)
  })
  sims <- do.call(rbind, sims)
  X <- t(as.matrix(sims[, c("X1", "X2")]))
  
  # simulate weights
  ws <- exp(rnorm(ncol(X)))
  ws <- ws / sum(ws)
  
  list(sims = sims, X = X, ws = ws)
}

#####
# show example 
set.seed(42452654)
invisible(list2env(get_sims(5000L), environment()))

# plot points
par(mar = c(5, 4, 1, 1))
plot(as.matrix(sims[, c("X1", "X2")]), col = sims$grp + 1L)

# find k-d tree and add borders 
out <- mssm:::test_KD_note(X, 50L)
out$indices <- out$indices + 1L
n_ele <- drop(out$n_elems)
idx <- mapply(`:`, cumsum(c(1L, head(n_ele, -1))), cumsum(n_ele))
stopifnot(all(sapply(idx, length) == n_ele))
idx <- lapply(idx, function(i) out$indices[i])
stopifnot(!anyDuplicated(unlist(idx)), length(unlist(idx)) == ncol(X))

grps <- lapply(idx, function(i) X[, i])

borders <- lapply(grps, function(x) apply(x, 1, range))
invisible(lapply(borders, function(b) 
  rect(b[1, "X1"], b[1, "X2"], b[2, "X1"], b[2, "X2"])))
```

![](./README-fig/sim_func-1.png)

Next, we compute the run-times for the previous examples and compare the approximations of the un-normalized log weights, ![\\log \\tilde W\_i](https://chart.googleapis.com/chart?cht=tx&chl=%5Clog%20%5Ctilde%20W_i "\log \tilde W_i"), and normalized weights, ![W\_i](https://chart.googleapis.com/chart?cht=tx&chl=W_i "W_i"). The `n_threads` sets the number of threads to use in the methods.

``` r
# run-times
microbenchmark::microbenchmark(
  `dual tree 1` = mssm:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, 
                               eps = 5e-3, n_threads = 1L),
  `dual tree 6` = mssm:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, 
                               eps = 5e-3, n_threads = 4L),
  `naive 1`     = mssm:::naive(X = X, ws = ws, Y = X, n_threads = 1L),
  `naive 6`     = mssm:::naive(X = X, ws = ws, Y = X, n_threads = 4L),
  times = 10L)
```

    ## Unit: milliseconds
    ##         expr     min      lq   mean  median      uq     max neval
    ##  dual tree 1  116.98  118.34  121.5  118.93  121.73  135.52    10
    ##  dual tree 6   41.45   41.82   43.4   41.96   43.52   51.77    10
    ##      naive 1 3318.78 3335.22 3409.1 3382.32 3509.36 3546.91    10
    ##      naive 6  873.27  935.32  965.1  967.98  992.91 1042.07    10

``` r
# The functions return the un-normalized log weights. We first compare
# the result on this scale
o1 <- mssm:::FSKA  (X = X, ws = ws, Y = X, N_min = 10L, eps = 5e-3, 
                    n_threads = 1L)
o2 <- mssm:::naive(X = X, ws = ws, Y = X, n_threads = 4L)

all.equal(o1, o2)
```

    ## [1] "Mean relative difference: 0.0015"

``` r
par(mar = c(5, 4, 1, 1))
hist((o1 - o2)/ abs((o1 + o2) / 2), breaks = 50, main = "", 
     xlab = "Delta un-normalized log weights")
```

![](./README-fig/comp_run_times-1.png)

``` r
# then we compare the normalized weights
func <- function(x){
  x_max <- max(x)
  x <- exp(x - x_max)
  x / sum(x)
}

o1 <- func(o1)
o2 <- func(o2)
all.equal(o1, o2)
```

    ## [1] "Mean relative difference: 0.0008531"

``` r
hist((o1 - o2)/ abs((o1 + o2) / 2), breaks = 50, main = "", 
     xlab = "Delta normalized log weights")
```

![](./README-fig/comp_run_times-2.png)

Finally, we compare the run-times as function of ![N = N\_s = N\_q](https://chart.googleapis.com/chart?cht=tx&chl=N%20%3D%20N_s%20%3D%20N_q "N = N_s = N_q"). The dashed line is "naive" method, the continuous line is the dual-tree method, and the dotted line is dual-tree method using 1 thread.

``` r
Ns <- 2^(7:14)
run_times <- lapply(Ns, function(N){
  invisible(list2env(get_sims(N), environment()))
  microbenchmark::microbenchmark(
    `dual-tree`   = mssm:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, eps = 5e-3, 
                                 n_threads = 4L),
    naive         = mssm:::naive(X = X, ws = ws, Y = X, n_threads = 4L),
    `dual-tree 1` = mssm:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, eps = 5e-3, 
                                 n_threads = 1L), 
    times = 5L)
})

Ns_xtra <- 2^(15:19)
run_times_xtra <- lapply(Ns_xtra, function(N){
  invisible(list2env(get_sims(N), environment()))
  microbenchmark::microbenchmark(
    `dual-tree` = mssm:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, eps = 5e-3, 
                               n_threads = 4L),
    times = 5L)
})
```

``` r
library(microbenchmark)
meds <- t(sapply(run_times, function(x) summary(x, unit = "s")[, "median"]))
meds_xtra <- 
  sapply(run_times_xtra, function(x) summary(x, unit = "s")[, "median"])
meds <- rbind(meds, cbind(meds_xtra, NA_real_, NA_real_))
dimnames(meds) <- list(
  N = c(Ns, Ns_xtra) * 3L, method = c("Dual-tree", "Naive", "Dual-tree 1"))
meds
```

    ##          method
    ## N         Dual-tree      Naive Dual-tree 1
    ##   384      0.001173  0.0007221    0.003361
    ##   768      0.002625  0.0026589    0.006639
    ##   1536     0.004571  0.0095613    0.012272
    ##   3072     0.008974  0.0377994    0.024477
    ##   6144     0.021037  0.1899465    0.053030
    ##   12288    0.038194  0.6819931    0.105499
    ##   24576    0.078382  2.9246776    0.193367
    ##   49152    0.124139 10.9888913    0.375339
    ##   98304    0.246900         NA          NA
    ##   196608   0.561036         NA          NA
    ##   393216   1.050689         NA          NA
    ##   786432   2.086806         NA          NA
    ##   1572864  4.452460         NA          NA

``` r
par(mar = c(5, 4, 1, 1))
matplot(c(Ns, Ns_xtra) * 3L, meds, lty = 1:3, type = "l", log = "xy", 
        ylab = "seconds", xlab = "N", col = "black")
```

![](./README-fig/plot_run_times_N-1.png)

Function Definitions
--------------------

``` r
# Stochastic gradient descent for mssm object. The function assumes that the 
# state vector is stationary. The default values are rather arbitrary.
# 
# Args:
#   object: an object of class mssmFunc. 
#   n_it: number of iterations. 
#   lrs: learning rates to use. Must have n_it elements. Like problem specific. 
#   avg_start: index to start averaging. See Polyak et al. (1992) for arguments
#              for averaging.
#   cfix: starting values for fixed coefficients. 
#   F.: starting value for transition matrix in conditional distribution of the 
#       current state given the previous state.
#   Q: starting value for covariance matrix in conditional distribution of the 
#      current state given the previous. 
#   verbose: TRUE if output should be printed doing estmation. 
# 
# Returns:
#   List with estimates and the log-likelihood approximation at each iteration. 
sgd <- function(
  object, n_it = 150L, 
  lrs = 1e-2 * (1:n_it)^(-1/2), avg_start = max(1L, as.integer(n_it * 4L / 5L)),
  cfix, F., Q,  verbose = FALSE)
{
  # make checks
  stopifnot(
    inherits(object, "mssmFunc"), object$control$what == "gradient", n_it > 0L, 
    all(lrs > 0.), avg_start > 1L, length(lrs) == n_it)
  n_fix <- nrow(object$X)
  n_rng <- nrow(object$Z)
  
  # objects for estimates at each iteration and log-likelihood approximations
  ests <- matrix(
    NA_real_, n_it + 1L, n_fix + n_rng * n_rng + n_rng * (n_rng  + 1L) / 2L)
  ests[1L, ] <- c(cfix, F., Q[lower.tri(Q, diag = TRUE)])
  lls <- rep(NA_real_, n_it)
  
  # indices of the different components
  idx_fix <- 1:n_fix
  idx_F   <- 1:(n_rng * n_rng) + n_fix
  idx_Q   <- 1:(n_rng * (n_rng  + 1L) / 2L) + n_fix + n_rng * n_rng
  
  # we only want the lower part of `Q` so we make the following map for the 
  # gradient
  library(matrixcalc) # TODO: get rid of this
  gr_map <- matrix(
    0., nrow = ncol(ests), ncol = length(cfix) + length(F.) + length(Q))
  gr_map[idx_fix, idx_fix] <- diag(length(idx_fix))
  gr_map[idx_F  , idx_F] <- diag(length(idx_F))
  dup_mat <- duplication.matrix(ncol(Q))
  gr_map[idx_Q  , -c(idx_fix, idx_F)] <- t(dup_mat)
  
  # function to set the parameters
  set_parems <- function(i){
    # select whether or not to average
    idx <- if(i > avg_start) avg_start:i else i
    
    # set new parameters
    cfix <<-             colMeans(ests[idx, idx_fix, drop = FALSE])
    F.[] <<-             colMeans(ests[idx, idx_F  , drop = FALSE])
    Q[]  <<- dup_mat %*% colMeans(ests[idx, idx_Q  , drop = FALSE])
    
  }
    
  # run gradient decent 
  max_half <- 25L
  for(i in 1:n_it + 1L){
    # get gradient. First, run the particle filter
    filter_out <- object$pf_filter(
      cfix = cfix, disp = numeric(), F. = F., Q = Q)
    lls[i - 1L] <- c(logLik(filter_out))
    
    # then get the gradient associated with each particle and the log 
    # normalized weight of the particles
    grads <- tail(filter_out$pf_output, 1L)[[1L]]$stats
    ws    <- tail(filter_out$pf_output, 1L)[[1L]]$ws_normalized
    
    # compute the gradient and take a small step
    grad <- colSums(t(grads) * drop(exp(ws)))
    lr_i <- lrs[i - 1L]
    k <- 0L
    while(k < max_half){
      ests[i, ] <- ests[i - 1L, ] + lr_i * gr_map %*% grad 
      set_parems(i)
      
      # check that Q is positive definite and the system is stationary
      c1 <- all(abs(eigen(F.)$values) < 1)
      c2 <- all(eigen(Q)$values > 0)
      if(c1 && c2)
        break
      
      # decrease learning rate
      lr_i <- lr_i * .5
      k <- k + 1L
    }
    
    # check if we failed to find a value within our constraints
    if(k == max_half)
      stop("failed to find solution within constraints")
    
    # print information if requested 
    if(verbose){
      cat(sprintf(
        "\nIt %5d: log-likelihood (current, max) %12.2f, %12.2f\n", 
        i - 1L, logLik(filter_out), max(lls, na.rm = TRUE)), 
          rep("-", 66), "\n", sep = "")
      cat("cfix\n")
      print(cfix)
      cat("F\n")
      print(F.)
      cat("Q\n")
      print(Q)
      
    }
  } 
  
  list(estimates = ests, logLik = lls, F. = F., Q = Q, cfix = cfix)
}
```

References
==========

Gray, Alexander G., and Andrew W. Moore. 2003. “Rapid Evaluation of Multiple Density Models.” In *AISTATS*.

Klaas, Mike, Mark Briers, Nando de Freitas, Arnaud Doucet, Simon Maskell, and Dustin Lang. 2006. “Fast Particle Smoothing: If I Had a Million Particles.” In *Proceedings of the 23rd International Conference on Machine Learning*, 481–88. ICML ’06. New York, NY, USA: ACM. <http://doi.acm.org/10.1145/1143844.1143905>.

Lin, Ming T, Junni L Zhang, Qiansheng Cheng, and Rong Chen. 2005. “Independent Particle Filters.” *Journal of the American Statistical Association* 100 (472). Taylor & Francis: 1412–21. doi:[10.1198/016214505000000349](https://doi.org/10.1198/016214505000000349).

Polyak, B., and A. Juditsky. 1992. “Acceleration of Stochastic Approximation by Averaging.” *SIAM Journal on Control and Optimization* 30 (4): 838–55. doi:[10.1137/0330046](https://doi.org/10.1137/0330046).

Poyiadjis, George, Arnaud Doucet, and Sumeetpal S. Singh. 2011. “Particle Approximations of the Score and Observed Information Matrix in State Space Models with Application to Parameter Estimation.” *Biometrika* 98 (1). Biometrika Trust: 65–80. <http://www.jstor.org/stable/29777165>.
