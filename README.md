
Fast Sum-Kernel Approximation
-----------------------------

This package contains a simple implementation of the dual-tree method like the one suggested by Gray and Moore (2003) and shown in Klaas et al. (2006). The problem we want to solve is the sum-kernel problem in Klaas et al. (2006). Particularly, we consider the situation where we have ![1,\\dots,N\_q](https://chart.googleapis.com/chart?cht=tx&chl=1%2C%5Cdots%2CN_q "1,\dots,N_q") query particles denoted by ![\\{\\vec Y\_i\\}\_{i=1,\\dots,N\_q}](https://chart.googleapis.com/chart?cht=tx&chl=%5C%7B%5Cvec%20Y_i%5C%7D_%7Bi%3D1%2C%5Cdots%2CN_q%7D "\{\vec Y_i\}_{i=1,\dots,N_q}") and ![1,\\dots,N\_s](https://chart.googleapis.com/chart?cht=tx&chl=1%2C%5Cdots%2CN_s "1,\dots,N_s") source particles denoted by ![\\{\\vec X\_j\\}\_{j=1,\\dots,N\_s}](https://chart.googleapis.com/chart?cht=tx&chl=%5C%7B%5Cvec%20X_j%5C%7D_%7Bj%3D1%2C%5Cdots%2CN_s%7D "\{\vec X_j\}_{j=1,\dots,N_s}"). For each query particle, we want to compute the weights

![W\_i = \\frac{\\tilde W\_i}{\\sum\_{k = 1}^{N\_q} \\tilde W\_i},\\qquad \\tilde W\_i = \\sum\_{j=1}^{N\_s} \\bar W\_j K(\\vec Y\_i, \\vec X\_j)](https://chart.googleapis.com/chart?cht=tx&chl=W_i%20%3D%20%5Cfrac%7B%5Ctilde%20W_i%7D%7B%5Csum_%7Bk%20%3D%201%7D%5E%7BN_q%7D%20%5Ctilde%20W_i%7D%2C%5Cqquad%20%5Ctilde%20W_i%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BN_s%7D%20%5Cbar%20W_j%20K%28%5Cvec%20Y_i%2C%20%5Cvec%20X_j%29 "W_i = \frac{\tilde W_i}{\sum_{k = 1}^{N_q} \tilde W_i},\qquad \tilde W_i = \sum_{j=1}^{N_s} \bar W_j K(\vec Y_i, \vec X_j)")

where each source particle has an associated weight ![\\bar W\_j](https://chart.googleapis.com/chart?cht=tx&chl=%5Cbar%20W_j "\bar W_j") and ![K](https://chart.googleapis.com/chart?cht=tx&chl=K "K") is kernel. Computing the above is ![\\mathcal{O}(N\_sN\_q)](https://chart.googleapis.com/chart?cht=tx&chl=%5Cmathcal%7BO%7D%28N_sN_q%29 "\mathcal{O}(N_sN_q)") which is major bottleneck if ![N\_s](https://chart.googleapis.com/chart?cht=tx&chl=N_s "N_s") and ![N\_q](https://chart.googleapis.com/chart?cht=tx&chl=N_q "N_q") is large. However, one can use a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) for the query particles and source particles and exploit that:

-   ![W\_j K(\\vec Y\_i, \\vec X\_j)](https://chart.googleapis.com/chart?cht=tx&chl=W_j%20K%28%5Cvec%20Y_i%2C%20%5Cvec%20X_j%29 "W_j K(\vec Y_i, \vec X_j)") is almost zero for some pairs of nodes in the two k-d trees.
-   ![K(\\cdot, \\vec X\_j)](https://chart.googleapis.com/chart?cht=tx&chl=K%28%5Ccdot%2C%20%5Cvec%20X_j%29 "K(\cdot, \vec X_j)") is almost identical for some nodes in the k-d tree for the source particles.

Thus, a substantial amount of computation can be skipped or approximated with e.g., the centroid in the source node with only a minor loss of precision. We start by defining a function to simulate the source and query particles (we will let the two sets be identical for simplicity). Further, we plot one draw of simulated points and illustrate the leafs in the k-d tree.

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
par(mar = c(5, 4, .5, .5))
plot(as.matrix(sims[, c("X1", "X2")]), col = sims$grp + 1L)

# find KD-tree and add borders 
out <- FSKA:::test_KD_note(X, 50L)
out$indices <- out$indices + 1L
n_ele <- drop(out$n_elems)
idx <- mapply(`:`, cumsum(c(1L, head(n_ele, -1))), cumsum(n_ele))
stopifnot(all(sapply(idx, length) == n_ele))
idx <- lapply(idx, function(i) out$indices[i])
stopifnot(!anyDuplicated(unlist(idx)))

grps <- lapply(idx, function(i) X[, i])

borders <- lapply(grps, function(x) apply(x, 1, range))
invisible(lapply(borders, function(b) 
  rect(b[1, "X1"], b[1, "X2"], b[2, "X1"], b[2, "X2"])))
```

![](./README-fig/sim_func-1.png)

Next, we compute the run-times for the previous examples and compare the approximations of the un-normalized log weights, ![\\log \\tilde W\_i](https://chart.googleapis.com/chart?cht=tx&chl=%5Clog%20%5Ctilde%20W_i "\log \tilde W_i"), and normalized weights, ![W\_i](https://chart.googleapis.com/chart?cht=tx&chl=W_i "W_i"). The `n_threads` sets the number of threads to use in the approximation method (one could easily implement the naive method support parallel computation).

``` r
# run-times
microbenchmark::microbenchmark(
  `dual tree 1` = FSKA:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, 
                               eps = 1e-3, n_threads = 1L),
  `dual tree 6` = FSKA:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, 
                               eps = 1e-3, n_threads = 6L),
  naive         = FSKA:::naive(X = X, ws = ws, Y = X), times = 10L)
```

    ## Unit: milliseconds
    ##         expr    min     lq   mean median     uq    max neval
    ##  dual tree 1  530.2  532.2  541.5  537.0  552.4  560.8    10
    ##  dual tree 6  360.5  477.9  620.7  675.3  714.4  796.4    10
    ##        naive 3089.3 3147.7 3239.8 3272.5 3320.0 3360.2    10

``` r
# The functions return the un-normalized log weights. We first compare
# the result on this scale
o1 <- FSKA:::FSKA  (X = X, ws = ws, Y = X, N_min = 10L, eps = 1e-3, 
                    n_threads = 1L)
o2 <- FSKA:::naive(X = X, ws = ws, Y = X)

all.equal(o1, o2)
```

    ## [1] "Mean relative difference: 0.0006629"

``` r
par(mar = c(5, 4, .5, .5))
hist(o1 - o2, breaks = 25, main = "", 
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

    ## [1] "Mean relative difference: 0.0005602"

``` r
hist(o1 - o2, breaks = 25, main = "", xlab = "Delta normalized weights")
```

![](./README-fig/comp_run_times-2.png)

Finally, we compare the run-times as function of ![N = N\_s = N\_q](https://chart.googleapis.com/chart?cht=tx&chl=N%20%3D%20N_s%20%3D%20N_q "N = N_s = N_q"). The dashed line is "naive" method and the continuous line is the dual-tree method.

``` r
Ns <- 2^(7:15)
run_times <- lapply(Ns, function(N){
  invisible(list2env(get_sims(N), environment()))
  microbenchmark::microbenchmark(
    `dual tree` = FSKA:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, eps = 1e-3, 
                               n_threads = 6L),
    naive       = FSKA:::naive(X = X, ws = ws, Y = X), times = 3L)
})

Ns_xtra <- 2^(16:20)
run_times_xtra <- lapply(Ns_xtra, function(N){
  invisible(list2env(get_sims(N), environment()))
  microbenchmark::microbenchmark(
    `dual tree` = FSKA:::FSKA (X = X, ws = ws, Y = X, N_min = 10L, eps = 1e-3, 
                               n_threads = 6L),
    times = 3L)
})
```

``` r
library(microbenchmark)
meds <- t(sapply(run_times, function(x) summary(x, unit = "s")[, "median"]))
meds_xtra <- 
  sapply(run_times_xtra, function(x) summary(x, unit = "s")[, "median"])
meds <- rbind(meds, cbind(meds_xtra, NA_real_))
dimnames(meds) <- list(
  N = c(Ns, Ns_xtra) * 3L, method = c("Dual-tree", "Naive"))
meds
```

    ##          method
    ## N         Dual-tree      Naive
    ##   384       0.01427   0.002136
    ##   768       0.05601   0.008391
    ##   1536      0.19884   0.033368
    ##   3072      0.55536   0.142153
    ##   6144      0.86816   0.535374
    ##   12288     0.90936   2.094292
    ##   24576     0.35816   8.327800
    ##   49152     0.52883  33.389924
    ##   98304     0.94832 133.103673
    ##   196608    1.82977         NA
    ##   393216    3.81140         NA
    ##   786432    7.79603         NA
    ##   1572864  18.98285         NA
    ##   3145728  52.43355         NA

``` r
par(mar = c(5, 4, .5, .5))
matplot(c(Ns, Ns_xtra) * 3L, meds, lty = 1:2, type = "l", log = "xy", 
        ylab = "seconds", xlab = "N", col = "black")
```

![](./README-fig/plot_run_times_N-1.png)

References
==========

Gray, Alexander G., and Andrew W. Moore. 2003. “Rapid Evaluation of Multiple Density Models.” In *AISTATS*.

Klaas, Mike, Mark Briers, Nando de Freitas, Arnaud Doucet, Simon Maskell, and Dustin Lang. 2006. “Fast Particle Smoothing: If I Had a Million Particles.” In *Proceedings of the 23rd International Conference on Machine Learning*, 481–88. ICML ’06. New York, NY, USA: ACM. <http://doi.acm.org/10.1145/1143844.1143905>.
