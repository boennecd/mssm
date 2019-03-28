#include "fast-kernal-approx.h"
#include <limits>
#include <math.h>
#include <cmath>
#include "kernals.h"
#include "thread_pool.h"

#ifdef FSKA_PROF
#include <gperftools/profiler.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#endif

void comp_weights(
    arma::vec&, const source_node&, const query_node&, const arma::mat&,
    const arma::vec&, const arma::mat&, const double, const mvariate&,
    thread_pool&, std::vector<std::future<void> >&);

// [[Rcpp::export]]
arma::vec FSKA(
    const arma::mat &X, const arma::vec &ws, const arma::mat &Y,
    const arma::uword N_min, const double eps,
    const unsigned int n_threads){
#ifdef FSKA_PROF
  std::stringstream ss;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  ss << std::put_time(&tm, "profile-FSKA-%d-%m-%Y-%H-%M-%S.log");
  Rcpp::Rcout << "Saving profile output to '" << ss.str() << "'" << std::endl;
  const std::string s = ss.str();
  ProfilerStart(s.c_str());
#endif

  KD_note X_root = get_KD_tree(X, N_min);
  source_node X_root_source(X, ws, X_root);

  KD_note Y_root = get_KD_tree(Y, N_min);
  query_node Y_root_query(Y, Y_root);

  arma::vec ws_log = arma::log(ws);

  mvariate kernal(X.n_rows);
  thread_pool pool(n_threads);
  std::vector<std::future<void> > futures;

  arma::vec log_weights(Y.n_cols);
  log_weights.fill(std::numeric_limits<double>::quiet_NaN());
  comp_weights(log_weights, X_root_source, Y_root_query, X, ws_log, Y, eps,
               kernal, pool, futures);

  while(!futures.empty()){
    futures.back().get();
    futures.pop_back();
  }

#ifdef FSKA_PROF
  ProfilerStop();
#endif

  return log_weights;
}

struct comp_w_centroid {
  arma::vec &log_weights;
  const source_node &X_node;
  const query_node &Y_node;
  const arma::mat &Y;
  const mvariate &kernal;
  const bool is_single_threaded;

  comp_w_centroid(
    arma::vec &log_weights, const source_node &X_node,
    const query_node &Y_node, const arma::mat &Y, const mvariate &kernal,
    const bool is_single_threaded):
  log_weights(log_weights), X_node(X_node), Y_node(Y_node), Y(Y),
  kernal(kernal), is_single_threaded(is_single_threaded) { }

  void operator()(){
    if(!Y_node.node.is_leaf()){
      comp_w_centroid(log_weights, X_node, *Y_node.left , Y, kernal,
                      is_single_threaded)();
      comp_w_centroid(log_weights, X_node, *Y_node.right, Y, kernal,
                      is_single_threaded)();
      return;
    }

    const std::vector<arma::uword> &idx = Y_node.node.get_indices();
    const arma::vec &X_centroid = X_node.centroid;
    double x_weight_log = std::log(X_node.weight);
    const double *xp = X_centroid.begin();
    const arma::uword N = X_centroid.n_elem;

    arma::vec out;
    double *o = nullptr;
    if(!is_single_threaded){
      out.set_size(idx.size());
      o = out.begin();
    }
    for(auto i : idx){
      double dist = norm_square(xp, Y.colptr(i), N);
      double new_term = kernal(dist, true) + x_weight_log;
      if(!std::isnan(log_weights[i]))
        new_term = log_sum_log(log_weights[i], new_term);
      if(is_single_threaded){
        log_weights[i]  = new_term;
        continue;

      }

      *(o++) = new_term;
    }

    if(is_single_threaded)
      return;

    o = out.begin();
    std::lock_guard<std::mutex> guard(*Y_node.idx_mutex);
    for(auto i : idx)
      log_weights[i] = *(o++);
  }

};

struct comp_all {
  arma::vec &log_weights;
  const source_node &X_node;
  const query_node &Y_node;
  const arma::mat &X;
  const arma::vec &ws_log;
  const arma::mat &Y;
  const mvariate &kernal;
  const bool is_single_threaded;

  comp_all(
    arma::vec &log_weights, const source_node &X_node,
    const query_node &Y_node, const arma::mat &X, const arma::vec &ws_log,
    const arma::mat &Y, const mvariate &kernal, const bool is_single_threaded):
    log_weights(log_weights), X_node(X_node), Y_node(Y_node), X(X),
    ws_log(ws_log), Y(Y), kernal(kernal),
    is_single_threaded(is_single_threaded) { }

  void operator()(){
#ifdef FSKA_DEBUG
    if(!X_node.node.is_leaf() or !Y_node.node.is_leaf())
      throw "comp_all called with non-leafs";
#endif

    const std::vector<arma::uword> &idx_y = Y_node.node.get_indices(),
      &idx_x = X_node.node.get_indices();
    arma::vec x_y_ws(idx_x.size());

    arma::vec out;
    double *o = nullptr;
    if(!is_single_threaded){
      out.set_size(idx_x.size());
      o = out.begin();
    }
    for(auto i_y : idx_y){
      const arma::uword N = Y.n_rows;
      double max_log_w = std::numeric_limits<double>::lowest();
      double *x_y_ws_i = x_y_ws.begin();
      for(auto i_x : idx_x){
        double dist = norm_square(X.colptr(i_x), Y.colptr(i_y), N);
        *x_y_ws_i = ws_log[i_x] + kernal(dist, true);
        if(*x_y_ws_i > max_log_w)
          max_log_w = *x_y_ws_i;

        x_y_ws_i++;
      }
      double new_term = log_sum_log(x_y_ws, max_log_w);
      if(!std::isnan(log_weights[i_y]))
        new_term = log_sum_log(log_weights[i_y], new_term);
      if(is_single_threaded){
          log_weights[i_y] = new_term;
        continue;

      }

      *(o++) = new_term;
    }

    if(is_single_threaded)
      return;

    o = out.begin();
    std::lock_guard<std::mutex> guard(*Y_node.idx_mutex);
    for(auto i_y : idx_y)
      log_weights[i_y] = *(o++);
  }
};


/*   comp_weights(log_weights, X_root_source, Y_root_query, X, ws_log, Y, eps,
 kernal, pool, futures); */

void comp_weights(
    arma::vec &log_weights, const source_node &X_node,
    const query_node &Y_node, const arma::mat &X,
    const arma::vec &ws_log, const arma::mat &Y, const double eps,
    const mvariate &kernal, thread_pool &pool,
    std::vector<std::future<void> > &futures)
  {
    auto dists = Y_node.borders.min_max_dist(X_node.borders);
    double k_min = kernal(dists[1], false), k_max = kernal(dists[0], false);
    if(X_node.weight *
       (k_max - k_min) / ((k_max + k_min) / 2. + 1e-16) < 2. * eps){
      futures.push_back(
        pool.submit(comp_w_centroid(
            log_weights, X_node, Y_node, Y, kernal,
            pool.thread_count < 2L)));

      return;
    }

    if(X_node.node.is_leaf() and Y_node.node.is_leaf()){
      futures.push_back(
        pool.submit(comp_all(
          log_weights, X_node, Y_node, X, ws_log, Y, kernal,
          pool.thread_count < 2L)));
      return;
    }

    if(!X_node.node.is_leaf() and  Y_node.node.is_leaf()){
      comp_weights(
        log_weights, *X_node.left ,  Y_node,
        X, ws_log, Y, eps, kernal, pool, futures);
      comp_weights(
        log_weights, *X_node.right,  Y_node,
        X, ws_log, Y, eps, kernal, pool, futures);
      return;
    }
    if( X_node.node.is_leaf() and !Y_node.node.is_leaf()){
      comp_weights(
        log_weights,  X_node     , *Y_node.left,
        X, ws_log, Y, eps, kernal, pool, futures);
      comp_weights(
        log_weights,  X_node     , *Y_node.right,
        X, ws_log, Y, eps, kernal, pool, futures);
      return;
    }

    comp_weights(
      log_weights, *X_node.left , *Y_node.left ,
      X, ws_log, Y, eps, kernal, pool, futures);
    comp_weights(
      log_weights, *X_node.left , *Y_node.right,
      X, ws_log, Y, eps, kernal, pool, futures);
    comp_weights(
      log_weights, *X_node.right, *Y_node.left ,
      X, ws_log, Y, eps, kernal, pool, futures);
    comp_weights(
      log_weights, *X_node.right, *Y_node.right,
      X, ws_log, Y, eps, kernal, pool, futures);
  }



inline std::unique_ptr<const source_node> set_child
  (const arma::mat &X, const arma::vec &ws, const KD_note &node,
   const bool is_left)
  {
    if(node.is_leaf())
      return std::unique_ptr<source_node>();
    if(is_left)
      return std::unique_ptr<source_node>(
        new source_node(X, ws, node.get_left ()));
    return std::unique_ptr<source_node>(
        new source_node(X, ws, node.get_right()));
  }

arma::vec set_centroid
  (const source_node &snode, const arma::mat &X, const arma::vec &ws)
  {
    if(snode.node.is_leaf()){
      arma::vec centroid(X.n_rows, arma::fill::zeros);
      const auto &indices = snode.node.get_indices();
      double sum_w = 0.;
      for(auto idx : indices){
        centroid += ws[idx] * X.unsafe_col(idx);
        sum_w += ws[idx];
      }
      centroid /= sum_w;

      return centroid;
    }

    double w1 = snode.left->weight, w2 = snode.right->weight;
    return
      (w1 / (w1 + w2)) * snode.left->centroid +
      (w2 / (w1 + w2)) * snode.right->centroid;
  }

inline double set_weight(const source_node &snode, const arma::vec &ws)
  {
    if(snode.node.is_leaf()){
      const auto &indices = snode.node.get_indices();
      double weight = 0.;
      for(auto idx : indices)
        weight += ws[idx];

      return weight;
    }

    return snode.left->weight + snode.right->weight;
  }

template<class T>
hyper_rectangle set_borders(const T &snode, const arma::mat &X){
  if(snode.node.is_leaf())
    return hyper_rectangle(X, snode.node.get_indices());

  return hyper_rectangle(snode.left->borders, snode.right->borders);
}

source_node::source_node
  (const arma::mat &X, const arma::vec &ws, const KD_note &node):
  node(node), left(set_child(X, ws, node, true)),
  right(set_child(X, ws, node, false)), centroid(set_centroid(*this, X, ws)),
  weight(set_weight(*this, ws)), borders(set_borders(*this, X)) { }



inline std::unique_ptr<const query_node> set_child_query
  (const arma::mat &X, const KD_note &node, const bool is_left)
{
  if(node.is_leaf())
    return std::unique_ptr<query_node>();
  if(is_left)
    return std::unique_ptr<query_node>(
      new query_node(X, node.get_left ()));
  return std::unique_ptr<query_node>(
    new query_node(X, node.get_right()));
}

query_node::query_node(const arma::mat &Y, const KD_note &node):
  node(node), left(set_child_query(Y, node, true)),
  right(set_child_query(Y, node, false)), borders(set_borders(*this, Y)),
  idx_mutex(new std::mutex()) { }



hyper_rectangle::hyper_rectangle(const arma::mat &X, const arma::uvec &idx)
{
  const arma::uword N = X.n_rows;
  borders.set_size(2L, N);
  borders.row(0L).fill(std::numeric_limits<double>::max());
  borders.row(1L).fill(std::numeric_limits<double>::lowest());

  for(auto i : idx){
    const double *x = X.colptr(i);
    for(unsigned int j = 0; j < N; ++j, ++x){
      if(*x < borders(0L, j))
        borders(0L, j) = *x;
      if(*x >= borders(1L, j))
        borders(1L, j) = *x;
    }
  }
}

hyper_rectangle::hyper_rectangle
  (const hyper_rectangle &r1, const hyper_rectangle &r2)
  {
#ifdef FSKA_DEBUG
    if(r1.borders.n_rows != r2.borders.n_rows or
         r1.borders.n_cols != r2.borders.n_cols)
      throw "dimension do not match";
#endif
    const arma::uword N = r1.borders.n_cols;
    borders.set_size(2L, N);

    double *b = borders.begin();
    const double *x1 = r1.borders.begin(), *x2 = r2.borders.begin();
    for(unsigned int i = 0; i < 2L * N; ++i, ++b, ++x1, ++x2)
      if(i % 2L == 0L)
        *b = std::min(*x1, *x2);
      else
        *b = std::max(*x1, *x2);
  }


std::array<double, 2> hyper_rectangle::min_max_dist
  (const hyper_rectangle &other) const
  {
#ifdef FSKA_DEBUG
  if(this->borders.n_rows != other.borders.n_rows or
       this->borders.n_cols != other.borders.n_cols)
    throw "dimension do not match";
#endif
    std::array<double, 2> out = { 0L, 0L};
    double &dmin = out[0L], &dmax = out[1L];

    const arma::mat &oborders = other.borders;
    const arma::uword N = this->borders.n_rows;
    for(unsigned int i = 0; i < N; ++i){
      /* min - max */
      dmin += std::pow(std::max(std::max(
         borders(0L, i) - oborders(1L, i),
        oborders(0L, i) -  borders(1L, i)), 0.), 2L);
      /* max - min */
      dmax += std::pow(std::max(
         borders(1L, i) - oborders(0L, i),
        oborders(1L, i) -  borders(0L, i)), 2L);
    }

    return out;
  }
