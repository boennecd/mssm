#include "fast-kernel-approx.h"
#include <limits>
#include <math.h>
#include <cmath>
#include "kernels.h"
#include "thread_pool.h"
#include <utility>
#include <math.h>
#include "utils.h"
#include <functional>

#ifdef MSSM_PROF
#include <gperftools/profiler.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#endif

using std::ref;
using std::cref;

void comp_weights(
    arma::vec&, const source_node&, const query_node&, const arma::mat&,
    const arma::vec&, const arma::mat&, const double, const trans_obj&,
    thread_pool&, std::list<std::future<void> >&,
    const bool is_single_threaded = true);

using get_X_root_output =
  std::tuple<std::unique_ptr<KD_note>, std::unique_ptr<source_node>,
             arma::uvec>;

/* the function computes the k-d tree and permutate the input matrix
 * and weights. It returns a permutation vector to undo the permutation */
get_X_root_output get_X_root
  (arma::mat &X, arma::vec &ws, const arma::uword N_min)
{
  get_X_root_output out;
  auto &node = std::get<0>(out);
  auto &snode = std::get<1>(out);
  auto &old_idx = std::get<2>(out);

  node.reset(new KD_note(get_KD_tree(X, N_min)));

  /* make permutation to get original order */
  arma::uvec new_idx = node->get_indices_parent();
  old_idx.resize(X.n_cols);
  std::iota(old_idx.begin(), old_idx.end(), 0L);
  node->set_indices(old_idx);
  arma::uword i = 0L;
  for(auto n : new_idx)
    old_idx[n] = i++;

  /* permutate */
  X = X.cols(new_idx);
  ws = ws(new_idx);

  snode.reset(new source_node(X, ws, *node));

  return out;
}

using get_Y_root_output =
  std::tuple<std::unique_ptr<KD_note>, std::unique_ptr<query_node>,
             arma::uvec>;

/* the function computes the k-d tree and permutate the input matrix.
 * It returns a permutation vector to undo the permutation */
get_Y_root_output get_Y_root
  (arma::mat &Y, const arma::uword N_min)
{
  get_Y_root_output out;
  auto &node  = std::get<0L>(out);
  auto &snode = std::get<1L>(out);
  auto &old_idx = std::get<2>(out);

  node.reset(new KD_note(get_KD_tree(Y, N_min)));

  /* make permutation to get original order */
  arma::uvec new_idx = node->get_indices_parent();
  old_idx.resize(Y.n_cols);
  std::iota(old_idx.begin(), old_idx.end(), 0L);
  node->set_indices(old_idx);
  arma::uword i = 0L;
  for(auto n : new_idx)
    old_idx[n] = i++;

  /* permutate */
  Y = Y.cols(new_idx);

  snode.reset(new query_node(Y, *node));

  return out;
}

constexpr unsigned int max_futures       = 30000L;
constexpr unsigned int max_futures_clear = max_futures / 3L;

// [[Rcpp::export]]
arma::vec FSKA(
    arma::mat X, arma::vec ws, arma::mat Y,
    const arma::uword N_min, const double eps,
    const unsigned int n_threads){
#ifdef MSSM_PROF
  std::stringstream ss;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  ss << std::put_time(&tm, "profile-FSKA-%d-%m-%Y-%H-%M-%S.log");
  Rcpp::Rcout << "Saving profile output to '" << ss.str() << "'" << std::endl;
  const std::string s = ss.str();
  ProfilerStart(s.c_str());
#endif

  thread_pool pool(n_threads);

  auto f1 = pool.submit(std::bind(
    get_X_root, ref(X), ref(ws), N_min));
  auto f2 = pool.submit(std::bind(
    get_Y_root, ref(Y), N_min));

  const mvariate kernel(X.n_rows);
  auto X_root = f1.get();
  /* do this __after__ weights are permutated */
  const arma::vec ws_log = arma::log(ws);

  auto Y_root = f2.get();
  source_node &X_root_source = *std::get<1L>(X_root);
  query_node &Y_root_query   = *std::get<1L>(Y_root);
  const arma::uvec &permu_vec = std::get<2L>(Y_root);

  arma::vec log_weights(Y.n_cols);
  log_weights.fill(std::numeric_limits<double>::quiet_NaN());
  std::list<std::future<void> > futures;
  comp_weights(log_weights, X_root_source, Y_root_query, X, ws_log, Y, eps,
               kernel, pool, futures);

  while(!futures.empty()){
    futures.back().get();
    futures.pop_back();
  }

#ifdef MSSM_PROF
  ProfilerStop();
#endif

  return log_weights(permu_vec);
}

inline void set_func(double &o, const double n){
  if(!std::isnan(o))
    o = log_sum_log(o, n);
  else
    o = n;
}

void comp_w_centroid
  (arma::vec &log_weights, const source_node &X_node,
   const query_node &Y_node, const arma::mat &Y, const trans_obj &kernel,
   const bool is_single_threaded)
{
  if(!Y_node.node.is_leaf()){
    comp_w_centroid(log_weights, X_node, *Y_node.left , Y, kernel,
                    is_single_threaded);
    comp_w_centroid(log_weights, X_node, *Y_node.right, Y, kernel,
                    is_single_threaded);
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
    double new_term = kernel(xp, Y.colptr(i), N, x_weight_log) ;
    if(!is_single_threaded){
      *(o++) = new_term;
      continue;
    }

    set_func(log_weights[i], new_term);

  }

  if(is_single_threaded)
    return;

  o = out.begin();
  std::lock_guard<std::mutex> guard(*Y_node.idx_mutex);
  for(auto i : idx)
    set_func(log_weights[i], *(o++));
}

void comp_all(
    arma::vec &log_weights, const source_node &X_node,
    const query_node &Y_node, const arma::mat &X, const arma::vec &ws_log,
    const arma::mat &Y, const trans_obj &kernel, const bool is_single_threaded)
{
#ifdef MSSM_DEBUG
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
      *x_y_ws_i = kernel(X.colptr(i_x), Y.colptr(i_y), N, ws_log[i_x]);
      if(*x_y_ws_i > max_log_w)
        max_log_w = *x_y_ws_i;

      x_y_ws_i++;
    }
    double new_term = log_sum_log(x_y_ws, max_log_w);
    if(!is_single_threaded){
      *(o++) = new_term;
      continue;

    }

    set_func(log_weights[i_y], new_term);

  }

  if(is_single_threaded)
    return;

  o = out.begin();
  std::lock_guard<std::mutex> guard(*Y_node.idx_mutex);
  for(auto i_y : idx_y)
    set_func(log_weights[i_y], *(o++));
}

void comp_weights(
    arma::vec &log_weights, const source_node &X_node,
    const query_node &Y_node, const arma::mat &X,
    const arma::vec &ws_log, const arma::mat &Y, const double eps,
    const trans_obj &kernel, thread_pool &pool,
    std::list<std::future<void> > &futures, const bool single_threaded)
  {
    /* check if we need to clear futures */
    if(single_threaded and futures.size() > max_futures){
      std::size_t n_earsed = 0L;
      std::future_status status;
      constexpr std::chrono::milliseconds t_weight(1);
      std::list<std::future<void> >::iterator it;
      const std::list<std::future<void> >::const_iterator
        end = futures.end();
      while(n_earsed < max_futures_clear){
        for(it = futures.begin(); it != end; ){
          status = it->wait_for(t_weight);
          if(status == std::future_status::ready){
            it->get();
            it = futures.erase(it);
            n_earsed++;
            if(n_earsed >= max_futures_clear)
              break;

          } else
            ++it;
        }
      }
    }

    /* check if we should finish the rest in another thread */
    constexpr arma::uword stop_n_elem = 50L;
    if(single_threaded and
         X_node.node.n_elem < stop_n_elem and
         Y_node.node.n_elem < stop_n_elem){
      futures.push_back(
        pool.submit(std::bind(
            comp_weights,
            ref(log_weights), cref(X_node), cref(Y_node), cref(X),
            cref(ws_log), cref(Y), eps, cref(kernel), ref(pool),
            ref(futures), false)));
      return;
    }

    auto dists = Y_node.borders.min_max_dist(X_node.borders);
    double k_min = std::exp(kernel(dists[1])),
      k_max = std::exp(kernel(dists[0]));
    if(X_node.weight *
        (k_max - k_min) / ((k_max + k_min) / 2. + 1e-16) < 2. * eps){
      auto task = std::bind(
        comp_w_centroid,
        ref(log_weights), cref(X_node), cref(Y_node),
        cref(Y), cref(kernel), pool.thread_count < 2L);
      if(single_threaded)
        futures.push_back(pool.submit(std::move(task)));
      else
        task();

      return;
    }

    if(X_node.node.is_leaf() and Y_node.node.is_leaf()){
      auto task = std::bind(
        comp_all, ref(log_weights), cref(X_node), cref(Y_node),
        cref(X), cref(ws_log), cref(Y), cref(kernel),
        pool.thread_count < 2L);
      if(single_threaded)
        futures.push_back(pool.submit(std::move(task)));
      else
        task();
      return;
    }

    if(!X_node.node.is_leaf() and  Y_node.node.is_leaf()){
      comp_weights(
        log_weights, *X_node.left ,  Y_node,
        X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
      comp_weights(
        log_weights, *X_node.right,  Y_node,
        X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
      return;
    }
    if( X_node.node.is_leaf() and !Y_node.node.is_leaf()){
      comp_weights(
        log_weights,  X_node     , *Y_node.left,
        X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
      comp_weights(
        log_weights,  X_node     , *Y_node.right,
        X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
      return;
    }

    comp_weights(
      log_weights, *X_node.left , *Y_node.left ,
      X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
    comp_weights(
      log_weights, *X_node.left , *Y_node.right,
      X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
    comp_weights(
      log_weights, *X_node.right, *Y_node.left ,
      X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
    comp_weights(
      log_weights, *X_node.right, *Y_node.right,
      X, ws_log, Y, eps, kernel, pool, futures, single_threaded);
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
  weight(set_weight(*this, ws)), borders(set_borders(*this, X))
  { }



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
