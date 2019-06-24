#include "fast-kernel-approx.h"
#include <limits>
#include <cmath>
#include <utility>
#include "utils.h"
#include <functional>
#include "misc.h"

#ifdef MSSM_PROF
#include "profile.h"
#endif

static constexpr unsigned int max_futures       = 30000L;
static constexpr unsigned int max_futures_clear = max_futures / 3L;

template<bool has_extra>
using get_X_root_output =
  std::tuple<std::unique_ptr<KD_note>, std::unique_ptr<source_node<has_extra > >,
             arma::uvec>;

/* the function computes the k-d tree and permutate the input matrix
 * and weights. It returns a permutation vector to undo the permutation */
template<bool has_extra>
get_X_root_output<has_extra> get_X_root
  (arma::mat &X, arma::vec &ws, const arma::uword N_min, arma::mat *xtra,
   thread_pool &pool)
{
  get_X_root_output<has_extra> out;
  auto &node = std::get<0>(out);
  auto &snode = std::get<1>(out);
  auto &old_idx = std::get<2>(out);

  node.reset(new KD_note(get_KD_tree(X, N_min, pool)));

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
  if(has_extra)
    *xtra = xtra->cols(new_idx);

  snode.reset(new source_node<has_extra>(X, ws, *node, xtra));

  return out;
}

using get_Y_root_output =
  std::tuple<std::unique_ptr<KD_note>, std::unique_ptr<query_node>,
             arma::uvec>;

/* the function computes the k-d tree and permutate the input matrix.
 * It returns a permutation vector to undo the permutation */
template<bool has_extra>
get_Y_root_output get_Y_root
  (arma::mat &Y, const arma::uword N_min, arma::mat *xtra,
   thread_pool &pool)
{
  get_Y_root_output out;
  auto &node  = std::get<0L>(out);
  auto &snode = std::get<1L>(out);
  auto &old_idx = std::get<2>(out);

  node.reset(new KD_note(get_KD_tree(Y, N_min, pool)));

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
  if(has_extra)
    *xtra = xtra->cols(new_idx);

  snode.reset(new query_node(Y, *node));

  return out;
}

template<bool has_extra>
void set_func
  (double &o, const double log_weight, const double *source, const double *query,
   const double *source_extra, double *query_extra, FSKA_cpp_xtra_func *extra_func)
{
  o = log_sum_log(o, log_weight);
  if(!has_extra)
    return;

  (*extra_func)(source, query, source_extra, query_extra, log_weight);
}

template<bool has_extra>
struct comp_w_centroid {
  arma::vec &log_weights;
  const source_node<has_extra> &X_node;
  const query_node &Y_node;
  const arma::mat &Y;
  const trans_obj &kernel;
  const bool is_single_threaded;
  arma::mat *Y_extra;
  FSKA_cpp_xtra_func &extra_func;

 void operator()(){
    const arma::uword start = Y_node.node.get_start(),
      end   = Y_node.node.get_end();

    const arma::vec &X_centroid = X_node.centroid;
    double x_weight_log = std::log(X_node.weight);
    const double *xp = X_centroid.begin(),
      *xp_extra = has_extra ?  X_node.extra->memptr() : nullptr;
    const arma::uword N = X_centroid.n_elem;

    arma::vec out;
    arma::mat xtra;
    double *o = nullptr;
    M_THREAD_LOCAL std::vector<double> mem;

    /* setup needed objects */
    if(!is_single_threaded){
      const unsigned int n_p = end - start,
        required_mem = has_extra ? n_p * (1L + Y_extra->n_rows) : n_p;
      if(mem.size() < required_mem)
        mem.resize(required_mem);
      double *mem_ = mem.data();
      out = arma::vec(mem_, n_p, false);
      o = out.begin();
      if(has_extra){
        xtra = arma::mat(mem_ + n_p, Y_extra->n_rows, n_p, false);
        xtra.zeros();
      }

    }

    for(arma::uword i = start; i < end; ++i){
      const double *yp = Y.colptr(i);
      double new_term = kernel(xp, yp, N, x_weight_log) ;
      if(!is_single_threaded){
        *(o++) = new_term;

        if(has_extra)
          /* compute stats */
          extra_func(xp, yp, xp_extra, xtra.colptr(i - start), new_term);

        continue;
      }

      double *Y_extra_ptr = has_extra ? Y_extra->colptr(i) : nullptr;

      set_func<has_extra>(
        log_weights[i], new_term, xp, yp, xp_extra,
        Y_extra_ptr, &extra_func);

    }

    if(is_single_threaded)
      return;

    /* travers down the tree from left to right. We use that data is sorted and
    * we only lock one lock at a time */
    o = out.begin();
    M_THREAD_LOCAL std::vector<const query_node*> tasks;
    const std::size_t required_size = Y_node.node.get_depth() + 1L;
    if(tasks.size() < required_size)
      tasks.resize(required_size);

    int yi = 0L;
    tasks.front() = &Y_node;
    const query_node** yip = tasks.data();
    while(yi > -1){
      const query_node* this_node = *yip;

      if(this_node->is_leaf){
        const arma::uword
        this_start = this_node->node.get_start(),
          this_end   = this_node->node.get_end();

          {
            std::lock_guard<std::mutex> gr(*this_node->idx_mutex);
            double * w_out = log_weights.begin() + this_start;
            for(arma::uword i = this_start; i < this_end; ++i, ++o, ++w_out)
              *w_out = log_sum_log(*w_out, *o);

            if(has_extra)
              Y_extra->cols(this_start, this_end - 1L) +=
                xtra.cols(this_start - start, this_end - start - 1L);
          }

          --yip;
          --yi;
          continue;

      }

      *   yip  = this_node->right.get();
      *(++yip) = this_node->left.get();
      ++yi;
    }
  }
};

template<bool has_extra>
struct  comp_all {
  arma::vec &log_weights;
  const source_node<has_extra> &X_node;
  const query_node &Y_node;
  const arma::mat &X;
  const arma::vec &ws_log;
  const arma::mat &Y;
  const trans_obj &kernel;
  const bool is_single_threaded;
  arma::mat *X_extra;
  arma::mat *Y_extra;
  FSKA_cpp_xtra_func &extra_func;

  void operator()(){
#ifdef MSSM_DEBUG
    if(!X_node.is_leaf or !Y_node.is_leaf)
      throw std::domain_error(
          "comp_all called with non-leafs");
#endif

    const arma::uword
      start_X = X_node.node.get_start(), end_X = X_node.node.get_end(),
        start_Y = Y_node.node.get_start(), end_Y = Y_node.node.get_end();

    arma::vec out, stats_inner, x_y_ws;
    arma::mat xtra;
    double *o = nullptr;
    M_THREAD_LOCAL std::vector<double> mem;

    /* setup required memory */
    if(!is_single_threaded){
      const unsigned int n_p = end_Y - start_Y, n_p_x = end_X - start_X,
        required_mem = has_extra ?
      n_p * (1L + Y_extra->n_rows) + n_p_x + Y_extra->n_rows : n_p + n_p_x;
      if(mem.size() < required_mem)
        mem.resize(required_mem);
      double *mem_ = mem.data();
      out = arma::vec(mem_, n_p, false);
      mem_ += n_p;
      o = out.begin();
      if(has_extra){
        xtra = arma::mat(mem_, Y_extra->n_rows, n_p, false);
        mem_ += Y_extra->n_rows *  n_p;
        xtra.zeros();

        stats_inner = arma::vec(mem_, Y_extra->n_rows, false);
        mem_ += Y_extra->n_rows;
      }

      x_y_ws = arma::vec(mem_, n_p_x, false);

    } else {
      const unsigned int required_mem =
        has_extra ?
        end_X - start_X + Y_extra->n_rows :
      end_X - start_X;
      if(mem.size() < required_mem)
        mem.resize(required_mem);

      double * mem_ = mem.data();
      if(has_extra){
        stats_inner = arma::vec(mem_, Y_extra->n_rows, false);
        mem_ += Y_extra->n_rows;

      }

      x_y_ws = arma::vec(mem_, end_X - start_X, false);

    }

    for(arma::uword i_y = start_Y; i_y < end_Y; ++i_y){
      const arma::uword N = Y.n_rows;
      double max_log_w = std::numeric_limits<double>::lowest();
      double *x_y_ws_i = x_y_ws.begin();
      if(has_extra)
        stats_inner.zeros();
      const double * yp = Y.colptr(i_y);
      for(arma::uword i_x = start_X; i_x < end_X; ++i_x){
        const double * xp = X.colptr(i_x),
          *xp_extra = has_extra ? X_extra->colptr(i_x) : nullptr;
        *x_y_ws_i = kernel(xp, yp, N, ws_log[i_x]);
        if(*x_y_ws_i > max_log_w)
          max_log_w = *x_y_ws_i;

        if(has_extra)
          /* compute stats */
          extra_func(xp, yp, xp_extra, stats_inner.memptr(), *x_y_ws_i);

        x_y_ws_i++;
      }

      double new_term = log_sum_log(x_y_ws, max_log_w);
      if(!is_single_threaded){
        /* save log weight */
        *(o++) = new_term;
        /* add stats terms */
        if(has_extra)
          xtra.col(i_y - start_Y) += stats_inner;

        continue;

      }

      /* update weights */
      log_weights[i_y] = log_sum_log(log_weights[i_y], new_term);

      /* add stats */
      if(has_extra)
        Y_extra->col(i_y) += stats_inner;
    }

    if(is_single_threaded)
      return;

    o = out.begin();
    double * w_out = log_weights.begin() + start_Y;
    std::lock_guard<std::mutex> guard(*Y_node.idx_mutex);
    for(arma::uword i_y = start_Y; i_y < end_Y; ++i_y, ++o, ++w_out)
      *w_out = log_sum_log(*w_out, *o);

    if(has_extra)
      Y_extra->cols(start_Y, end_Y - 1L) += xtra;
  }
};

template<bool has_extra>
struct comp_weights {
  arma::vec &log_weights;
  const arma::mat &X;
  const arma::vec &ws_log;
  const arma::mat &Y;
  const double eps;
  const trans_obj &kernel;
  thread_pool &pool;
  std::list<std::future<void> > &futures;
  arma::mat *X_extra;
  arma::mat *Y_extra;
  FSKA_cpp_xtra_func &extra_func;

  template<bool is_main_thread>
  void do_work
  (const source_node<has_extra> &X_node, const query_node &Y_node) const
  {
    /* check if we need to clear futures. TODO: avoid the use of list here? */
    if(is_main_thread and futures.size() > max_futures){
      std::size_t n_earsed = 0L;
      std::future_status status;
      constexpr std::chrono::milliseconds t_weight(1);
      std::list<std::future<void> >::iterator it;
      while(n_earsed < max_futures_clear){
        for(it = futures.begin(); it != futures.end(); ){
          status = it->wait_for(t_weight);
          if(status == std::future_status::ready){
            it->get();
            it = futures.erase(it);
            n_earsed++;
            if(n_earsed >= max_futures_clear)
              break;

          } else {
            ++it;
            std::this_thread::yield();

          }
        }
      }
    }

    /* check if we should finish the rest in another thread */
    static constexpr arma::uword stop_n_elem = 50L;
    if(is_main_thread and
         X_node.node.n_elem < stop_n_elem and
         Y_node.node.n_elem < stop_n_elem){
      futures.push_back(
        pool.submit(std::bind(
            &comp_weights<has_extra>::do_work<false>, std::ref(*this),
            std::cref(X_node), std::cref(Y_node))));
      return;
    }

    auto log_dens = kernel(Y_node.borders, X_node.borders);
    double k_min = std::exp(log_dens[0L]), k_max = std::exp(log_dens[1L]);
    if(X_node.weight *
       (k_max - k_min) / ((k_max + k_min) / 2. + 1e-16) < 2. * eps){
      comp_w_centroid<has_extra> task =
        {
          log_weights, X_node, Y_node,
          Y, kernel, pool.thread_count < 2L, Y_extra, extra_func
        };
      if(is_main_thread)
        futures.push_back(pool.submit(std::move(task)));
      else
        task();

      return;
    }

    if(X_node.is_leaf and Y_node.is_leaf){
      comp_all<has_extra> task = {
        log_weights, X_node, Y_node,
        X, ws_log, Y, kernel,
        pool.thread_count < 2L, X_extra, Y_extra, extra_func
      };
      if(is_main_thread)
        futures.push_back(pool.submit(std::move(task)));
      else
        task();
      return;
    }

    if(!X_node.is_leaf and  Y_node.is_leaf){
      do_work<is_main_thread>(*X_node.left ,  Y_node      );
      do_work<is_main_thread>(*X_node.right,  Y_node      );

      return;
    }
    if( X_node.is_leaf and !Y_node.is_leaf){
      do_work<is_main_thread>( X_node      , *Y_node.left );
      do_work<is_main_thread>( X_node      , *Y_node.right);

      return;
    }

    do_work<is_main_thread>(  *X_node.left , *Y_node.left );
    do_work<is_main_thread>(  *X_node.left , *Y_node.right);
    do_work<is_main_thread>(  *X_node.right, *Y_node.left );
    do_work<is_main_thread>(  *X_node.right, *Y_node.right);
  }
};

template<bool has_extra>
FSKA_cpp_permutation FSKA_cpp(
    arma::vec &log_weights, arma::mat &X, arma::mat &Y, arma::vec &ws_log,
    const arma::uword N_min, const double eps, const trans_obj &kernel,
    thread_pool &pool, const bool has_transformed, arma::mat *X_extra,
    arma::mat *Y_extra, FSKA_cpp_xtra_func extra_func)
{
#ifdef MSSM_DEBUG
  if(log_weights.n_elem != Y.n_cols)
    throw std::invalid_argument(
        "Dimensions of 'log_weights' and 'Y' do not match");
  if(has_extra != (bool)X_extra or has_extra != (bool)Y_extra)
    throw std::invalid_argument(
        "'has_extra' not equal to 'X_extra' or 'Y_extra'");
  if(Y_extra and Y_extra->n_cols != Y.n_cols)
    throw std::invalid_argument(
        "invalid 'Y_extra' and 'Y'");
  if(X_extra and X_extra->n_cols != X.n_cols)
    throw std::invalid_argument(
        "invalid 'X_extra' and 'X'");
  if((bool)extra_func != has_extra)
    throw std::invalid_argument(
      "invalid 'extra_func' and 'has_extra'");
#endif

#ifdef MSSM_PROF
  // profiler prof("FSKA_cpp");
#endif

  /* transform X and Y before doing any computation */
  if(!has_transformed){
    auto t1 = pool.submit(std::bind(
      &trans_obj::trans_X, &kernel, std::ref(X)));
    auto t2 = pool.submit(std::bind(
      &trans_obj::trans_Y, &kernel, std::ref(Y)));
    t1.get();
    t2.get();
  }

  /* form trees */
  auto X_root = get_X_root<has_extra>(X, ws_log, N_min, X_extra, pool);
  auto Y_root = get_Y_root<has_extra>(Y,         N_min, Y_extra, pool);

  std::list<std::future<void> > futures;
  source_node<has_extra> &X_root_source = *std::get<1L>(X_root);
  query_node &Y_root_query   = *std::get<1L>(Y_root);

  /* compute weights etc. This is a bad design. The class we define
   * must not get destructed due to a 'this' pointer used in the function... */
  comp_weights<has_extra> worker {
    log_weights, X, ws_log, Y, eps,
    kernel, pool, futures, X_extra, Y_extra, extra_func };
  worker.template do_work<true>(X_root_source, Y_root_query);

  while(!futures.empty()){
    futures.back().get();
    futures.pop_back();
  }

  /* transform back */
  if(!has_transformed){
    auto ta = pool.submit(std::bind(
      &trans_obj::trans_inv_X, &kernel, std::ref(X)));
    kernel.trans_inv_Y(Y);
    ta.get();
  }

  return
  { std::move(std::get<2L>(X_root)) , std::move(std::get<2L>(Y_root)) };
}

template FSKA_cpp_permutation FSKA_cpp<true>(
    arma::vec&, arma::mat&, arma::mat&, arma::vec&, const arma::uword,
    const double, const trans_obj&, thread_pool&, const bool,
    arma::mat*, arma::mat*, FSKA_cpp_xtra_func);
template FSKA_cpp_permutation FSKA_cpp<false>(
    arma::vec&, arma::mat&, arma::mat&, arma::vec&, const arma::uword,
    const double, const trans_obj&, thread_pool&, const bool,
    arma::mat*, arma::mat*, FSKA_cpp_xtra_func);

template<bool has_extra>
std::unique_ptr<const source_node<has_extra> > set_child
  (const arma::mat &X, const arma::vec &ws, const KD_note &node,
   const arma::mat *extra, const bool is_left)
{
  typedef std::unique_ptr<const source_node<has_extra> > ptr_out;
  if(node.is_leaf())
    return ptr_out();
  if(is_left)
    return ptr_out(
      new source_node<has_extra>(X, ws, node.get_left (), extra));
  return ptr_out(
      new source_node<has_extra>(X, ws, node.get_right(), extra));
}

template<bool has_extra>
arma::vec set_centroid
  (const source_node<has_extra> &snode, const arma::mat &X, const arma::vec &ws)
{
  if(snode.is_leaf){
    arma::vec centroid(X.n_rows, arma::fill::zeros);
    const auto &indices = snode.node.get_indices();
    double sum_w = 0.;
    for(auto idx : indices){
      const double w = std::exp(ws[idx]);
      centroid += w * X.unsafe_col(idx);
      sum_w += w;
    }
    centroid /= sum_w;

    return centroid;
  }

  double w1 = snode.left->weight, w2 = snode.right->weight;
  return
    (w1 / (w1 + w2)) * snode.left->centroid +
    (w2 / (w1 + w2)) * snode.right->centroid;
}

template<bool has_extra>
double set_weight(const source_node<has_extra> &snode, const arma::vec &ws)
{
  if(snode.is_leaf){
    const auto &indices = snode.node.get_indices();
    double weight = 0.;
    for(auto idx : indices)
      weight += std::exp(ws[idx]);

    return weight;
  }

  return snode.left->weight + snode.right->weight;
}

template<class T>
hyper_rectangle set_borders(const T &snode, const arma::mat &X){
  if(snode.is_leaf)
    return hyper_rectangle(X, snode.node.get_indices());

  return hyper_rectangle(snode.left->borders, snode.right->borders);
}

template<bool has_extra>
std::unique_ptr<arma::vec> set_extra
  (const source_node<has_extra> &snode, const arma::mat *extra,
   const arma::vec &ws)
{
#ifdef MSSM_DEBUG
  if((bool)extra != has_extra)
    throw std::invalid_argument("invalid 'has_extra' and 'extra'");
#endif

  typedef std::unique_ptr<arma::vec> ptr_out;

  if(!has_extra)
    return ptr_out();

  ptr_out out(new arma::vec(extra->n_rows, arma::fill::none));
  arma::vec &xtr = *out;

  if(snode.is_leaf){
    xtr.zeros();
    const auto &indices = snode.node.get_indices();
    double sum_w = 0.;
    for(auto idx : indices){
      const double w = std::exp(ws[idx]);
      xtr += w * extra->unsafe_col(idx);
      sum_w += w;
    }
    xtr /= sum_w;

    return out;
  }

  double w1 = snode.left->weight, w2 = snode.right->weight;
  xtr =
    (w1 / (w1 + w2)) * *snode.left->extra +
      (w2 / (w1 + w2)) * *snode.right->extra;
  return out;
}

template<bool has_extra>
source_node<has_extra>::source_node
  (const arma::mat &X, const arma::vec &ws, const KD_note &node,
   const arma::mat *extra):
  node(node), left(set_child<has_extra>(X, ws, node, extra, true)),
  right(set_child<has_extra>(X, ws, node, extra, false)),
  centroid(set_centroid<has_extra>(*this, X, ws)),
  weight(set_weight(*this, ws)), borders(set_borders(*this, X)),
  extra(set_extra<has_extra>(*this, extra, ws))
  { }

template class source_node<true >;
template class source_node<false>;

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
