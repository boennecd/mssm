#include "kd-tree.h"
#include <numeric>

using std::domain_error;
using std::invalid_argument;
using std::ref;
using std::cref;

class row_order {
  using idx_ptr = std::unique_ptr<std::vector<arma::uword> >;

  const arma::mat &X;
public:
  row_order(const arma::mat&);

  struct index_partition {
    idx_ptr left;
    idx_ptr right;
    arma::uword dim;
  };

  index_partition get_split
    (std::vector<arma::uword>&, const hyper_rectangle&);
};

KD_note get_KD_tree
  (const arma::mat &X, const arma::uword N_min, thread_pool &pool){
  std::unique_ptr<std::vector<arma::uword> > idx_in;

  std::vector<std::future<void> > futures;
  std::mutex lc;

  auto out = KD_note(X, N_min, std::move(idx_in), nullptr, 0L, nullptr, pool,
                     futures, lc);

  out.set_depth();

  return out;
}

KD_note::KD_note(
  const arma::mat &X, const arma::uword N_min, idx_ptr &&idx_in_r,
  row_order *order, const arma::uword depth, const hyper_rectangle *rect,
  thread_pool &pool, std::vector<std::future<void> > &futures,
  std::mutex &lc):
  n_elem(idx_in_r ? idx_in_r->size() : X.n_cols)
  {
    /* fraction at which to start multi-threading */
    static constexpr double frac_mult = .01;        /* ~ 2^7 split */
    static constexpr unsigned int max_nodes = 300L; /* ~ 2^1 + 2^2 + ... + 2^x */

    idx_ptr idx_in(std::move(idx_in_r));
    std::unique_ptr<row_order> ord_ptr;
    std::unique_ptr<hyper_rectangle> rect_ptr;

    const bool is_first_call = !order;
    if(is_first_call){
      idx_in.reset(new std::vector<arma::uword>(X.n_cols));
      std::iota(idx_in->begin(), idx_in->end(), 0L);
      ord_ptr.reset(new row_order(X));
      order = ord_ptr.get();
      rect_ptr.reset(new hyper_rectangle(X, *idx_in));
      rect = rect_ptr.get();

      futures.reserve(max_nodes);
    }

    bool do_split = idx_in->size() > N_min;
    if(do_split){
      /* split indices*/
      idx_ptr idx_left;
      idx_ptr idx_right;

      hyper_rectangle rect_left = *rect, rect_right = *rect;
      {
        auto split_dim = order->get_split(*idx_in, *rect);
        idx_left  = std::move(split_dim.left);
        idx_right = std::move(split_dim.right);

        rect_left .shrink(X, *idx_left , split_dim.dim);
        rect_right.shrink(X, *idx_right, split_dim.dim);
      }

      const bool
        finish =
          pool.thread_count < 2L or
          idx_in->size() <= 50L or
          (double)idx_in->size() <= (double)X.n_cols * frac_mult;

      /* set left and right child */
      auto get_worker =
        [&](std::unique_ptr<KD_note> &ptr, idx_ptr &&indices,
            hyper_rectangle new_rect) {
          return set_child {
            ptr, std::move(indices), new_rect, X, N_min, order, depth, pool,
            futures, lc };
      };

      auto task_left  = get_worker(
             left , std::move(idx_left ), std::move(rect_left)),
           task_right = get_worker(
             right, std::move(idx_right), std::move(rect_right));

      if(finish){
        task_left();
        task_right();

      } else {
        std::lock_guard<std::mutex> ga(lc);
        futures.push_back(pool.submit(std::move(task_left )));
        futures.push_back(pool.submit(std::move(task_right)));

      }

      if(is_first_call)
        /* We cannot return before all futures are done */
        while(!futures.empty()){
          std::future_status status;
          auto &current_back = futures.back();
          status = current_back.wait_for(std::chrono::nanoseconds(1L));
          if(status == std::future_status::ready){
            std::lock_guard<std::mutex> ga(lc);
            if(&current_back == &futures.back()){
              current_back.get();
              futures.pop_back();
            }

          } else
            std::this_thread::yield();
        }

      return;
    }

    /* it is a leaf */
    idx = std::move(idx_in);
  }



row_order::row_order(const arma::mat &X): X(X) { }

row_order::index_partition row_order::get_split(
    std::vector<arma::uword> &indices, const hyper_rectangle &rect)
  {
    /* find index to split at */
    const arma::mat &borders = rect.get_borders();
    arma::uword row = 0L;
    double d_max = borders(1L, 0L) - borders(0L, 0L);
    for(unsigned int j = 1L; j < borders.n_cols; ++j){
      const double diff = borders(1L, j) - borders(0L, j);
      if(diff > d_max){
        d_max = diff;
        row = j;
      }
    }

    /* sort indices */
    const arma::uword inc = X.n_rows;
    const double * const x = X.begin();
    std::size_t split_at = indices.size() / 2L;
    std::nth_element(
      indices.begin(), indices.begin() + split_at, indices.end(),
      [&](const arma::uword i1, const arma::uword i2){
        return *(x + row + i1 * inc) < *(x + row + i2 * inc);
      });

    /* copy first and second half */
    index_partition out;
    out.dim = row;

    out.left .reset(new std::vector<arma::uword>(split_at));
    out.right.reset(new std::vector<arma::uword>(indices.size() - split_at));

    std::vector<arma::uword> &left  = *out.left;
    std::vector<arma::uword> &right = *out.right;

    /* TODO: do something smarter? */
    auto idx = indices.begin();
    for(std::size_t i = 0; i < split_at; ++i, ++idx)
      left[i] = *idx;
    for(std::size_t i = 0; idx != indices.end(); ++i, ++idx)
      right[i] = *idx;

    return out;
  }

const std::vector<arma::uword>& KD_note::get_indices() const {
#ifdef MSSM_DEBUG
  if(!is_leaf())
    throw domain_error("'get_indices' called on non-leaf");
#endif
  return *idx;
}

std::vector<arma::uword> KD_note::get_indices_parent(){
  if(is_leaf())
    return get_indices();

  std::vector<arma::uword> out(left->n_elem + right->n_elem);
  get_indices_parent(out.data());

  return out;
}

void KD_note::get_indices_parent(arma::uword *out){
  if(is_leaf()){
    memcpy(out, idx->data(), sizeof(arma::uword) * idx->size());
    return;
  }

  left ->get_indices_parent(out);
  right->get_indices_parent(out + left->n_elem);
}

std::vector<const KD_note*> KD_note::get_leafs() const {
  if(is_leaf())
    return { this };

  auto out  = left->get_leafs();
  auto leafs_right = right->get_leafs();
  out.reserve(out.size() + leafs_right.size());
  for(auto lr : leafs_right)
    out.push_back(lr);

  return out;
}

const KD_note& KD_note::get_left() const {
#ifdef MSSM_DEBUG
  if(!left)
    throw domain_error("get_*_node called on leaf note");
#endif

  return *left;
}

const KD_note& KD_note::get_right() const {
#ifdef MSSM_DEBUG
  if(!right)
    throw domain_error("get_*_node called on leaf note");
#endif

  return *right;
}

void KD_note::set_indices(arma::uvec &new_idx) {
#ifdef MSSM_DEBUG
  if(new_idx.n_elem != n_elem)
    throw invalid_argument("indices length do not match with node size");
#endif

  if(is_leaf()){
    const arma::uword *i = new_idx.begin();
    for(auto &k : *idx)
      k = *(i++);

    return;
  }

  const arma::uword n_left = left->n_elem;
  arma::uvec left_idx (new_idx.begin()         , n_left       , false);
  arma::uvec right_idx(new_idx.begin() + n_left, right->n_elem, false);

  left ->set_indices(left_idx);
  right->set_indices(right_idx);
}

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
#ifdef MSSM_DEBUG
  if(r1.borders.n_rows != r2.borders.n_rows or
       r1.borders.n_cols != r2.borders.n_cols)
    throw invalid_argument("dimension do not match");
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
#ifdef MSSM_DEBUG
  if(this->borders.n_rows != other.borders.n_rows or
       this->borders.n_cols != other.borders.n_cols)
    throw invalid_argument("dimension do not match");
#endif
  std::array<double, 2> out = { 0L, 0L};
  double &dmin = out[0L], &dmax = out[1L];

  const arma::mat &oborders = other.borders;
  const arma::uword N = this->borders.n_cols;
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

#ifdef MSSM_DEBUG
inline double round_to_digits(const double value, const unsigned int digits)
{
  if (value == 0.0)
    return 0.0;

  double factor = pow(10.0, digits - ceil(log10(fabs(value))));
  return round(value * factor) / factor;
}

std::ostream& operator<<(std::ostream &os, const hyper_rectangle &rect){
  constexpr unsigned int n_d = 3L;
  const double *d = rect.borders.begin();
  for(unsigned int i = 0; i < rect.borders.n_cols; ++i){
    if(i > 0L)
      os << " x ";
    double x1 = *(d++), x2 = *(d++);
    os << '[' << round_to_digits(x1, n_d) << ", " <<
      round_to_digits(x2, n_d) << ']';

  }
  os << '\n';

  return os;
}
#endif

void hyper_rectangle::shrink
  (const arma::mat &X, const std::vector<arma::uword> &idx,
   const arma::uword dim)
  {
#ifdef MSSM_DEBUG
    if(idx.size() < 1L)
      throw std::logic_error("'shrink' called with no elements");
#endif
    double &lower = borders(0L, dim) , &upper = borders(1L, dim);
    auto i = idx.begin();
    {
      /* set to first point */
      lower = upper = X(dim, *(i++));
    }
    auto end = idx.end();
    for(; i != end; ++i){
      double x = X(dim, *i);
      if(x > upper)
        upper = x;
      else if(x < lower)
        lower = x;
    }
  }
