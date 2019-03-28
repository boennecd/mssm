#include "kd-tree.h"
#include <numeric>

class row_order {
  using idx_ptr = std::unique_ptr<std::vector<arma::uword> >;

  std::vector<arma::uword> depth_idx;
  const arma::mat &X;
public:
  row_order(const arma::mat&);

  struct index_partition {
    idx_ptr left;
    idx_ptr right;
  };

  index_partition get_split(std::vector<arma::uword>&, const arma::uword);
};

KD_note get_KD_tree(const arma::mat &X, const arma::uword N_min){
  std::unique_ptr<std::vector<arma::uword> > idx_in;

  return KD_note(X, N_min, idx_in, nullptr, 0L);
}

KD_note::KD_note(
  const arma::mat &X, const arma::uword N_min, idx_ptr &idx_in,
  row_order *order, const arma::uword depth)
  {
    std::unique_ptr<row_order> ord_ptr;
    if(!order){
      /* first call */
      idx_in.reset(new std::vector<arma::uword>(X.n_cols));
      std::iota(idx_in->begin(), idx_in->end(), 0L);
      ord_ptr.reset(new row_order(X));
      order = ord_ptr.get();

    }

    bool do_split = idx_in->size() > N_min;
    if(do_split){
      /* split indices*/

      idx_ptr idx_left;
      idx_ptr idx_right;

      {
        auto split_dim = order->get_split(*idx_in, depth);
        idx_left  = std::move(split_dim.left);
        idx_right = std::move(split_dim.right);
      }
      idx_in.release(); /* do not need indices anymore */

      left .reset(new KD_note(X, N_min, idx_left , order, depth + 1L));
      right.reset(new KD_note(X, N_min, idx_right, order, depth + 1L));

      return;
    }

    /* it is a leaf */
    idx = std::move(idx_in);
  }

/* function to compute sse */
class sse {
  double sse = 0., x_bar = 0.;
  std::size_t n = 0L;
public:
  void update(const double x){
    ++n;
    double e_t = x - x_bar;
    x_bar += e_t / n;
    sse += e_t * (x - x_bar);
  }

  double get_sse(){
    return sse;
  }
};

row_order::row_order(const arma::mat &X): X(X) {
  /* compute sses */
  std::vector<sse> row_sses(X.n_rows);

  for(auto x = X.begin(); x != X.end(); )
    for(auto &rs : row_sses)
      rs.update(*(x++));

  /* TODO: cheaper option than resize? */
  depth_idx.resize(X.n_rows);
  std::iota(depth_idx.begin(), depth_idx.end(), 0L);
  std::sort(
    depth_idx.begin(), depth_idx.end(),
    [&](arma::uword i1, arma::uword i2){
      return row_sses[i1].get_sse() > row_sses[i2].get_sse();
    });
}

row_order::index_partition row_order::get_split(
    std::vector<arma::uword> &indices, const arma::uword depth)
  {
    /* sort indices */
    const arma::uword inc = X.n_rows, row = depth_idx[depth % X.n_rows];
    const double * const x = X.begin();
    std::sort(
      indices.begin(), indices.end(), [&](arma::uword i1, arma::uword i2){
        return *(x + row + i1 * inc) < *(x + row + i2 * inc);
      });

    /* copy first and second half */
    index_partition out;

    std::size_t split_at = indices.size() / 2;
    out.left.reset (new std::vector<arma::uword>(split_at));
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
  return *idx;
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
#ifdef FSKA_DEBUG
  if(!left)
    throw "get_*_node called on leaf note";
#endif

  return *left;
}

const KD_note& KD_note::get_right() const {
#ifdef FSKA_DEBUG
  if(!right)
    throw "get_*_node called on leaf note";
#endif

  return *right;
}
