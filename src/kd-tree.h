#ifndef KD_TREE_H
#define KD_TREE_H
#include "arma.h"
#include "thread_pool.h"
#include <array>

#ifdef MSSM_DEBUG
#include <iostream>
#endif

class hyper_rectangle {
  /* with borders for the hyper rectangle. Dims are 2 x [dim] */
  arma::mat borders;
public:
  hyper_rectangle(const arma::mat&, const arma::uvec&);
  hyper_rectangle(const hyper_rectangle&, const hyper_rectangle&);

  /* first element is min and second element is max */
  std::array<double, 2> min_max_dist(const hyper_rectangle&) const;

  const arma::mat& get_borders() const {
    return borders;
  };

  void shrink(
      const arma::mat&, const std::vector<arma::uword>&, const arma::uword);

#ifdef MSSM_DEBUG
  friend std::ostream& operator<<(std::ostream&, const hyper_rectangle&);
#endif
};

class row_order;

class KD_note {
  using idx_ptr = std::unique_ptr<std::vector<arma::uword> >;
  std::unique_ptr<std::vector<arma::uword> > idx;
  std::unique_ptr<KD_note> left;
  std::unique_ptr<KD_note> right;
  arma::uword depth;
public:
  const arma::uword n_elem;
  bool is_leaf() const {
    return !(left or right);
  };
  const std::vector<arma::uword> &get_indices() const;
  std::vector<arma::uword> get_indices_parent();
  void set_indices(arma::uvec&);
  std::vector<const KD_note*> get_leafs() const;
  const KD_note& get_left () const;
  const KD_note& get_right() const;
  /* assumes 'set_depth' has been called */
  arma::uword get_depth() const {
    return depth;
  }

  /* return the start index and end index. Only valid if data is sorted by
   * using the 'get_indices_parent' and 'set_indices' member functions */
  arma::uword get_start() const {
    if(is_leaf())
      return idx->front();

    return left->get_start();
  }
  arma::uword get_end() const {
    if(is_leaf())
      return idx->back() + 1L;

    return right->get_end();
  }

  KD_note(KD_note&&) = default;

  friend KD_note get_KD_tree(
      const arma::mat&, const arma::uword, thread_pool&);

private:
  KD_note(const arma::mat&, const arma::uword, idx_ptr&&, row_order*,
          const arma::uword, const hyper_rectangle*,
          thread_pool&, std::vector<std::future<void> >&, std::mutex&);

  void get_indices_parent(arma::uword*);

  /* util class used in constructor */
  struct set_child {
    std::unique_ptr<KD_note> &ptr;
    idx_ptr indices;
    hyper_rectangle child_rect;
    const arma::mat &X;
    const arma::uword N_min;
    row_order *order;
    const arma::uword depth;
    thread_pool &pool;
    std::vector<std::future<void> > &futures;
    std::mutex &lc;

    void operator()()
    {
      ptr.reset(new KD_note(
          X, N_min, std::move(indices), order, depth + 1L, &child_rect,
          pool, futures, lc));
    }
  };

  void set_depth() {
    if(is_leaf()){
      depth = 1L;
      return;
    }

    left ->set_depth();
    right->set_depth();

    depth = std::max(left->depth, right->depth) + 1L;
  }
};

KD_note get_KD_tree(const arma::mat&, const arma::uword, thread_pool&);

#endif
