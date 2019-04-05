#ifndef KD_TREE_H
#define KD_TREE_H
#include "arma.h"

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
public:
  const arma::uword n_elem;
  bool is_leaf() const {
    return (!left) and (!right);
  };
  const std::vector<arma::uword> &get_indices() const;
  std::vector<arma::uword> get_indices_parent();
  void set_indices(arma::uvec&);
  std::vector<const KD_note*> get_leafs() const;
  const KD_note& get_left () const;
  const KD_note& get_right() const;

  KD_note(KD_note&&) = default;

  friend KD_note get_KD_tree(const arma::mat&, const arma::uword);

private:
  KD_note(const arma::mat&, const arma::uword, idx_ptr&, row_order*,
          const arma::uword, const hyper_rectangle*);

  void get_indices_parent(arma::uword *);
};

KD_note get_KD_tree(const arma::mat&, const arma::uword);

#endif
