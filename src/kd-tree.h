#ifndef KD_TREE_H
#define KD_TREE_H
#include "arma.h"

class row_order;

class KD_note {
  using idx_ptr = std::unique_ptr<std::vector<arma::uword> >;

  const arma::uword n_elem;
  std::unique_ptr<std::vector<arma::uword> > idx;
  std::unique_ptr<KD_note> left;
  std::unique_ptr<KD_note> right;
public:
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
          const arma::uword);

  void get_indices_parent(arma::uword *);
};

KD_note get_KD_tree(const arma::mat&, const arma::uword);

#endif
