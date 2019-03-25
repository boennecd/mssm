#include <RcppArmadillo.h>

class row_order;

class KD_note {
  using idx_ptr = std::unique_ptr<std::vector<arma::uword> >;

  std::unique_ptr<std::vector<arma::uword> > idx;
  std::unique_ptr<KD_note> left;
  std::unique_ptr<KD_note> right;
public:
  bool is_leaf() const;
  const std::vector<arma::uword> &get_indices() const;
  std::vector<const KD_note*> get_leafs() const;
  friend KD_note get_KD_tree(const arma::mat&, const arma::uword);

private:
  KD_note(const arma::mat&, const arma::uword, idx_ptr&, row_order*,
          const arma::uword);
};

KD_note get_KD_tree(const arma::mat&, const arma::uword);
