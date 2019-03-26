#include "kd-tree.hpp"
#include <array>

class hyper_rectangle {
  /* with borders for the hyper rectangle. Dims are 2 x [dim] */
  arma::mat borders;
public:
  hyper_rectangle(const arma::mat&, const arma::uvec&);
  hyper_rectangle(const hyper_rectangle&, const hyper_rectangle&);

  /* first element is min and second element is max */
  std::array<double, 2> min_max_dist(const hyper_rectangle&) const;
};

class source_node {
public:
  const KD_note &node;
  const std::unique_ptr<const source_node> left;
  const std::unique_ptr<const source_node> right;
  const arma::vec centroid;
  const double weight;
  const hyper_rectangle borders;

  source_node(const arma::mat&, const arma::vec&, const KD_note&);
};

class query_node {
public:
  const KD_note &node;
  const std::unique_ptr<const query_node> left;
  const std::unique_ptr<const query_node> right;
  const hyper_rectangle borders;

  query_node(const arma::mat&, const KD_note&);
};
