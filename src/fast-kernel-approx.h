#include "kd-tree.h"
#include <array>
#include <mutex>

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
  const std::unique_ptr<std::mutex> idx_mutex;

  query_node(const arma::mat&, const KD_note&);
};
