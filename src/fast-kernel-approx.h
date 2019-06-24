#include "kd-tree.h"
#include "dists.h"
#include "thread_pool.h"
#include <array>
#include <mutex>

template<bool has_extra = false>
class source_node {
public:
  const KD_note &node;
  const bool is_leaf = node.is_leaf();
  const std::unique_ptr<const source_node<has_extra> > left;
  const std::unique_ptr<const source_node<has_extra> > right;
  const arma::vec centroid;
  const double weight;
  const hyper_rectangle borders;
  const std::unique_ptr<arma::vec> extra;

  /* takes in the matrix with source particles, log weights, and the root
   * in the k-d tree */
  source_node(const arma::mat&, const arma::vec&, const KD_note&,
              const arma::mat*);
};

class query_node {
public:
  const KD_note &node;
  const bool is_leaf = node.is_leaf();
  const std::unique_ptr<const query_node> left;
  const std::unique_ptr<const query_node> right;
  const hyper_rectangle borders;
  const std::unique_ptr<std::mutex> idx_mutex;

  query_node(const arma::mat&, const KD_note&);
};

struct FSKA_cpp_permutation {
  arma::uvec X_perm;
  arma::uvec Y_perm;
};

/* Make extra computation and stores it in the fourth argument. First two
 * arguments are the source and query particles. The third argument is the
 * computed source particles extra information, the fourth is the query
 * particle extra information which can be written to and the last argument
 * is the log weight of the pair */
typedef std::function<void (const double *, const double *, const double *, double *,
                            const double)> FSKA_cpp_xtra_func;

/* Function to approximate otherwise O(N^2) computation. May permutate the the
 * referenced vectors and matrices. The returned object can be used to undo
 * the permutation. Use -infinity for uninitialized weights.
 * The function also takes two matrix pointers and a function to use on the
 * two matrices' columns given the two particles and log weight of the pair */
template<bool has_extra = false>
FSKA_cpp_permutation FSKA_cpp(
    arma::vec&, arma::mat&, arma::mat&, arma::vec&, const arma::uword,
    const double, const trans_obj&, thread_pool&,
    bool has_transformed = false, arma::mat *X_extra = nullptr,
    arma::mat *Y_extra = nullptr,
    FSKA_cpp_xtra_func extra_func = FSKA_cpp_xtra_func());
