#include "smoother.h"
#include "fast-kernel-approx.h"
#ifdef MSSM_PROF
#include "profile.h"
#endif

using std::to_string;

inline void check_smoother_input
  (problem_data &data, const std::vector<const arma::mat *> &particles,
   const std::vector<const arma::vec *> &weights){
  const unsigned n_periods = data.n_periods;
  if(n_periods != particles.size())
    throw std::invalid_argument(
        "smoother: invalid 'particles' (size " +
          to_string(particles.size()) + " but should be " +
          to_string(n_periods) + ")");
  if(n_periods != weights.size())
    throw std::invalid_argument(
        "smoother: invalid 'weights' (size " +
          to_string(weights.size()) + " but should be " +
          to_string(n_periods) + ")");
  for(auto &p : particles)
    if(!p or p->n_rows != particles.at(0)->n_rows)
      throw std::invalid_argument("smoother: un-equal rows in 'particles'");
}

/* functor used in inner loop in smoothing */
namespace {
  struct smoother_inner {
    const unsigned start, end, state_dim, N_old;
    const double * state_new;
    double * smooth_w;
    const double * new_w;
    const trans_obj * state_dist;
    const arma::mat &old_ps;
    const arma::vec &old_ws;

    void operator()(){
      state_new += start * state_dim;
      smooth_w  += start;
      new_w     += start;
      arma::vec work_mem(N_old);

      for(unsigned i = start; i < end;
          ++i, state_new += state_dim, ++smooth_w, ++new_w){
        std::fill(work_mem.begin(), work_mem.end(), 0.);

        const double *state_old = old_ps.memptr();
        double max_w = -std::numeric_limits<double>::infinity();
        const double *old_w = old_ws.begin();
        double *w_mem = work_mem.begin();
        for(unsigned j = 0; j < N_old;
            ++j, state_old += state_dim, ++old_w, ++w_mem){
          *w_mem =
            state_dist->operator()(state_new, state_old, state_dim, *old_w);
          if(*w_mem > max_w)
            max_w = *w_mem;
        }

        *smooth_w = log_sum_log(work_mem, max_w);
        *smooth_w = log_sum_log(*smooth_w, *new_w);
      }
    }
  };
}

std::vector<arma::vec> smoother
  (problem_data &data, const std::vector<const arma::mat *> &particles,
   const std::vector<const arma::vec *> &weights){
#ifdef MSSM_PROF
  profiler prof("smoother");
#endif

  check_smoother_input(data, particles, weights);

  const unsigned n_periods = data.n_periods,
    state_dim = particles.at(0)->n_rows;

  /* handle the last period */
  std::vector<arma::vec> out(n_periods);
  unsigned time = n_periods - 1L;
  out.at(time) = *weights.at(time);

  /* iterate from back to front */
  auto os = out.rbegin()       + 1L; /* smoothing weights */
  auto ps = particles.rbegin() + 1L; /* particles */
  auto ws = weights.rbegin()   + 1L; /* filter weights */
  thread_pool &pool = data.ctrl.get_pool();
  for(; ps != particles.rend(); ++ps, ++ws, ++os, --time){
    /* Given
     *   - smoothing weights for next period
     *   - particles for next periode
     *   - filter weights for this period
     *   - particle for this period
     *
     * compute the smoothing weights. */

    if(time % 25L == 0L)
      Rcpp::checkUserInterrupt();

    const arma::vec &old_ws = *(os - 1L); /* from smoothing distribution */
    const arma::vec &new_ws = **ws;       /* from filter distribution    */
    const unsigned N_old = old_ws.size(), N_new = new_ws.n_elem;

    /* copy and transform */
    auto state_dist = data.get_sta_dist<trans_obj>(time);
    arma::mat old_ps = **(ps - 1L), new_ps = **ps;
    state_dist->trans_inv_X(new_ps);
    state_dist->trans_inv_Y(old_ps);

    arma::vec &smooth_ws = *os;
    smooth_ws.resize(N_new);
    const double *state_new = new_ps.memptr();
    auto smooth_w = smooth_ws.begin();
    const double *new_w = new_ws.begin();

    const unsigned inc = N_new / (4L * pool.thread_count) + 1L;
    unsigned start = 0L, end = 0L;
    std::vector<std::future<void> > futures;
    futures.reserve(N_new / inc + 1L);

    for(; start < N_new; start = end){
      end = std::min(end + inc, N_new);
      smoother_inner task {
        start, end, state_dim, N_old, state_new, smooth_w,
        new_w, state_dist.get(), old_ps, old_ws };
      futures.push_back(pool.submit(std::move(task)));

    }

    for(auto &fu : futures)
      fu.get();

    normalize_log_weights(smooth_ws);
  }

  return out;
}

std::vector<arma::vec> smoother_aprx
  (problem_data &data, const std::vector<const arma::mat *> &particles,
   const std::vector<const arma::vec *> &weights){
#ifdef MSSM_PROF
  profiler prof("smoother-k-d");
#endif

  check_smoother_input(data, particles, weights);

  const unsigned n_periods = data.n_periods;
  const arma::uword N_min = data.ctrl.KD_N_min;
  const double eps = data.ctrl.aprx_eps;

  /* handle the last period */
  std::vector<arma::vec> out(n_periods);
  unsigned time = n_periods - 1L;
  out.at(time) = *weights.at(time);

  /* iterate from back to front */
  auto os = out.rbegin()       + 1L;
  auto ps = particles.rbegin() + 1L;
  auto ws = weights.rbegin()   + 1L;
  thread_pool &pool = data.ctrl.get_pool();
  for(; ps != particles.rend(); ++ps, ++ws, ++os, --time){
    if(time % 25L == 0L)
      Rcpp::checkUserInterrupt();

    arma::vec old_ws        = *(os - 1L);
    const arma::vec &new_ws = **ws;
    const unsigned N_new = new_ws.n_elem;

    /* copy and transform */
    auto state_dist = data.get_sta_dist<trans_obj>(time);
    arma::mat old_ps = **(ps - 1L), new_ps = **ps;
    state_dist->trans_inv_X(new_ps);
    state_dist->trans_inv_Y(old_ps);

    arma::vec &smooth_ws = *os;
    smooth_ws.resize(N_new);
    smooth_ws.fill(-std::numeric_limits<double>::infinity());

    /* Notice: we assume that the function is symmetrical in the two particle
     * arguments */
    auto permu_indices = FSKA_cpp<false>(
      smooth_ws, old_ps, new_ps, old_ws, N_min, eps, *state_dist,
      pool, true);

    /* permutate */
    smooth_ws = smooth_ws(permu_indices.Y_perm);

    /* add weights from source particle */
    {
      double *s = smooth_ws.begin();
      for(auto x : new_ws){
        *s = log_sum_log(*s, x);
        ++s;
      }
    }

    normalize_log_weights(smooth_ws);
  }

  return out;
}
