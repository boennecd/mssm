#include "laplace.h"
#include <functional>
#include "nlopt.h"
#include <math.h>

#ifdef MSSM_PROF
#include "profile.h"
#endif

using namespace std::placeholders;

sym_band_mat get_concentration
  (const arma::mat &F, const arma::mat &Q, const arma::mat &Q0,
   const unsigned n_periods){
#ifdef MSSM_DEBUG
  if(n_periods < 1L)
    throw std::invalid_argument("not implemented for less than two periods");
#endif

  const unsigned dim_dia = Q.n_cols, dim_off = F.n_cols;
  sym_band_mat out(dim_dia, dim_off, n_periods);
  out.zeros();

  /* fill in off diagonal blocks */
  const arma::mat QiF = solve(Q, F), FtQiF = F.t() * QiF;
  {
    arma::mat negFtQi = -QiF.t();
    for(unsigned int i = 0; i < n_periods - 1L; ++i)
      out.set_upper_block(i, negFtQi);
  }

  /* fill in diagonal blocks */
  out.set_diag_block(0L, Q0.i() + FtQiF);
  {
    const arma::mat QipFtQiF = Q.i() + FtQiF;
    for(unsigned int i = 1; i < n_periods - 1L; ++i)
      out.set_diag_block(i, QipFtQiF);
  }
  out.set_diag_block(n_periods - 1L, Q.i());

  return out;
}

namespace {
  double call_laplace_approx(unsigned int, const double*, double*, void*);
  void   call_Q_constraint(unsigned, double*, unsigned, const double*,
                           double*, void*);
  void   call_F_constraint(unsigned, double*, unsigned, const double*,
                           double*, void*);

  /* Takes a pointer to values of upper triangular matrix and size of the
   * matrix and return the full dense symmetric matrix */
  arma::mat create_Q(const double *vals, const arma::SizeMat Q_size){
    arma::mat Q_new(Q_size);
    for(unsigned j = 0; j < Q_size.n_cols; ++j)
      for(unsigned i = 0; i <= j; ++i)
        Q_new(i, j) = *vals++;

    Q_new = arma::symmatu(Q_new);
    return Q_new;
  }

  /* class to estimate parameter with Laplace approximation */
  struct Laplace_util {
    problem_data &data;
    const arma::SizeMat Q_size = arma::size(data.get_Q()),
      F_size = arma::size(data.get_F());
    const bool has_disp = data.get_disp().n_elem > 0;
    const unsigned state_dim = data.get_sta_dist<cdist>(0L)->state_dim(),
      n_periods = data.n_periods,
      cfix_dim = data.get_cfix().n_elem,
      Q_dim = (Q_size.n_cols * (Q_size.n_cols + 1L)) / 2L,
      outer_dim = Q_dim + F_size.n_cols * F_size.n_rows + has_disp,
      cfix_n_disp = cfix_dim + has_disp;
    /* current maximum log-likelihood value */
    double max_ll = -std::numeric_limits<double>::infinity();
    /* counters for number of outer and inner evaluations */
    unsigned long it_inner = 0L, it_outer = 0L;
    /* parameters to nlopt */
    const double ftol_abs, ftol_rel, ftol_abs_inner, ftol_rel_inner;
    const unsigned maxeval, maxeval_inner;

    /* contains objects to evaluate conditional densities */
    const std::vector<std::unique_ptr<cdist> > obs_dists = ([&]{
      std::vector<std::unique_ptr<cdist> > out;
      out.reserve(data.n_periods);
      for(unsigned i = 0; i < data.n_periods; ++i)
        out.push_back(data.get_obs_dist(i));

      return out;
    })();

    /* matrix that contains random effect modes */
    arma::mat random_effects =
      arma::mat(state_dim, data.n_periods, arma::fill::zeros);

    /* pointer to concentraiton matrix */
    std::unique_ptr<sym_band_mat> concentration_mat;

    /* funciton used in mode approximation for multithreading. It writes to
     * the part of the gradient and Hessian for the state and returns a
     * vector and matrix with the gradient and Hessian terms related to fixed
     * coefficients. */
    struct mode_objective_inner_output {
      arma::vec obs_coef_grad_terms;
      arma::mat obs_coef_hess_terms;
      double ll_terms;
    };
    mode_objective_inner_output mode_objective_inner
      (const unsigned start, const unsigned end,
       const double * const state_mode_start, const bool do_hess,
       const comp_out what, double * const grad, sym_band_mat * neg_hess_mat){
      mode_objective_inner_output out;
      out.ll_terms = 0.;
      std::unique_ptr<double[]> obs_grad_hess;
      if(do_hess){
        unsigned int obs_grad_hess_size = cfix_n_disp * (cfix_n_disp + 1L);
        obs_grad_hess.reset(new double[obs_grad_hess_size]);
        std::fill(
          obs_grad_hess.get(), obs_grad_hess.get() + obs_grad_hess_size, 0.);

      }

      arma::mat work_mem =
        do_hess ? arma::mat(state_dim, state_dim) : arma::mat(0L, 0L);
      for(unsigned i = start; i < end; ++i){
        const unsigned inc = i * state_dim;
        const arma::vec state(state_mode_start + inc, state_dim);

        const std::unique_ptr<arma::vec> gr_ptr =
          do_hess ?
          std::unique_ptr<arma::vec>(
            new arma::vec(grad + cfix_dim + inc, state_dim, false)) :
          std::unique_ptr<arma::vec>();
        work_mem.zeros();

        auto &obs_dist = obs_dists.at(i);
        out.ll_terms +=
          obs_dist->log_density_state(state, gr_ptr.get(), &work_mem, what);

        if(do_hess)
          neg_hess_mat->set_diag_block(i, work_mem, -1.);

        if(do_hess)
          obs_dist->comp_stats_state_only(
              state, obs_grad_hess.get(), what);
      }

      if(do_hess){
        out.obs_coef_grad_terms = arma::vec(obs_grad_hess.get(), cfix_dim);
        out.obs_coef_hess_terms = arma::mat(
          obs_grad_hess.get() + cfix_n_disp, cfix_n_disp, cfix_n_disp);
        out.obs_coef_hess_terms = out.obs_coef_hess_terms.submat(
          0L, 0L, cfix_dim - 1L, cfix_dim - 1L);
      }

      return out;
    }

    /* Evaluates the log-likelihood and gradient w.r.t. the fixed coefficients
     * and random effects for given state space parameters. If 'do_hess' is
     * true then maximization is performed. */
    struct mode_objective_res {
      arma::vec mode;
      double ll; /* log-likelihood */
      bool failed;
    };
    mode_objective_res mode_objective
      (const arma::vec &params, const bool do_hess)
    {
#ifdef MSSM_DEBUG
      if(!concentration_mat)
        throw std::runtime_error("'concentration_mat' not set");
#endif

      it_inner++;
      const unsigned int n = random_effects.n_elem + cfix_dim;
      const double *x = params.memptr();

#ifdef MSSM_DEBUG
      if(n != random_effects.n_elem + cfix_dim)
        throw std::invalid_argument("wrong 'n' in 'mode_objective'");
#endif
      const bool verbose = data.ctrl.trace > 2L;
      if(verbose)
        Rcpp::Rcout << "Running mode objective... ";

      /* check whether we need to compute the gradient and Hessian */
      comp_out what = do_hess ? Hessian : log_densty;
      arma::vec grad_vec = do_hess ? arma::vec(n) : arma::vec();
      double * grad = do_hess ? grad_vec.memptr() : nullptr;
      if(do_hess)
        std::fill(grad, grad + n, 0.);

      /* maybe a copy of concentration matrix */
      std::unique_ptr<sym_band_mat> neg_hess;
      if(do_hess)
        neg_hess.reset(new sym_band_mat(*concentration_mat));

      /* update fixed coefficients */
      {
        arma::vec dum(x, cfix_dim);
        data.set_cfix(dum);
      }

      /* handle terms from observation's conditional density */
      double ll = 0.;
      const double * const state_mode_start = x + cfix_dim;
      std::vector<std::future<mode_objective_inner_output> > futures;
      const unsigned n_tasks = obs_dists.size();
      thread_pool &pool = data.ctrl.get_pool();

      /* TODO: smarter way to set this? */
      const unsigned inc = n_tasks / 5L + 1L;
      futures.reserve(n_tasks / inc + 1L);
      unsigned start = 0L, end = 0L;
      for(; start < n_tasks; start = end){
        end = std::min(end + inc, n_tasks);
        auto task = std::bind(
          &Laplace_util::mode_objective_inner, this, start, end,
          state_mode_start, do_hess, what, grad, neg_hess.get());
        futures.push_back(pool.submit(std::move(task)));
      }

      /* handle log-likelihood terms from state equation */
      arma::vec con_state = concentration_mat->mult(x + cfix_dim);
      {
        const double *xi = state_mode_start;
        for(auto z : con_state)
          ll -= z * *xi++ * .5;
      }

      /* get results from other threads */
      std::unique_ptr<arma::mat> obs_hess;
      if(do_hess)
        obs_hess.reset(new arma::mat(cfix_dim, cfix_dim, arma::fill::zeros));
      {
        arma::vec obs_grad =
          do_hess ? arma::vec(grad, cfix_dim, false) : arma::vec();
        for(auto &fu : futures){
          auto out = fu.get();
          if(do_hess){
            obs_grad  += out.obs_coef_grad_terms;
            *obs_hess += out.obs_coef_hess_terms;
          }
          ll += out.ll_terms;

        }
      }

      /* compute gradient terms from state equation */
      if(do_hess){
        double *gr = grad + cfix_dim;
        for(auto z : con_state)
          *gr++ -= z;
      }

      if(verbose){
        Rprintf("Objective value is %10.4f\n", ll);
        if(data.ctrl.trace > 3L){
          arma::vec t1(x, n);
          Rcpp::Rcout << "Mode: " << t1.t();
        }

      }

      if(!do_hess)
        return { params, ll, false };

      /* use step halving. First, find direction. Then perform maximization */
      const arma::vec direction = ([&] {
        const arma::span
          sp_grad(0L, cfix_dim - 1L), sp_state(cfix_dim, n - 1L);
        arma::vec out(n);
        out(sp_grad) =
          /* this part of the hessian matrix is not multiplied by -1 */
          arma::solve(*obs_hess, -grad_vec(sp_grad));

        /* TODO: implement another method to solve that does not assume that
         *       the hessian is (strictly) positive definite? */
        int info;
        out(sp_state) = neg_hess->solve(grad_vec(sp_state), info);

        if(info < 0)
          throw std::runtime_error("neg_hess->solve failed");

        if(info > 0){ /* TODO: assumes cholesky decomposition is still used! */
          /* fall back to gradient decent */
          out = grad_vec;

        }

        return out;
      })();

      double step_size = 1.;

      std::unique_ptr<mode_objective_res> out;
      const unsigned max_it = 50L;
      for(unsigned i = 0; i < max_it; ++i, step_size *= .5){
        arma::vec new_params = params + step_size * direction;
        if(verbose)
          Rprintf("Using step size %12.8f\n", step_size);
        out.reset(
          new mode_objective_res(mode_objective(new_params, false)));

        const double diff = out->ll - ll;
        if(diff > 0.)
          break;

        if(i == max_it - 1L){
          out->failed = true;
          break;

        }
      }

      return std::move(*out);
    }

    /* call the above until the criterion(s) is satisfied */
    bool failed_mode = false;
    mode_objective_res mode_objective(const arma::vec &params){
      mode_objective_res out {
        params,
        -std::numeric_limits<double>::infinity(),
        true };
      const unsigned long it_inner_start = it_inner;
      for(;;) {
        const double old_value = out.ll;
        out = mode_objective(out.mode, true);
        const double new_val = out.ll;

        if(!failed_mode and out.failed){
          failed_mode = true;
          Rcpp::Rcout << "Mode approxmation failed at least once\n";
        }

        const double adiff = std::abs(new_val - old_value);
        if(ftol_abs_inner > 0. and adiff < ftol_abs_inner)
          return out;
        if(ftol_rel_inner > 0. and
             adiff / (std::abs(new_val) + 1e-8) < ftol_rel_inner)
          return out;

        if(it_inner - it_inner_start > maxeval_inner)
          break;
      }

      throw std::runtime_error(
          "Failed to find mode within " + std::to_string(maxeval_inner) +
            " iterations");
    }

    /* quick (implementation wise) way to constraint a matrix to be positive
     * definte. TODO: do something smarter... */
    class Q_constraint_util {
      /* catch previous values to safe computations */
      arma::mat Q_old;
      arma::vec eigvals;
      unsigned call_number = 0L;
    public:

      double operator()(const double *d, const arma::SizeMat size){
        /* check whehter we have a new matrix */
        arma::mat Q_new = create_Q(d, size);
        const bool match = arma::size(Q_new) == arma::size(Q_old) and
          std::equal(Q_new.begin(), Q_new.end(), Q_old.begin());

        /* if so then update the eigen values */
        if(!match){
          Q_old = std::move(Q_new);
          eigvals = arma::eig_sym(Q_old);
          call_number = 0L;
        }

        /* return minus this eigen value*/
        unsigned this_num = call_number++;
        if(this_num >= size.n_cols)
          this_num = call_number = 0L;

        /* TODO: account for precision */
        return -eigvals(this_num) + std::numeric_limits<double>::epsilon();
      }
    };

    Q_constraint_util Q_constraint_u;

    /* method for NLOPT to call */
    void Q_constraint
      (unsigned m, double *result, unsigned n, const double *x, double *grad,
       void *f_data)
    {
#ifdef MSSM_DEBUG
      if(m != Q_size.n_cols)
        throw std::invalid_argument("invalid m in 'Q_constraint'");
#endif

      for(unsigned i = 0; i < Q_size.n_cols; ++i, ++result)
        *result = Q_constraint_u(x + F_size.n_cols * F_size.n_rows, Q_size);
    }

    /* constraint F such that the system is stationary.
     * constraint F such that it is not too close to being singular (required
     * for some computation but could be avoided). TODO: avoid this... */
    void F_constraint
      (unsigned m, double *result, unsigned n, const double *x, double *grad,
       void *f_data) const
    {
#ifdef MSSM_DEBUG
      if(m != 2L)
        throw std::invalid_argument("m != 2 in 'F_constraint'");
#endif
      arma::mat F(x, F_size.n_rows, F_size.n_cols);
      arma::cx_vec eigs_vals = arma::eig_gen(F);
      double minx = std::numeric_limits<double>::infinity(),
             maxv = 0.;
      for(auto d : eigs_vals){
        const double da = std::sqrt(d.real() * d.real() + d.imag() * d.imag());
        if(da < minx)
          minx = da;
        if(da > maxv)
          maxv = da;
      }

      const double
        tol = std::numeric_limits<double>::epsilon() * F_size.n_cols * maxv;

      * result       = -minx     + tol;
      *(result + 1L) =  maxv - 1 + tol;
    }

    /* function used for multithreading in Laplace approximation. Writes to
     * the i'th block of the concentration matrix and returns the
     * log-likelihood terms. */
    double laplace_approx_inner
      (const unsigned start, const unsigned end,
       double * const state_mode_start){
      double out = 0.;

      arma::mat work_mem(state_dim, state_dim);
      arma::vec dummy(state_dim);
      for(unsigned i = start; i < end; ++i){
        const unsigned inc = i * state_dim;
        arma::vec state(state_mode_start + inc, state_dim, false);

        work_mem.zeros();
        dummy.zeros(); /* TODO: needed? */

        out += obs_dists.at(i)->log_density_state(
          state, &dummy, &work_mem, Hessian);

        concentration_mat->set_diag_block(i, work_mem, -1.);
      }

      return out;
    }

    /* This function computes the approximate log-likelihood given fixed
     * coefficients and random effects. The Hessian will have terms from both
     * the observed outcomes and the state equation. */
    double laplace_approx(
        unsigned int n, const double *x, double *grad, void *data_in){
#ifdef MSSM_DEBUG
      if(n != outer_dim)
        throw std::invalid_argument("wrong 'n' in 'laplace_approx'");
#endif

      /* check constraints. TODO: ok to return -Inf? */
      const double * const Qmem = x + F_size.n_cols * F_size.n_rows;
      {
        const double test_failed_res =
          -std::numeric_limits<double>::infinity();

        Q_constraint_util Qu;
        for(unsigned i = 0; i < Q_size.n_cols; ++i)
          if(Qu(Qmem, Q_size) >= 0.)
            return test_failed_res;

        std::array<double, 2L> F_test_val;
        F_constraint(2L, F_test_val.data(), n, x, nullptr, nullptr);
        if(F_test_val[0L] >= 0. or F_test_val[1L] >= 0.)
          return test_failed_res;
      }

      /* set parameters */
      {
        arma::mat F_new(x, data.get_F().n_rows, data.get_F().n_cols);
        arma::mat Q_new = create_Q(Qmem, Q_size);

        data.set_F(F_new);
        data.set_Q(Q_new);

        arma::mat Q0_new = get_Q0(Q_new, F_new);
        data.set_Q0(Q0_new);

        if(has_disp){
          arma::vec dum(x + n - 1L, 1L);
          data.set_disp(dum);
        }

        if(arma::rank(Q0_new) < Q0_new.n_cols)
          return -std::numeric_limits<double>::infinity();
      }

      if(it_outer % 10L == 0L)
        Rcpp::checkUserInterrupt();
      it_outer++;

      const bool verbose = ([&]{
        if(data.ctrl.trace == 1L and (it_outer -1L) % 10L == 0L)
          return true;
        else if(data.ctrl.trace > 1L)
          return true;

        return false;
      })();
      if(verbose){
        Rprintf("It: %5d: Making Laplace approximation at Q\n", it_outer);
        Rcpp::Rcout << data.get_Q();

        if(has_disp)
          Rprintf(", dispersion %16.6f\n", data.get_disp()(0L));

        Rcpp::Rcout << ", F\n"
                    << data.get_F()
                    << ", and Q0\n"
                    << data.get_Q0();
      }

      /* set concentration matrix */
      concentration_mat.reset(new sym_band_mat(get_concentration(
        data.get_F(), data.get_Q(), data.get_Q0(), data.n_periods)));

      /* make log-likelihood approximation. First, find the mode */
      const unsigned n_inner = random_effects.n_elem + cfix_dim;
      std::unique_ptr<double[]> val(new double[n_inner]);

      /* set starting values  */
      {
        double *d = val.get();
        const arma::vec cfix_copy = data.get_cfix();
        const double *z = cfix_copy.memptr();
        for(unsigned i = 0; i < cfix_dim; ++i)
          *d++ = *z++;
        z = random_effects.memptr();
        for(unsigned i = 0; i < random_effects.n_elem; ++i)
          *d++ = *z++;
      }

      const unsigned long it_inner_old = it_inner;
      {
        arma::vec val_vec(val.get(), n_inner, false);
        auto mout = mode_objective(val_vec);
        val_vec = mout.mode;
      }

      if(verbose)
        Rprintf("It: %5d: %5d inner iterations\n",
                it_outer, it_inner - it_inner_old);

      /* update values for modes and cfix */
      double * const state_mode_start = val.get() + cfix_dim;
      {
        arma::vec dum(val.get(), cfix_dim);
        data.set_cfix(dum);
        double *z = state_mode_start;
        for(auto &r : random_effects)
          r = *z++;

      }

      /* make log-likehood approximation at the mode. Add terms from state
       * equation */
      auto get_abs_ldeter = [&]{
        int info;
        double out = concentration_mat->ldeterminant(info);
        if(info < 0)
          throw std::runtime_error(
              "'ldeterminant' returned info " + std::to_string(info));
        return (info != 0) ? std::numeric_limits<double>::infinity() : out;
      };

      double ll = 0.;
      {
        arma::vec state(state_mode_start, random_effects.n_elem, false),
        con_state = concentration_mat->mult(state);
        double *s = state.memptr();
        for(auto c : con_state)
          ll -= c * *s++ * .5;
        ll += get_abs_ldeter() * .5;
        if(std::isinf(ll))
          return -std::numeric_limits<double>::infinity();
      }

      /* compute terms from observation's conditional density and update the
       * Hessian */
      {
        thread_pool &pool = data.ctrl.get_pool();
        std::vector<std::future<double> > futures;

        const unsigned n_tasks = obs_dists.size();
        /* TODO: smarter way to set this? */
        const unsigned inc = n_tasks / 5L + 1L;
        futures.reserve(n_tasks / inc + 1L);
        unsigned start = 0L, end = 0L;
        for(; start < n_tasks; start = end){
          end = std::min(end + inc, n_tasks);
          auto task = std::bind(
            &Laplace_util::laplace_approx_inner, this, start, end,
            state_mode_start);

          futures.push_back(pool.submit(std::move(task)));

        }

        for(auto &fu: futures)
          ll += fu.get();
      }

      /* add the final term from the Hessian */
      ll -= get_abs_ldeter() * .5;

      if(verbose){
        if(ll > max_ll)
          max_ll = ll;

        Rprintf("It: %5d: cfix is ", it_outer);
        Rcpp::Rcout << data.get_cfix().t();
        Rprintf("It: %5d: Log-likelihood approximation (current, max): %14.4f, %14.4f\n",
                it_outer, ll, max_ll);
      }

      return ll;
    }

  class get_nlopt_problem {
  public:
    nlopt_opt opt, opt_inner;
    get_nlopt_problem(const unsigned n):
      opt      (nlopt_create(NLOPT_AUGLAG  , n)),
      opt_inner(nlopt_create(NLOPT_LN_SBPLX, n)) { }

    ~get_nlopt_problem(){
      nlopt_destroy(opt_inner);
      nlopt_destroy(opt);
    }
  };

  public:
    Laplace_util
    (problem_data &data, const double ftol_abs, const double ftol_rel,
     const double ftol_abs_inner, const double ftol_rel_inner,
     const unsigned maxeval, const unsigned maxeval_inner):
    data(data), ftol_abs(ftol_abs), ftol_rel(ftol_rel),
    ftol_abs_inner(ftol_abs_inner), ftol_rel_inner(ftol_rel_inner),
    maxeval(maxeval), maxeval_inner(maxeval_inner) { }

    /* uses Laplace approximation to estimate the parameters */
    Laplace_aprx_output operator()(){
      max_ll = -std::numeric_limits<double>::infinity();
      it_inner = it_outer = 0L;
      const bool verbose = data.ctrl.trace > 0L;
      if(verbose){
        std::string msg =
          "Estimating parameters with Laplace approximation. Settings are:\n";
        msg += "ftol_rel        %17.10f\n";
        msg += "ftol_abs        %17.10f\n";
        msg += "ftol_rel-inner  %17.10f\n";
        msg += "ftol_abs-inner  %17.10f\n";
        msg += "maxeval         %6d\n";
        msg += "maxeval-inner   %6d\n";
        Rprintf(msg.c_str(), ftol_rel, ftol_abs, ftol_rel_inner,
                ftol_abs_inner, maxeval, maxeval_inner);
      }

      /* setup parameters */
      std::unique_ptr<double[]> vals =
        std::unique_ptr<double[]>(new double[outer_dim]);
      {
        double *d = vals.get();
        for(auto x : data.get_F())
          *d++ = x;
        const arma::mat &Q = data.get_Q();
        for(unsigned j = 0; j < Q.n_cols; ++j)
          for(unsigned i = 0; i <= j; ++i)
            *d++ = Q(i, j);

        if(has_disp)
          *d = data.get_disp()(0L);

      }

      /* setup problem */
      get_nlopt_problem probs(outer_dim);
      nlopt_opt &opt = probs.opt,
          &opt_inner = probs.opt_inner;
      nlopt_set_ftol_abs(opt_inner, ftol_abs);
      nlopt_set_ftol_rel(opt_inner, ftol_rel);
      nlopt_set_maxeval(opt_inner, maxeval);

      nlopt_set_max_objective(opt, call_laplace_approx, this);
      nlopt_set_local_optimizer(opt, opt_inner);

      /* add constraints */
      std::unique_ptr<double[]> Q_constraint_tol(new double[Q_size.n_cols]);
      std::fill(
        Q_constraint_tol.get(), Q_constraint_tol.get() + Q_size.n_cols, 0.);
      nlopt_add_inequality_mconstraint
        (opt, Q_size.n_cols, call_Q_constraint, this,
         Q_constraint_tol.get());

      std::array<double, 2L> F_constraint_tol = { 0., 0. };
      nlopt_add_inequality_mconstraint
        (opt, 2L, call_F_constraint, this, F_constraint_tol.data());

      std::unique_ptr<double[]> lbs(new double[outer_dim]);
      if(has_disp){
        /* add positivity constraint to dispersion parameter */
        lbs.reset(new double[outer_dim]);
        std::fill(lbs.get(), lbs.get() + outer_dim - 1L, -HUGE_VAL);
        /* TODO: replace eps with something else... */
        constexpr double eps = std::numeric_limits<double>::epsilon();
        lbs[outer_dim - 1L] = eps;

        /* add positivity constraints to diagonal elements of Q */
        const unsigned Q_start = F_size.n_cols * F_size.n_rows;
        double *lbs_i = lbs.get() + Q_start;
        for(unsigned i = 0; i < Q_size.n_cols; ++i){
          *lbs_i = eps;
          lbs_i += i + 2L;
        }

        nlopt_set_lower_bounds(opt, lbs.get());

      }

      /* solve problem */
      double maxf;
      int nlopt_result_code = nlopt_optimize(opt, vals.get(), &maxf);

      /* setup output object and return */
      Laplace_aprx_output out;
      out.cfix = data.get_cfix(); /* TODO: is this from the final iteration? */
      out.F = arma::mat(vals.get(), F_size.n_rows, F_size.n_cols);
      out.Q = create_Q(vals.get() + out.F.n_elem, Q_size);
      out.logLik = maxf;
      out.n_it = it_outer;
      out.code = nlopt_result_code;
      if(has_disp)
        out.disp = data.get_disp();

      return out;
    }

    friend double call_laplace_approx
      (unsigned int, const double*, double*, void*);
    friend void call_Q_constraint
      (unsigned, double*, unsigned, const double*, double*, void*);
    friend void call_F_constraint
      (unsigned, double*, unsigned, const double*, double*, void*);
  };

  double call_laplace_approx
    (unsigned int n, const double *x, double *grad, void *data_in){
    Laplace_util *obj = (Laplace_util*)(data_in);
    return obj->laplace_approx(n, x, grad, nullptr);
  }
  void call_Q_constraint
    (unsigned m, double *result, unsigned n, const double *x, double *grad,
     void *f_data){
    Laplace_util *obj = (Laplace_util*)(f_data);
    obj->Q_constraint(m, result, n, x, grad, f_data);
  }
  void call_F_constraint
    (unsigned m, double *result, unsigned n, const double *x, double *grad,
     void *f_data){
    Laplace_util *obj = (Laplace_util*)(f_data);
    obj->F_constraint(m, result, n, x, grad, f_data);
  }
}

Laplace_aprx_output Laplace_aprx
  (problem_data &data, const double ftol_abs, const double ftol_rel,
   const double ftol_abs_inner, const double ftol_rel_inner,
   const unsigned maxeval, const unsigned maxeval_inner){
#ifdef MSSM_PROF
  profiler prof("Laplace");
#endif

  return Laplace_util(data, ftol_abs, ftol_rel, ftol_abs_inner, ftol_rel_inner,
                      maxeval, maxeval_inner)();
}
