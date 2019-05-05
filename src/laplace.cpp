#include "laplace.h"
#include <functional>
#include "nlopt.h"
#include <math.h>

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
  double call_mode_objective(unsigned int, const double*, double*, void*);
  double call_laplace_approx(unsigned int, const double*, double*, void*);
  double call_Q_constraint(unsigned int, const double*, double*, void*);
  double call_F_constraint1(unsigned int, const double*, double*, void*);
  double call_F_constraint2(unsigned int, const double*, double*, void*);

  /* Takes a pointer to values of upper diagonal and size of the matrix and
   * return the full dense symmetric matrix */
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
    const unsigned state_dim = data.get_sta_dist<cdist>(0L)->state_dim(),
      n_periods = data.n_periods,
      cfix_dim = data.get_cfix().n_elem,
      Q_dim = (Q_size.n_cols * (Q_size.n_cols + 1L)) / 2L,
      outer_dim = Q_dim + F_size.n_cols * F_size.n_rows;
    /* current maximum log-likelihood value */
    double max_ll = -std::numeric_limits<double>::infinity();
    /* counters for number of outer and inner evaluations */
    unsigned it_inner = 0L, it_outer = 0L;
    /* parameters to nlopt */
    const double ftol_abs, ftol_rel, ftol_abs_inner, ftol_rel_inner;
    const unsigned maxeval, maxeval_inner;

    /* matrix that contains random effect modes */
    arma::mat random_effects =
      arma::mat(state_dim, data.n_periods, arma::fill::zeros);

    /* pointer to concentraiton matrix */
    std::unique_ptr<sym_band_mat> concentration_mat;

    /* Evaluates the log-likelihood and gradient w.r.t. the fixed coefficients
     * and random effects for given state space parameters. Used for mode
     * approximation */
    double mode_objective(
        unsigned int n, const double *x, double *grad, void *data_in)
    {
      it_inner++;

#ifdef MSSM_DEBUG
      if(n != random_effects.n_elem + cfix_dim)
        throw std::invalid_argument("wrong 'n' in 'mode_objective'");
#endif
      const bool verbose = data.ctrl.trace > 2L;
      if(verbose)
        Rcpp::Rcout << "Running mode objective... ";

      /* check whether we need to compute the gradient */
      const bool do_grad = grad;
      comp_out what = grad ? gradient : log_densty;
      if(do_grad)
        std::fill(grad, grad + n, 0.);

      /* update fixed coefficients */
      {
        arma::vec dum(x, cfix_dim);
        data.set_cfix(dum);
      }

      /* handle terms from observation equation */
      double ll = 0.;
      const double * const state_mode_start = x + cfix_dim;
      for(unsigned i = 0; i < n_periods; ++i){
        const unsigned inc = i * state_dim;
        arma::vec state(state_mode_start + inc, state_dim);

        const std::unique_ptr<arma::vec> gr_ptr =
          do_grad ?
          std::unique_ptr<arma::vec>(
            new arma::vec(grad + cfix_dim + inc, state_dim, false)) :
          std::unique_ptr<arma::vec>();

        auto obs_dist = data.get_obs_dist(i);
        ll += obs_dist->log_density_state(state, gr_ptr.get(), nullptr, what);

        if(!do_grad)
          continue;

        obs_dist->comp_stats_state_only(state, grad, what);

      }

      /* handle terms from state equation */
#ifdef MSSM_DEBUG
      if(!concentration_mat)
        throw std::runtime_error("'concentration_mat' not set");
#endif

      arma::vec con_state = concentration_mat->mult(x + cfix_dim);
      {
        const double *xi = state_mode_start;
        for(auto z : con_state)
          ll -= z * *xi++ * .5;
      }

      if(do_grad){
        double *gr = grad + cfix_dim;
        for(auto z : con_state)
          *gr++ -= z;
      }

      if(verbose)
        Rprintf("Objective value is %10.4f\n", ll);

      return ll;
    }

    /* quick (implementation wise) way to constraint a postiive definte
     * matrix. TODO: do something smarter... */
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

        return
          -eigvals(this_num) + std::sqrt(std::numeric_limits<double>::epsilon());
      }
    };

    Q_constraint_util Q_constraint_u;

    /* method for NLOPT to call */
    double Q_constraint(
        unsigned int n, const double *x, double *grad, void *data_in)
    {
      return Q_constraint_u(x + F_size.n_cols * F_size.n_rows, Q_size);
    }

    /* constraint F such that the system is stationary */
    double F_constraint1(
        unsigned int n, const double *x, double *grad, void *data_in) const
    {
      arma::mat F(x, F_size.n_rows, F_size.n_cols);
      arma::cx_vec eigs_vals = arma::eig_gen(F);
      double maxv = 0.;
      for(auto d : eigs_vals){
        const double da = std::sqrt(d.real() * d.real() + d.imag() * d.imag());
        if(da > maxv)
          maxv = da;
      }

      return maxv - 1 + std::pow(std::numeric_limits<double>::epsilon(), .25);
    }

    /* constraint F such that it is not too close to being singular (required
     * for some computation but could be avoided). TODO: avoid this... */
    double F_constraint2(
        unsigned int n, const double *x, double *grad, void *data_in) const
    {
      arma::mat F(x, F_size.n_rows, F_size.n_cols);
      arma::vec eigs_vals = arma::real(arma::eig_gen(F));
      double minx = std::numeric_limits<double>::infinity();
      for(auto d : eigs_vals){
        const double da = std::abs(d);
        if(da < minx)
          minx = da;
      }

      return -minx + std::pow(std::numeric_limits<double>::epsilon(), .25);
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
        Q_constraint_util Qu;
        for(unsigned i = 0; i < Q_size.n_cols; ++i)
          if(Qu(Qmem, Q_size) >= 0.)
            return -std::numeric_limits<double>::infinity();
        if(F_constraint1(n, x, grad, data_in) >= 0.)
          return -std::numeric_limits<double>::infinity();
        if(F_constraint2(n, x, grad, data_in) >= 0.)
          return -std::numeric_limits<double>::infinity();
      }

      /* set parameters */
      {
        arma::mat F_new(x, data.get_F().n_rows, data.get_F().n_cols);
        arma::mat Q_new = create_Q(Qmem, Q_size);

        data.set_F(F_new);
        data.set_Q(Q_new);

        arma::mat Q0_new = get_Q0(Q_new, F_new);
        data.set_Q0(Q0_new);

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
        Rcpp::Rcout << data.get_Q()
                    << ", F\n"
                    << data.get_F()
                    << ", and Q0\n"
                    << data.get_Q0();
      }

      /* set concentration matrix */
      concentration_mat.reset(new sym_band_mat(get_concentration(
        data.get_F(), data.get_Q(), data.get_Q0(), data.n_periods)));

      /* make log-likelihood approximation. First, find the mode */
      nlopt_opt opt;
      unsigned n_inner = random_effects.n_elem + cfix_dim;
      opt = nlopt_create(NLOPT_LD_TNEWTON, n_inner);

      nlopt_set_max_objective(opt, call_mode_objective, this);
      nlopt_set_ftol_abs(opt, ftol_abs_inner);
      nlopt_set_ftol_rel(opt, ftol_rel_inner);
      nlopt_set_maxeval(opt, maxeval_inner);

      /* set starting values  */
      std::unique_ptr<double[]> val(new double[n_inner]);
      {
        double *d = val.get();
        const double *z = data.get_cfix().memptr();
        for(unsigned i = 0; i < cfix_dim; ++i)
          *d++ = *z++;
        z = random_effects.memptr();
        for(unsigned i = 0; i < random_effects.n_elem; ++i)
          *d++ = *z++;
      }

      double maxf;
      const unsigned it_inner_old = it_inner;
      int nlopt_result_code = nlopt_optimize(opt, val.get(), &maxf);
      nlopt_destroy(opt);
      if(nlopt_result_code < 1L or nlopt_result_code > 4L)
        throw std::runtime_error(
            "laplace_approx: Got code " + std::to_string(nlopt_result_code) +
              " from nlopt");

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
        if(isinf(ll))
          return -std::numeric_limits<double>::infinity();
      }

      /* compute terms from observations conditional density and update the
       * Hessian */
      arma::mat work_mem(state_dim, state_dim);
      arma::vec dummy(state_dim);
      for(unsigned i = 0; i < n_periods; ++i){
        const unsigned inc = i * state_dim;
        arma::vec state(state_mode_start + inc, state_dim, false);

        dummy.zeros();
        work_mem.zeros();
        auto obs_dist = data.get_obs_dist(i);
        ll += obs_dist->log_density_state(
          state, &dummy, &work_mem, Hessian);

        concentration_mat->set_diag_block(i, work_mem, -1.);
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

      }

      /* setup problem */
      nlopt_opt opt, opt_inner;
      opt = nlopt_create(NLOPT_AUGLAG, outer_dim);
      opt_inner = nlopt_create(NLOPT_LN_SBPLX, outer_dim);
      nlopt_set_ftol_abs(opt_inner, ftol_abs);
      nlopt_set_ftol_rel(opt_inner, ftol_rel);
      nlopt_set_maxeval(opt_inner, maxeval);

      nlopt_set_max_objective(opt, call_laplace_approx, this);
      nlopt_set_local_optimizer(opt, opt_inner);

      /* add constraints */
      /* TODO: replace by 'nlopt_add_inequality_mconstraint' */
      for(unsigned i = 0; i < Q_size.n_cols; ++i)
        nlopt_add_inequality_constraint(
          opt, call_Q_constraint, this, 0);
      /* TODO: replace by 'nlopt_add_inequality_mconstraint' */
      nlopt_add_inequality_constraint(
        opt, call_F_constraint1, this, 0);
      nlopt_add_inequality_constraint(
        opt, call_F_constraint2, this, 0);

      /* solve problem */
      double maxf;
      int nlopt_result_code = nlopt_optimize(opt, vals.get(), &maxf);
      nlopt_destroy(opt);
      nlopt_destroy(opt_inner);

      /* setup output object and return */
      Laplace_aprx_output out;
      out.cfix = data.get_cfix(); /* TODO: is this from the final iteration? */
      out.F = arma::mat(vals.get(), F_size.n_rows, F_size.n_cols);
      out.Q = create_Q(vals.get() + out.F.n_elem, Q_size);
      out.logLik = maxf;
      out.n_it = it_outer;
      out.code = nlopt_result_code;

      return out;
    }

    friend double call_mode_objective
      (unsigned int, const double*, double*, void*);
    friend double call_laplace_approx
      (unsigned int, const double*, double*, void*);
    friend double call_Q_constraint
      (unsigned int, const double*, double*, void*);
    friend double call_F_constraint1
      (unsigned int, const double*, double*, void*);
    friend double call_F_constraint2
      (unsigned int, const double*, double*, void*);
  };

  double call_mode_objective
    (unsigned int n, const double *x, double *grad, void *data_in){
    Laplace_util *obj = (Laplace_util*)(data_in);
    return obj->mode_objective(n, x, grad, nullptr);
  }
  double call_laplace_approx
    (unsigned int n, const double *x, double *grad, void *data_in){
    Laplace_util *obj = (Laplace_util*)(data_in);
    return obj->laplace_approx(n, x, grad, nullptr);
  }
  double call_Q_constraint
    (unsigned int n, const double *x, double *grad, void *data_in){
    Laplace_util *obj = (Laplace_util*)(data_in);
    return obj->Q_constraint(n, x, grad, nullptr);
  }
  double call_F_constraint1
    (unsigned int n, const double *x, double *grad, void *data_in){
    Laplace_util *obj = (Laplace_util*)(data_in);
    return obj->F_constraint1(n, x, grad, nullptr);
  }
  double call_F_constraint2
    (unsigned int n, const double *x, double *grad, void *data_in){
    Laplace_util *obj = (Laplace_util*)(data_in);
    return obj->F_constraint2(n, x, grad, nullptr);
  }
}

Laplace_aprx_output Laplace_aprx
  (problem_data &data, const double ftol_abs, const double ftol_rel,
   const double ftol_abs_inner, const double ftol_rel_inner,
   const unsigned maxeval, const unsigned maxeval_inner){
  return Laplace_util(data, ftol_abs, ftol_rel, ftol_abs_inner, ftol_rel_inner,
                      maxeval, maxeval_inner)();
}
