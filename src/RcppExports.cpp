// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// test_KD_note
Rcpp::List test_KD_note(const arma::mat& X, const arma::uword N_min);
RcppExport SEXP _mssm_test_KD_note(SEXP XSEXP, SEXP N_minSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type N_min(N_minSEXP);
    rcpp_result_gen = Rcpp::wrap(test_KD_note(X, N_min));
    return rcpp_result_gen;
END_RCPP
}
// naive
arma::vec naive(const arma::mat& X, const arma::vec ws, const arma::mat Y, unsigned int n_threads);
RcppExport SEXP _mssm_naive(SEXP XSEXP, SEXP wsSEXP, SEXP YSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type n_threads(n_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(naive(X, ws, Y, n_threads));
    return rcpp_result_gen;
END_RCPP
}
// FSKA
arma::vec FSKA(const arma::mat& X, const arma::vec& ws, const arma::mat& Y, const arma::uword N_min, const double eps, const unsigned int n_threads);
RcppExport SEXP _mssm_FSKA(SEXP XSEXP, SEXP wsSEXP, SEXP YSEXP, SEXP N_minSEXP, SEXP epsSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type N_min(N_minSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type n_threads(n_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(FSKA(X, ws, Y, N_min, eps, n_threads));
    return rcpp_result_gen;
END_RCPP
}
// sample_mv_normal
arma::mat sample_mv_normal(const arma::uword N, const arma::mat& Q, const arma::vec& mu);
RcppExport SEXP _mssm_sample_mv_normal(SEXP NSEXP, SEXP QSEXP, SEXP muSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uword >::type N(NSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu(muSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_mv_normal(N, Q, mu));
    return rcpp_result_gen;
END_RCPP
}
// sample_mv_tdist
arma::mat sample_mv_tdist(const arma::uword N, const arma::mat& Q, const arma::vec& mu, const double nu);
RcppExport SEXP _mssm_sample_mv_tdist(SEXP NSEXP, SEXP QSEXP, SEXP muSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uword >::type N(NSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu(muSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_mv_tdist(N, Q, mu, nu));
    return rcpp_result_gen;
END_RCPP
}
// pf_filter
Rcpp::List pf_filter(const arma::vec& Y, const arma::vec& cfix, const arma::vec& ws, const arma::vec& offsets, const arma::vec& disp, const arma::mat& X, const arma::mat& Z, const arma::uvec& time_indices_elems, const arma::uvec& time_indices_len, const arma::mat& F, const arma::mat& Q, const arma::mat& Q0, const std::string& fam, const arma::vec& mu0, const arma::uword n_threads, const double nu, const double covar_fac, const double ftol_rel, const arma::uword N_part, const std::string& what, const std::string& which_sampler, const std::string& which_ll_cp, const unsigned int trace, const arma::uword KD_N_max, const double aprx_eps);
RcppExport SEXP _mssm_pf_filter(SEXP YSEXP, SEXP cfixSEXP, SEXP wsSEXP, SEXP offsetsSEXP, SEXP dispSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP time_indices_elemsSEXP, SEXP time_indices_lenSEXP, SEXP FSEXP, SEXP QSEXP, SEXP Q0SEXP, SEXP famSEXP, SEXP mu0SEXP, SEXP n_threadsSEXP, SEXP nuSEXP, SEXP covar_facSEXP, SEXP ftol_relSEXP, SEXP N_partSEXP, SEXP whatSEXP, SEXP which_samplerSEXP, SEXP which_ll_cpSEXP, SEXP traceSEXP, SEXP KD_N_maxSEXP, SEXP aprx_epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type cfix(cfixSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type offsets(offsetsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type disp(dispSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type time_indices_elems(time_indices_elemsSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type time_indices_len(time_indices_lenSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type F(FSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q0(Q0SEXP);
    Rcpp::traits::input_parameter< const std::string& >::type fam(famSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const double >::type covar_fac(covar_facSEXP);
    Rcpp::traits::input_parameter< const double >::type ftol_rel(ftol_relSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type N_part(N_partSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type what(whatSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type which_sampler(which_samplerSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type which_ll_cp(which_ll_cpSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type KD_N_max(KD_N_maxSEXP);
    Rcpp::traits::input_parameter< const double >::type aprx_eps(aprx_epsSEXP);
    rcpp_result_gen = Rcpp::wrap(pf_filter(Y, cfix, ws, offsets, disp, X, Z, time_indices_elems, time_indices_len, F, Q, Q0, fam, mu0, n_threads, nu, covar_fac, ftol_rel, N_part, what, which_sampler, which_ll_cp, trace, KD_N_max, aprx_eps));
    return rcpp_result_gen;
END_RCPP
}
// run_Laplace_aprx
Rcpp::List run_Laplace_aprx(const arma::vec& Y, const arma::vec& cfix, const arma::vec& ws, const arma::vec& offsets, const arma::vec& disp, const arma::mat& X, const arma::mat& Z, const arma::uvec& time_indices_elems, const arma::uvec& time_indices_len, const arma::mat& F, const arma::mat& Q, const arma::mat& Q0, const std::string& fam, const arma::vec& mu0, const arma::uword n_threads, const double nu, const double covar_fac, const double ftol_rel, const arma::uword N_part, const std::string& what, const unsigned int trace, const arma::uword KD_N_max, const double aprx_eps, const double ftol_abs, const double la_ftol_rel, const double ftol_abs_inner, const double la_ftol_rel_inner, const unsigned maxeval, const unsigned maxeval_inner);
RcppExport SEXP _mssm_run_Laplace_aprx(SEXP YSEXP, SEXP cfixSEXP, SEXP wsSEXP, SEXP offsetsSEXP, SEXP dispSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP time_indices_elemsSEXP, SEXP time_indices_lenSEXP, SEXP FSEXP, SEXP QSEXP, SEXP Q0SEXP, SEXP famSEXP, SEXP mu0SEXP, SEXP n_threadsSEXP, SEXP nuSEXP, SEXP covar_facSEXP, SEXP ftol_relSEXP, SEXP N_partSEXP, SEXP whatSEXP, SEXP traceSEXP, SEXP KD_N_maxSEXP, SEXP aprx_epsSEXP, SEXP ftol_absSEXP, SEXP la_ftol_relSEXP, SEXP ftol_abs_innerSEXP, SEXP la_ftol_rel_innerSEXP, SEXP maxevalSEXP, SEXP maxeval_innerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type cfix(cfixSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type ws(wsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type offsets(offsetsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type disp(dispSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type time_indices_elems(time_indices_elemsSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type time_indices_len(time_indices_lenSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type F(FSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q0(Q0SEXP);
    Rcpp::traits::input_parameter< const std::string& >::type fam(famSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const double >::type covar_fac(covar_facSEXP);
    Rcpp::traits::input_parameter< const double >::type ftol_rel(ftol_relSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type N_part(N_partSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type what(whatSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type KD_N_max(KD_N_maxSEXP);
    Rcpp::traits::input_parameter< const double >::type aprx_eps(aprx_epsSEXP);
    Rcpp::traits::input_parameter< const double >::type ftol_abs(ftol_absSEXP);
    Rcpp::traits::input_parameter< const double >::type la_ftol_rel(la_ftol_relSEXP);
    Rcpp::traits::input_parameter< const double >::type ftol_abs_inner(ftol_abs_innerSEXP);
    Rcpp::traits::input_parameter< const double >::type la_ftol_rel_inner(la_ftol_rel_innerSEXP);
    Rcpp::traits::input_parameter< const unsigned >::type maxeval(maxevalSEXP);
    Rcpp::traits::input_parameter< const unsigned >::type maxeval_inner(maxeval_innerSEXP);
    rcpp_result_gen = Rcpp::wrap(run_Laplace_aprx(Y, cfix, ws, offsets, disp, X, Z, time_indices_elems, time_indices_len, F, Q, Q0, fam, mu0, n_threads, nu, covar_fac, ftol_rel, N_part, what, trace, KD_N_max, aprx_eps, ftol_abs, la_ftol_rel, ftol_abs_inner, la_ftol_rel_inner, maxeval, maxeval_inner));
    return rcpp_result_gen;
END_RCPP
}
// get_Q0
arma::mat get_Q0(const arma::mat& Qmat, const arma::mat& Fmat);
RcppExport SEXP _mssm_get_Q0(SEXP QmatSEXP, SEXP FmatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Qmat(QmatSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Fmat(FmatSEXP);
    rcpp_result_gen = Rcpp::wrap(get_Q0(Qmat, Fmat));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests();

static const R_CallMethodDef CallEntries[] = {
    {"_mssm_test_KD_note", (DL_FUNC) &_mssm_test_KD_note, 2},
    {"_mssm_naive", (DL_FUNC) &_mssm_naive, 4},
    {"_mssm_FSKA", (DL_FUNC) &_mssm_FSKA, 6},
    {"_mssm_sample_mv_normal", (DL_FUNC) &_mssm_sample_mv_normal, 3},
    {"_mssm_sample_mv_tdist", (DL_FUNC) &_mssm_sample_mv_tdist, 4},
    {"_mssm_pf_filter", (DL_FUNC) &_mssm_pf_filter, 25},
    {"_mssm_run_Laplace_aprx", (DL_FUNC) &_mssm_run_Laplace_aprx, 29},
    {"_mssm_get_Q0", (DL_FUNC) &_mssm_get_Q0, 2},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_mssm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
