#include "utils.h"
#include "blas-lapack.h"

using std::invalid_argument;

inline arma::mat set_chol_(const arma::mat &X)
{
  return arma::trimatu(arma::chol(X));
}

chol_decomp::chol_decomp(const arma::mat &X):
  X(X), chol_(set_chol_(X)) { }

static constexpr char C_U = 'U', C_N = 'N', C_L = 'L';
static constexpr double D_one = 1., D_zero = 0.;
static constexpr int I_one = 1L;

inline void solve_half_(const arma::mat &chol_, arma::mat &X,
                        const bool transpose){
#ifdef MSSM_DEBUG
  if(X.n_rows != chol_.n_cols)
    throw invalid_argument("dims do not match with 'chol_'");
#endif
  int n = X.n_cols, m = X.n_rows;
  char trans = transpose ? 'T' : 'N';
  dtrsm(
      &C_L, &C_U, &trans, &C_N, &m, &n, &D_one, chol_.begin(), &m, X.begin(),
      &m);
}

void chol_decomp::solve_half(arma::mat &X, const bool transpose) const {
  solve_half_(chol_, X, !transpose);
}

void chol_decomp::solve_half(arma::vec &x, const bool transpose) const {
  arma::mat dum(x.begin(), x.n_elem, 1L, false);
  solve_half_(chol_, dum, !transpose);
}

arma::mat chol_decomp::solve_half(const arma::mat &X, const bool transpose) const
{
  arma::mat out = X;
  solve_half(out, transpose);
  return out;
}

arma::vec chol_decomp::solve_half(const arma::vec &x, const bool transpose) const
{
  arma::vec out = x;
  solve_half(out, transpose);
  return out;
}

void chol_decomp::mult_half(arma::mat &Z, const bool transpose) const
{
  char trans = transpose ? 'N' : 'T';
  int m = Z.n_rows, n = Z.n_cols;
#ifdef MSSM_DEBUG
  if(m != (int)chol_.n_cols)
    throw std::invalid_argument("Invalid 'Z' in 'chol_decomp::mult_half'");
#endif

  dtrmm(
    &C_L, &C_U, &trans, &C_N, &m, &n, &D_one, chol_.memptr(), &m,
    Z.memptr(), &m);
}

void chol_decomp::mult_half(arma::vec &z, const bool transpose) const
{
  char trans = transpose ? 'N' : 'T';
  int m = z.n_elem;
#ifdef MSSM_DEBUG
  if(m != (int)chol_.n_cols)
    throw std::invalid_argument("Invalid 'z' in 'chol_decomp::mult_half'");
#endif

  dtrmm(
      &C_L, &C_U, &trans, &C_N, &m, &I_one, &D_one, chol_.memptr(), &m,
      z.memptr(), &m);
}

arma::mat chol_decomp::mult_half
  (const arma::mat &Z, const bool transpose) const
{
  arma::mat out = Z;
  mult_half(out, transpose);
  return out;
}

arma::vec chol_decomp::mult_half
  (const arma::vec &z, const bool transpose) const
{
  arma::vec out = z;
  mult_half(out, transpose);
  return out;
}

void chol_decomp::solve(arma::mat &out) const
{
#ifdef MSSM_DEBUG
  if(out.n_rows != chol_.n_cols)
    throw invalid_argument("dims do not match with 'chol_'");
#endif

  int n = chol_.n_cols, nrhs = out.n_cols, info;
  dpotrs(&C_U, &n, &nrhs, chol_.memptr(), &n, out.memptr(),
         &n, &info);
  if(info != 0)
    throw std::runtime_error("'dpotrs' failed with info " +
                             std::to_string(info));
}

arma::mat chol_decomp::solve(const arma::mat &X) const
{
  arma::mat out = X;
  solve(out);
  return out;
}
arma::vec chol_decomp::solve(const arma::vec &x) const
{
  arma::vec out = x;
  arma::mat dum(out.memptr(), out.n_elem, 1L, false);
  solve(dum);
  return out;
}

const arma::mat& chol_decomp::get_inv() const
{
  /* set inverse */
  std::call_once(*is_inv_set, [&](){
    arma::mat &inv_mat = *inv_;
    inv_mat = chol_;
    int n = chol_.n_cols, info;
    char upper = 'U';
    lapack::dpotri(&upper, &n, inv_mat.memptr(), &n, &info);
    if(info != 0)
      throw std::runtime_error("'dpotri' failed with info " +
                               std::to_string(info));

    inv_mat = arma::symmatu(inv_mat);
  });

  return *inv_;
}

void arma_dsyr(arma::mat &A, const arma::vec &x, const double alpha)
{
  int n = A.n_cols;
#ifdef MSSM_DEBUG
  if(A.n_cols != A.n_rows)
    throw invalid_argument("invalid 'A'");
  if(x.n_elem != A.n_cols)
    throw invalid_argument("invalid 'x'");
#endif

  dsyr(
    &C_U, &n, &alpha, x.memptr(), &I_one, A.memptr(), &n);
}

void arma_dsyr(arma::mat &A, const arma::vec &x, const double alpha,
               const int n)
{
  const int lda = A.n_rows;
#ifdef MSSM_DEBUG
  if(A.n_cols != A.n_rows)
    throw invalid_argument("arma_dsyr: invalid 'A'");
  if((int)x.n_elem != n)
    throw invalid_argument("arma_dsyr: invalid 'x'");
  if(n > lda)
    throw invalid_argument("arma_dsyr: invalid 'lda'");
#endif

  dsyr(
      &C_U, &n, &alpha, x.memptr(), &I_one, A.memptr(), &lda);
}

const arma::mat& LU_fact::get_LU() const
{
  /* set LU factorization if needed */
  std::call_once(*is_comp, [&](){
    *LU = X;

    int info;
    lapack::dgetrf(
      &m, &n, LU->memptr(), &m, ipiv.get(), &info);

    if(info != 0L)
      throw std::runtime_error(
          "'dgetrf' failed with info: " + std::to_string(info));
  });

  return *LU;
}

inline void check_dgetrs_info(const int info){
  if(info != 0L)
    throw std::runtime_error(
        "'dgetrs' failed with info: " + std::to_string(info));
}

void LU_fact::solve(arma::mat &Z) const {
  get_LU();
#ifdef MSSM_DEBUG
  if((int)Z.n_rows != n)
    throw std::invalid_argument("'Z.n_rows' does not match with LU dim");
#endif
  int nrhs = Z.n_cols;

  int info;
  dgetrs(
    &C_N, &n, &nrhs, LU->memptr(), &m, ipiv.get(), Z.memptr(),
    &n, &info);

  check_dgetrs_info(info);
}

void LU_fact::solve(arma::vec &z) const {
  get_LU();
#ifdef MSSM_DEBUG
  if((int)z.n_elem != n)
    throw std::invalid_argument("'z.n_elem' does not match with LU dim");
#endif

  int info;
  dgetrs(
      &C_N, &n, &I_one, LU->memptr(), &m, ipiv.get(), z.memptr(),
      &n, &info);

  check_dgetrs_info(info);
}

template<bool add>
inline void sym_band_mat_set
  (double *mem, const int dim, const int ku, const arma::mat &x,
   const int istart, const int jstart, const double alpha){
  for(unsigned xj = 0; xj < x.n_cols; ++xj){
    const int j = xj + jstart;
    if(j >= dim)
      break;
    const int m = ku - j, col_skip = j * (ku + 1L);
    for(unsigned int xi = 0; xi < x.n_rows; ++xi){
      const int i = xi + istart;
      if(i < std::max(0L, (long)(j - ku)))
        continue;
      if(i > j)
        break;

      if(add)
        *(mem + m + i + col_skip) += alpha * x(xi, xj);
      else
        *(mem + m + i + col_skip)  = x(xi, xj);
    }
  }
}

void sym_band_mat::set_diag_block
  (const unsigned int number, const arma::mat &new_mat, const double alpha){
#ifdef MSSM_DEBUG
  if((int)number >= n_bands)
    throw std::invalid_argument("number out-of-bounds");
  if((int)new_mat.n_rows != dim_dia or (int)new_mat.n_cols != dim_dia)
    throw std::invalid_argument("incorrect dimension of new_mat");
#endif
  const int start = number * dim_dia;
  if(alpha == 0.)
    sym_band_mat_set<false>(mem.get(), dim, ku, new_mat, start, start, alpha);
  else
    sym_band_mat_set<true>(mem.get(), dim, ku, new_mat, start, start, alpha);
}

void sym_band_mat::set_upper_block
  (const unsigned int number, const arma::mat &new_mat){
#ifdef MSSM_DEBUG
  if((int)number >= n_bands - 1L)
    throw std::invalid_argument("number out-of-bounds");
  if((int)new_mat.n_rows != dim_dia or (int)new_mat.n_cols != dim_off)
    throw std::invalid_argument(
        "incorrect dimension of new_mat (" + std::to_string(new_mat.n_rows) +
          ", " + std::to_string(new_mat.n_cols) + ", " +
          std::to_string(dim_dia) + ", " + std::to_string(dim_off) + ")");
#endif
  const int i_start = number * dim_dia, j_start = (number + 1L) * dim_dia;
  sym_band_mat_set<false>(mem.get(), dim, ku, new_mat, i_start, j_start, 0.);
}

arma::vec sym_band_mat::mult(const double *x) const {
  arma::vec out(dim, arma::fill::zeros);
  dsbmv(
      &C_U, &dim, &ku, &D_one, mem.get(), &ku1, x, &I_one,
      &D_zero, out.memptr(), &I_one);

      return out;
}

arma::vec sym_band_mat::mult(const arma::vec &x) const{
#ifdef MSSM_DEBUG
  if((int)x.n_elem != dim)
    throw std::invalid_argument(
        "invalid dimension in 'sym_band_mat::mult' (" +
          std::to_string(x.n_elem) + ", " + std::to_string(dim) + ")");
#endif
  return mult(x.memptr());
}

std::unique_ptr<double[]> sym_band_mat::get_chol(int &info) const {
  /* copy matrix */
  std::unique_ptr<double[]> cp(new double[mem_size]);
  std::copy(mem.get(), mem.get() + mem_size, cp.get());

  /* compute cholesky decomposition */
  lapack::dpbtrf(&C_U, &dim, &ku, cp.get(), &ku1, &info);

  return cp;
}

/* TODO: replace by method that uses LU decomposition */
double sym_band_mat::ldeterminant(int &info) const {
  std::unique_ptr<double[]> cp = get_chol(info);

  if(info != 0L)
    return 0.;

  /* TODO: more stable way to do this? */
  double dia_sum = 0.;
  double *x = cp.get() + ku;
  for(int i = 0; i < dim; ++i, x += ku1)
    dia_sum += std::log(*x);

  return 2 * dia_sum;
}

double sym_band_mat::ldeterminant() const {
  int info;
  const double out = ldeterminant(info);

  if(info != 0)
    throw std::runtime_error(
        "'dpbtrf' failed with code " + std::to_string(info));

  return out;
}

arma::vec sym_band_mat::solve(const arma::vec &x, int &info) const {
#ifdef MSSM_DEBUG
  if((int)x.n_elem != dim)
    throw std::invalid_argument(
        "invalid dimension in 'sym_band_mat::solve' (" +
          std::to_string(x.n_elem) + ", " + std::to_string(dim) + ")");
#endif
  std::unique_ptr<double[]> cp = get_chol(info);
  arma::vec out = x;

  if(info != 0L){
    std::fill(
      out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
    return out;
  }

  dpbtrs(
    &C_U, &dim, &ku, &I_one, cp.get(), &ku1, out.memptr(), &dim, &info);

  return out;
}

arma::vec sym_band_mat::solve(const arma::vec &x) const {
  int info;
  const arma::vec out = sym_band_mat::solve(x, info);

  if(info != 0L)
    throw std::runtime_error("sym_band_mat::solve: got info " +
                             std::to_string(info));

  return out;
}

arma::mat sym_band_mat::get_dense() const  {
  arma::mat out(dim, dim, arma::fill::zeros);
  for(int j = 0; j < dim; ++j){
    const int m = ku - j, col_skip = j * (ku + 1L);
    for(int i = std::max(0, j - ku); i <= j; ++i)
      out(i, j) = *(mem.get() + m + i + col_skip);
  }

  out = arma::symmatu(out);

  return out;
}

// [[Rcpp::export(.get_Q0)]]
arma::mat get_Q0(const arma::mat &Qmat, const arma::mat &Fmat){
#ifdef MSSM_DEBUG
  if(arma::size(Qmat) != arma::size(Fmat))
    throw std::invalid_argument("'Qmat' and 'Fmat' sizes' do not match");
#endif

  arma::cx_vec eigval;
  arma::cx_mat eigvec;
  arma::eig_gen(eigval, eigvec, Fmat);

  if(std::any_of(eigval.begin(), eigval.end(),
                 [](const arma::cx_vec::value_type x){
                   return
                   std::sqrt(x.real() * x.real() + x.imag() * x.imag()) >= 1.;
                 }))
    throw std::runtime_error("Divergent series");

  arma::mat dum(arma::size(Qmat), arma::fill::zeros);
  arma::cx_mat T(Qmat, dum);
  T = arma::solve(eigvec, T);
  T = arma::solve(eigvec, T.t());
  arma::cx_mat Z = T / (1 - eigval * eigval.t());

  return arma::real(eigvec * Z * eigvec.t());
}

