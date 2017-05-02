#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// ----------------------------------------------------------------------------
// ------------------- Begin caffe wrappers for cusolverDn --------------------
// ----------------------------------------------------------------------------

template <>
void caffe_gpu_inverse_qr<float>(const CBLAS_SIDE SideA,
    const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha,
    float* const A, float* const TAU, float* const B, const int Lwork,
    float* const Workspace, int* const devInfo) {
  // Note: cusolverDn uses fortran-order
  const int ldb = N;
  int lda = (SideA == CblasLeft) ? M : N;
  cublasSideMode_t cuSideA =
      (SideA == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasOperation_t cuTransR =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransQ =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  // First generate QR factorization of A
  CUSOLVER_CHECK(cusolverDnSgeqrf(Caffe::cusolver_dn_handle(), lda, lda, A,
      lda, TAU, Workspace, Lwork, devInfo));
  if ((SideA == CblasLeft && TransA == CblasNoTrans) ||
      (SideA == CblasRight && TransA == CblasTrans)) {
    // First do trsm and then ormqr
    // Pre/Post-Multiply B by inverse of R or R'
    CUBLAS_CHECK(cublasStrsm(Caffe::cublas_handle(), cuSideA,
        CUBLAS_FILL_MODE_UPPER, cuTransR, CUBLAS_DIAG_NON_UNIT, N, M, &alpha,
        A, lda, B, ldb));
    // Post/Pre-Multiply B by inverse of Q' or Q without generating Q
    CUSOLVER_CHECK(cusolverDnSormqr(Caffe::cusolver_dn_handle(), cuSideA,
      cuTransQ, N, M, lda, A, lda, TAU, B, ldb, Workspace, Lwork, devInfo));
  } else {
    // First do ormqr and then trsm
    // Pre/Post-Multiply B by inverse of Q or Q' without generating Q
    CUSOLVER_CHECK(cusolverDnSormqr(Caffe::cusolver_dn_handle(), cuSideA,
      cuTransQ, N, M, lda, A, lda, TAU, B, ldb, Workspace, Lwork, devInfo));
    // Post/Pre-Multiply B by inverse of R' or R
    CUBLAS_CHECK(cublasStrsm(Caffe::cublas_handle(), cuSideA,
        CUBLAS_FILL_MODE_UPPER, cuTransR, CUBLAS_DIAG_NON_UNIT, N, M, &alpha,
        A, lda, B, ldb));
  }
}

template <>
void caffe_gpu_inverse_qr<double>(const CBLAS_SIDE SideA,
    const CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha,
    double* const A, double* const TAU, double* const B, const int Lwork,
    double* const Workspace, int* const devInfo) {
  // Note: cusolverDn uses fortran-order
  const int ldb = N;
  int lda = (SideA == CblasLeft) ? M : N;
  cublasSideMode_t cuSideA =
      (SideA == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasOperation_t cuTransR =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransQ =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  // First generate QR factorization of A
  CUSOLVER_CHECK(cusolverDnDgeqrf(Caffe::cusolver_dn_handle(), lda, lda, A,
      lda, TAU, Workspace, Lwork, devInfo));
  if ((SideA == CblasLeft && TransA == CblasNoTrans) ||
      (SideA == CblasRight && TransA == CblasTrans)) {
    // First do trsm and then ormqr
    // Pre/Post-Multiply B by inverse of R or R'
    CUBLAS_CHECK(cublasDtrsm(Caffe::cublas_handle(), cuSideA,
        CUBLAS_FILL_MODE_UPPER, cuTransR, CUBLAS_DIAG_NON_UNIT, N, M, &alpha,
        A, lda, B, ldb));
    // Post/Pre-Multiply B by inverse of Q' or Q without generating Q
    CUSOLVER_CHECK(cusolverDnDormqr(Caffe::cusolver_dn_handle(), cuSideA,
      cuTransQ, N, M, lda, A, lda, TAU, B, ldb, Workspace, Lwork, devInfo));
  } else {
    // First do ormqr and then trsm
    // Pre/Post-Multiply B by inverse of Q or Q' without generating Q
    CUSOLVER_CHECK(cusolverDnDormqr(Caffe::cusolver_dn_handle(), cuSideA,
      cuTransQ, N, M, lda, A, lda, TAU, B, ldb, Workspace, Lwork, devInfo));
    // Post/Pre-Multiply B by inverse of R' or R
    CUBLAS_CHECK(cublasDtrsm(Caffe::cublas_handle(), cuSideA,
        CUBLAS_FILL_MODE_UPPER, cuTransR, CUBLAS_DIAG_NON_UNIT, N, M, &alpha,
        A, lda, B, ldb));
  }
}

template <>
void caffe_gpu_orthogonalize<float>(const int M, const int N,
    float* const A, float* const TAU, const int Lwork,
    float* const Workspace, int* const devInfo) {
  // A is M * N (i.e. M rows, N columns) in row major with N >= M.
  const int lda = N;
  // First generate QR factorization of transpose(A)
  CUSOLVER_CHECK(cusolverDnSgeqrf(Caffe::cusolver_dn_handle(), N, M, A,
      lda, TAU, Workspace, Lwork, devInfo));
  // Generate orthogonal Q matrix and overwrite its transpose on A.
  CUSOLVER_CHECK(cusolverDnSorgqr(Caffe::cusolver_dn_handle(), N, M, M, A,
      lda, TAU, Workspace, Lwork, devInfo));
}

template <>
void caffe_gpu_orthogonalize<double>(const int M, const int N,
    double* const A, double* const TAU, const int Lwork,
    double* const Workspace, int* const devInfo) {
  // A is M * N (i.e. M rows, N columns) in row major with N >= M.
  const int lda = N;
  // First generate QR factorization of transpose(A)
  CUSOLVER_CHECK(cusolverDnDgeqrf(Caffe::cusolver_dn_handle(), N, M, A,
      lda, TAU, Workspace, Lwork, devInfo));
  // Generate orthogonal Q matrix and overwrite its transpose on A.
  CUSOLVER_CHECK(cusolverDnDorgqr(Caffe::cusolver_dn_handle(), N, M, M, A,
      lda, TAU, Workspace, Lwork, devInfo));
}

template <>
void caffe_gpu_buffersize_qr<float>(const int M, const int N,
    float* const A, float* const TAU, int* const Lwork) {
  const int lda = M;
  int lwork_geqrf = 0, lwork_orgqr = 0;
  CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(Caffe::cusolver_dn_handle(),
      M, N, A, lda, &lwork_geqrf));
  CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(Caffe::cusolver_dn_handle(),
      M, N, N, A, lda, TAU, &lwork_orgqr));
  *Lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
}

template <>
void caffe_gpu_buffersize_qr<double>(const int M, const int N,
    double* const A, double* const TAU, int* const Lwork) {
  const int lda = M;
  int lwork_geqrf = 0, lwork_orgqr = 0;
  CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(Caffe::cusolver_dn_handle(),
      M, N, A, lda, &lwork_geqrf));
  CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(Caffe::cusolver_dn_handle(),
      M, N, N, A, lda, TAU, &lwork_orgqr));
  *Lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
}

// ----------------------------------------------------------------------------
// -------------------- End caffe wrappers for cusolverDn ---------------------
// ----------------------------------------------------------------------------

template <typename Dtype>
__global__ void absymm_kernel(const int n, const Dtype alpha, const Dtype beta,
    const Dtype* const x, Dtype* const y) {
  CUDA_KERNEL_LOOP(index, n * n) {
    y[index] = (alpha * x[index]) + (beta * x[((index % n)* n) + (index / n)]);
  }
}

template <>
void caffe_gpu_absymm<float>(const int N, const float alpha, const float beta,
    const float* const A, float* const B) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  absymm_kernel<float><<<CAFFE_GET_BLOCKS(N * N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, beta, A, B);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_absymm<double>(const int N, const double alpha,
    const double beta, const double* const A, double* const B) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  absymm_kernel<double><<<CAFFE_GET_BLOCKS(N * N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, beta, A, B);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
