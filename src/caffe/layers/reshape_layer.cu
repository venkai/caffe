#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>

#include "caffe/layers/reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReshapeKernel(const int nthreads, const int N,
    const int C, const int H, const int W, const int r, const int r2,
    const bool inv, const Dtype* const src_data, Dtype* const dst_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index;
    const int w = n % W;
    n /= W;
    const int h = n % H;
    n /= H;
    const int c = n % C;
    n /= C;

    if (!inv) {
      // src: N x C x H x W  [C must be divisible by r*r]
      // dst: N x (C / (r * r)) x (H * r) x (W * r)
      const int new_c = c / r2;
      const int new_h = (h * r) + ((c / r) % r);
      const int new_w = (w * r) + (c % r);
      const int new_index =
          (((((n * (C / r2)) + new_c) * H * r) + new_h) * W * r) + new_w;
      dst_data[new_index] = src_data[index];
    } else {
      // src: N x C x H x W  [H and W must both be divisible by r]
      // dst: N x (C * (r * r)) x (H / r) x (W / r)
      const int new_c = (c * r2) + ((h % r) * r) + (w % r);
      const int new_h = h / r;
      const int new_w = w / r;
      const int new_index =
          (((((n * C * r2) + new_c) * (H / r)) + new_h) * (W / r)) + new_w;
      dst_data[new_index] = src_data[index];
    }
  }
}

template <typename Ftype, typename Btype>
void ReshapeLayer<Ftype, Btype>::dense_reshape_gpu(const vector<Blob*>& src,
    const vector<Blob*>& dst, const bool inv, const bool reshape_diffs) {
  const int N = src[0]->shape(0);
  const int C = src[0]->shape(1);
  const int H = src[0]->shape(2);
  const int W = src[0]->shape(3);
  const int nthreads = src[0]->count();
  if (!reshape_diffs) {
    const Ftype* const src_data = src[0]->gpu_data<Ftype>();
    Ftype* const dst_data = dst[0]->mutable_gpu_data<Ftype>();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReshapeKernel<Ftype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
        0, Caffe::thread_stream()>>>(
        nthreads, N, C, H, W, r_, r2_, inv, src_data, dst_data);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  } else {
    const Btype* const src_diff = src[0]->gpu_diff<Btype>();
    Btype* const dst_diff = dst[0]->mutable_gpu_diff<Btype>();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReshapeKernel<Btype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,
        0, Caffe::thread_stream()>>>(
        nthreads, N, C, H, W, r_, r2_, inv, src_diff, dst_diff);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(ReshapeLayer);

}  // namespace caffe