#include <algorithm>
#include <vector>

#include "caffe/layers/conv_rec_layer.hpp"

namespace caffe {

// Projects the normalized/regularized Gradient computed by previous solver
// iteration onto the tangent space in the Stiefel Manifold of the orthogonal
// weights from the previous iteration. Subsequently a descent curve based on
// Cayley's transform is used to determine the new weights. Such a procedure
// ensures that weights remain orthogonal while simultaneously optimizing the
// problem at hand. Note that the solver gradient can correspond to any of the
// pre-existing solver configurations like SGD, RMSProp, Adam, Momentum, etc.
template <typename Dtype>
void RecursiveConvLayer<Dtype>::orth_weight_update_gpu() {
  // Called by Forward_gpu whenever any Backward pass
  // is executed in the previous iteration.
  for (int i = 0; i < Nwts_; ++i) {
    // Recover previous iter weights which solver update clobbered W <- W + G
    caffe_gpu_axpy<Dtype>(this->blobs_[i]->count(), Dtype(1),
        this->blobs_[i]->gpu_diff(), this->blobs_[i]->mutable_gpu_data());
    // wt_buffer_ = transpose(G) * W
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, C_, C_, C_, (Dtype)1.,
        this->blobs_[i]->gpu_diff(), this->blobs_[i]->gpu_data(),
        (Dtype)0., wt_buffer_.mutable_gpu_data());
    // A = wt_buffer_ - transpose(wt_buffer_) = G'*W - W'*G
    caffe_gpu_absymm<Dtype>(C_, Dtype(1), Dtype(-1), wt_buffer_.gpu_data(),
        A_.mutable_gpu_data());
    // G <- A * W
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, C_, C_, C_, (Dtype)1.,
        A_.gpu_data(), this->blobs_[i]->gpu_data(), (Dtype)0.,
        this->blobs_[i]->mutable_gpu_diff());
    // W <- W - G = (I - A)*W
    caffe_gpu_axpy<Dtype>(this->blobs_[i]->count(), Dtype(-1),
        this->blobs_[i]->gpu_diff(), this->blobs_[i]->mutable_gpu_data());
    // A <- I + A
    caffe_gpu_axpy<Dtype>(A_.count(), Dtype(1), eye_.gpu_data(),
        A_.mutable_gpu_data());
    // Orthogonal weight update: W <- inv(A)*W; i.e. Wnew = inv(I+A)*(I-A)*Wold
    // where A = G'*W - W'*G; and G is the original diff used by the solver.
    caffe_gpu_inverse_qr<Dtype>(CblasLeft, CblasNoTrans, C_, C_, Dtype(1.0),
        A_.mutable_gpu_data(), tau_.mutable_gpu_data(),
        this->blobs_[i]->mutable_gpu_data(), Lwork_,
        workspace_.mutable_gpu_data(), dev_info_.mutable_gpu_data());
  }
}

template <typename Dtype>
__global__ void PermuteKernel(const int nthreads, const int num_axes,
    const int* const old_steps, const int* const new_steps,
    const int* const new_orders, const Dtype* const bottom_data,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int old_idx = 0, idx = index;
    for (int j = 0; j < num_axes; ++j) {
      old_idx += (idx / new_steps[j]) * old_steps[new_orders[j]];
      idx %= new_steps[j];
    }
    top_data[index] = bottom_data[old_idx];
  }
}

/* Modified from the PermutationLayer implementation in
  https://github.com/BVLC/caffe/commit/b68695db42aa79e874296071927536363fe1efbf
  by Wei Liu : https://github.com/weiliu89 */
template <typename Dtype>
void RecursiveConvLayer<Dtype>::permute_blobs_gpu(
    const vector<Blob<Dtype>*>& bottom, const bool channel_last,
    const bool permute_diffs) {
  // Called by both Forward_gpu and Backward_gpu.
  // Permute input blob (data or diff) from N*C*H*W to N*H*W*C or vice-versa.
  // The permuted blob is stored in the buffer mid_.
  if (channel_last) {
    mid_.Reshape(new_mid_shape_);
  } else {
    mid_.Reshape(old_mid_shape_);
  }
  // Start permuting bottom blob data
  const Dtype* const bottom_data = bottom[0]->gpu_data();
  Dtype* const top_data = mid_.mutable_gpu_data();
  if (channel_last) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
        count_, num_axes_, old_steps_.gpu_data(), new_steps_.gpu_data(),
        permute_order_.gpu_data(), bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
        count_, num_axes_, new_steps_.gpu_data(), old_steps_.gpu_data(),
        inv_permute_order_.gpu_data(), bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  }
  if (!permute_diffs) { return; }
  // Start permuting bottom blob diffs
  const Dtype* const bottom_diff = bottom[0]->gpu_diff();
  Dtype* const top_diff = mid_.mutable_gpu_diff();
  if (channel_last) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
        count_, num_axes_, old_steps_.gpu_data(), new_steps_.gpu_data(),
        permute_order_.gpu_data(), bottom_diff, top_diff);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
        count_, num_axes_, new_steps_.gpu_data(), old_steps_.gpu_data(),
        inv_permute_order_.gpu_data(), bottom_diff, top_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

// ----------------------------------------------------------------------------
// ---------------------- Helpers for FORWARD PASS ----------------------------
// ----------------------------------------------------------------------------

template <typename Dtype>
__global__ void ReLUForward(const int nthreads, const Dtype negative_slope,
    const Dtype* const bottom_data, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = bottom_data[index] > 0 ?
        bottom_data[index] : bottom_data[index] * negative_slope;
  }
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::forward_ReLU_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) const {
  const Dtype* const bottom_data = bottom[0]->gpu_data();
  Dtype* const top_data = top[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
      count_, negative_slope_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::forward_BN_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    const int iter) {
  const int offset = iter * C_;
  const Dtype* const bottom_data = bottom[0]->gpu_data();
  Dtype* const top_data = top[0]->mutable_gpu_data();
  if (bottom[0] != top[0]) {
    caffe_copy(count_, bottom_data, top_data);
  }
  if (!use_global_stats_) {
    // Compute Batch-Mean E(X)
    caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, inv_batch_size_,
        bottom_data, batch_sum_multiplier_.gpu_data(), (Dtype)0.,
        bn_mu_.mutable_gpu_data() + offset);
  }
  // Subtract Batch-Mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)-1., batch_sum_multiplier_.gpu_data(), bn_mu_.gpu_data() + offset,
      (Dtype)1., top_data);
  if (!use_global_stats_) {
    // Compute Batch-Variance E((X-EX)^2)
    caffe_gpu_mul(count_, top_data, top_data, mid_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, inv_batch_size_,
        mid_.gpu_data(), batch_sum_multiplier_.gpu_data(), (Dtype)0.,
        bn_sigma_.mutable_gpu_data() + offset);
    // Compute and save moving average
    caffe_gpu_axpby(C_, Dtype(1), bn_mu_.gpu_data() + offset,
        moving_average_fraction_,
        this->blobs_[bn_param_offset_]->mutable_gpu_data() + offset);
    caffe_gpu_axpby(C_, bias_correction_factor_, bn_sigma_.gpu_data() + offset,
        moving_average_fraction_,
        this->blobs_[bn_param_offset_ + 1]->mutable_gpu_data() + offset);
    // Compute Batch-St-dev = sqrt(Batch-Variance + epsilon)
    caffe_gpu_add_scalar(C_, eps_, bn_sigma_.mutable_gpu_data() + offset);
    caffe_gpu_powx(C_, bn_sigma_.gpu_data() + offset , Dtype(0.5),
        bn_sigma_.mutable_gpu_data() + offset);
  }
  // Replicate Batch-St-dev to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.gpu_data(),
      bn_sigma_.gpu_data() + offset, (Dtype)0., mid_.mutable_gpu_data());
  // Divide by Batch-St-dev
  caffe_gpu_div(count_, top_data, mid_.gpu_data(), top_data);
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
  if (requires_orth_weight_update_) {
    orth_weight_update_gpu();
  }
  const bool channel_last = true;
  const bool permute_diffs = true;
  // Permute bottom from N*C*H*W to N*H*W*C and copy to mid_
  permute_blobs_gpu(bottom, channel_last, !permute_diffs);
  top[0]->ReshapeLike(mid_);
  if (!use_global_stats_) {
    caffe_gpu_scal<Dtype>(this->blobs_[bn_param_offset_ + 2]->count(),
        moving_average_fraction_,
        this->blobs_[bn_param_offset_ + 2]->mutable_gpu_data());
    caffe_gpu_add_scalar(this->blobs_[bn_param_offset_ + 2]->count(), Dtype(1),
        this->blobs_[bn_param_offset_ + 2]->mutable_gpu_data());
  }
  for (int iter = 0; iter < Nrec_; ++iter) {
    // Standard 1x1 convolution
    const int wt_offset = rand_wt_order_[iter];
    const Dtype* const weights = this->blobs_[wt_offset]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, C_,
        (Dtype)1., mid_.gpu_data(), weights, (Dtype)0.,
        top[0]->mutable_gpu_data());
    // Compute activation function in-place
    forward_activation_func_gpu(top, top);  // a_{i+1} = \sigma(a_{i+1});
    // Apply BN in-place
    forward_BN_gpu(top, top, iter);
    if (iter == Nrec_ - 1) {
      // Permute top from N*H*W*C to N*C*H*W and copy to mid_
      permute_blobs_gpu(top, !channel_last, !permute_diffs);
      top[0]->ReshapeLike(mid_);
      caffe_copy(count_, mid_.gpu_data(), top[0]->mutable_gpu_data());
    } else {
      // mid_ <- top; //a_i <- a_{i+1};
      caffe_copy(count_, top[0]->gpu_data(), mid_.mutable_gpu_data());
    }
  }
}

// ----------------------------------------------------------------------------
// --------------------- Helpers for BACKWARD PASS ----------------------------
// ----------------------------------------------------------------------------

template <typename Dtype>
__global__ void ReLUBackward(const int nthreads, const Dtype negative_slope,
    const Dtype inv_negative_slope, const Dtype* const top_data,
    const Dtype* const top_diff, Dtype* const bottom_data,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = top_data[index] > 0 ?
        top_diff[index] : top_diff[index] * negative_slope;
    bottom_data[index] = top_data[index] > 0 ?
        top_data[index] : top_data[index] * inv_negative_slope;
  }
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::backward_ReLU_gpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) const {
  const Dtype* const top_data = top[0]->gpu_data();
  Dtype* const bottom_data = bottom[0]->mutable_gpu_data();
  const Dtype* const top_diff = top[0]->gpu_diff();
  Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
    count_, negative_slope_, inv_negative_slope_, top_data, top_diff,
    bottom_data, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::backward_BN_gpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom,
    const int iter) {
  const int offset = iter * C_;
  if (bottom[0] == top[0]) {
    caffe_copy(count_, top[0]->gpu_diff(), mid_.mutable_gpu_diff());
  }
  Dtype* const bottom_data = bottom[0]->mutable_gpu_data();
  const Dtype* const top_data = top[0]->gpu_data();
  Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* const top_diff =
      bottom[0] == top[0] ? mid_.gpu_diff() : top[0]->gpu_diff();

  // Replicate Batch-St-dev to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.gpu_data(),
      bn_sigma_.gpu_data() + offset, (Dtype)0., mid_.mutable_gpu_data());
  if (use_global_stats_) {
    caffe_gpu_div(count_, top[0]->gpu_diff(), mid_.gpu_data(), bottom_diff);
    // Invert BN --> Multiply by Batch-St-Dev
    caffe_gpu_mul(count_, top_data, mid_.gpu_data(), bottom_data);
    // Invert BN --> Add Batch-Mean
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
        (Dtype)1., batch_sum_multiplier_.gpu_data(),
        bn_mu_.gpu_data() + offset, (Dtype)1., bottom_data);
    return;
  }
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(count_, top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, (Dtype)1., bottom_diff,
      batch_sum_multiplier_.gpu_data(), (Dtype)0.,
      temp_bn_sum_.mutable_gpu_data());
  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.gpu_data(), temp_bn_sum_.gpu_data(),
      (Dtype)0., bottom_diff);
  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(count_, top_data, bottom_diff, bottom_diff);
  // sum(dE/dY)
  caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, (Dtype)1., top_diff,
      batch_sum_multiplier_.gpu_data(), (Dtype)0.,
      temp_bn_sum_.mutable_gpu_data());
  // reshape (broadcast) the above: sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.gpu_data(), temp_bn_sum_.gpu_data(),
      (Dtype)1., bottom_diff);
  // dE/dY - mean(dE/dY)- (mean(dE/dY \cdot Y) \cdot Y)
  caffe_gpu_axpby(count_, Dtype(1), top_diff, Dtype(-1. * inv_batch_size_),
      bottom_diff);
  // note: mid_.gpu_data() contains sqrt(var(X)+eps)
  caffe_gpu_div(count_, bottom_diff, mid_.gpu_data(), bottom_diff);
  // Invert BN --> Multiply by Batch-St-Dev
  caffe_gpu_mul(count_, top_data, mid_.gpu_data(), bottom_data);
  // Invert BN --> Add Batch-Mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.gpu_data(), bn_mu_.gpu_data() + offset,
      (Dtype)1., bottom_data);
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!this->param_propagate_down_[0] && !propagate_down[0]) {
    return;
  }
  const bool channel_last = true;
  const bool permute_diffs = true;
  // Permute top data & diffs from N*C*H*W to N*H*W*C and copy to mid_
  permute_blobs_gpu(top, channel_last, permute_diffs);
  bottom[0]->ReshapeLike(mid_);
  caffe_copy(count_, mid_.gpu_data(), bottom[0]->mutable_gpu_data());
  caffe_copy(count_, mid_.gpu_diff(), bottom[0]->mutable_gpu_diff());
  // TOP Data & Diff are now in BOTTOM, permuted in order (N*H*W) x C
  for (int iter = Nrec_ - 1; iter >= 0; --iter) {
    backward_BN_gpu(bottom, bottom, iter);
    backward_activation_func_gpu(bottom, bottom);
    // Invert data (bottom[0])*inv(W)->data(mid_),
    // compute diff(W) and backprop diff(bottom[0])->diff(mid_).
    const int wt_offset = rand_wt_order_[iter];
    const Dtype* const weights = this->blobs_[wt_offset]->gpu_data();
    Dtype* const weights_diff = this->blobs_[wt_offset]->mutable_gpu_diff();
    // First get BOTTOM data using the inverse of weights
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, C_, C_,
        (Dtype)1., bottom[0]->gpu_data(), weights, (Dtype)0.,
        mid_.mutable_gpu_data());
    // Note: BOTTOM Data is now in mid_, TOP Data & Diff still in bottom[0]
    // Compute diff with respect to weights if needed
    if (this->param_propagate_down_[0]) {
      // Standard SGD diff for W: pdv{loss}{W} = G
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, C_, C_, batch_size_,
          (Dtype)1., mid_.gpu_data(), bottom[0]->gpu_diff(), (Dtype)1.,
          weights_diff);
    }
    // Compute diff with respect to bottom activation.
    // We must always do this, even if propagate_down[0] is false.
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, C_, C_,
        (Dtype)1., bottom[0]->gpu_diff(), weights, (Dtype)0.,
        mid_.mutable_gpu_diff());
    // Note: BOTTOM Diff is now in mid_, TOP Data & Diff are still in bottom[0]
    // Transfer Data & Diff from mid_ to bottom[0]
    caffe_copy(count_, mid_.gpu_data(), bottom[0]->mutable_gpu_data());
    caffe_copy(count_, mid_.gpu_diff(), bottom[0]->mutable_gpu_diff());
  }
  // Permute bottom data & diffs from N*H*W*C to N*C*H*W and copy to mid_
  permute_blobs_gpu(bottom, !channel_last, permute_diffs);
  bottom[0]->ReshapeLike(mid_);
  caffe_copy(count_, mid_.gpu_data(), bottom[0]->mutable_gpu_data());
  caffe_copy(count_, mid_.gpu_diff(), bottom[0]->mutable_gpu_diff());

  // The next forward pass will project the solver's regularized weight diffs
  // on to the Tangent Space in the Stiefel manifold of the weights, and
  // recompute the new weights using Cayley's transform. This will ensure that
  // the weights always remain orthogonal in a natural way while simultaneously
  // optimizing the problem at hand.
  requires_orth_weight_update_ = true;
}

INSTANTIATE_LAYER_GPU_FUNCS(RecursiveConvLayer);

}  // namespace caffe
