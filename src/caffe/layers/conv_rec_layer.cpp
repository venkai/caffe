#include <vector>
#include <algorithm>
#include <memory>
#include "caffe/filler.hpp"
#include "caffe/layers/conv_rec_layer.hpp"


namespace caffe {

template <typename Dtype>
void RecursiveConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
  RecursiveConvParameter rec_conv_param =
      this->layer_param_.recursive_conv_param();
  // Set up Convolution HyperParameters
  Nrec_ = rec_conv_param.num_recursive_layers();
  if (rec_conv_param.has_num_unique_weights()) {
    Nwts_ = rec_conv_param.num_unique_weights();
    CHECK_GE(Nrec_, Nwts_)
        << "# of recursive layers (Nrec_) >= # of unique weights (Nwts_)";
  } else {
    Nwts_ = Nrec_;
  }
  rand_wt_order_.clear();
  for (int i = 0; i < Nrec_; ++i) {
    rand_wt_order_.push_back(i % Nwts_);
  }
  // Set up Batch-Norm HyperParameters
  use_global_stats_ = this->phase_ == TEST;
  moving_average_fraction_ = 0.999;
  eps_ = 1e-5;
  if (rec_conv_param.has_batch_norm_param()) {
    BatchNormParameter bn_param = rec_conv_param.batch_norm_param();
    moving_average_fraction_ = bn_param.moving_average_fraction();
    if (bn_param.has_use_global_stats()) {
      use_global_stats_ = bn_param.use_global_stats();
    }
    eps_ = bn_param.eps();
  }
  // Set up Activation Function HyperParameters
  // HyperParameters for ReLU
  negative_slope_ = 0.99;
  if (rec_conv_param.has_relu_param()) {
    negative_slope_ = rec_conv_param.relu_param().negative_slope();
    CHECK_GT(negative_slope_, eps_)
        << "Negative Slope must be strictly positive for bijectivity!";
  }
  inv_negative_slope_ = ((Dtype)1. / negative_slope_);
  // Handle the parameter blobs: CONV + BN.
  // - blobs_[0 ... (Nwts_ - 1)] holds the filter weights
  // - blobs_[Nwts_] holds global BN means
  // - blobs_[Nwts_ + 1] holds global BN variances
  // - blobs_[Nwts_ + 2] holds BN bias correction factor
  C_ = bottom[0]->shape(1);  // # of channels (can't change after LayerSetUp)
  bn_param_offset_ = Nwts_;
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(bn_param_offset_ + 3);
    // Initialize and fill weights
    for (int i = 0; i < Nwts_; ++i) {
      const vector<int> weight_shape{C_, C_};
      this->blobs_[i].reset(new Blob<Dtype>(weight_shape));
      shared_ptr<Filler<Dtype> >
          weight_filler(GetFiller<Dtype>(rec_conv_param.weight_filler()));
      weight_filler->Fill(this->blobs_[i].get());
    }
    // Initialize and fill BN mean, variance, bias correction
    const vector<int> bn_mean_shape{Nrec_, C_};
    this->blobs_[bn_param_offset_].reset(new Blob<Dtype>(bn_mean_shape));
    this->blobs_[bn_param_offset_ + 1].reset(new Blob<Dtype>(bn_mean_shape));
    this->blobs_[bn_param_offset_ + 2].reset(
        new Blob<Dtype>(vector<int>(1, 1)));
    // In the case of GPU mode, directly allocate to GPU memory
    switch (Caffe::mode()) {
    case Caffe::CPU:
      // Zero-Mean
      caffe_set(this->blobs_[bn_param_offset_]->count(), Dtype(0),
          this->blobs_[bn_param_offset_]->mutable_cpu_data());
      // Unit-Variance
      caffe_set(this->blobs_[bn_param_offset_ + 1]->count(), Dtype(1),
          this->blobs_[bn_param_offset_ + 1]->mutable_cpu_data());
      // No Scaling
      caffe_set(this->blobs_[bn_param_offset_ + 2]->count(), Dtype(1),
          this->blobs_[bn_param_offset_ + 2]->mutable_cpu_data());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for (int i = 0; i < Nwts_; ++i) {
        this->blobs_[i]->mutable_gpu_data();  // CPU -> GPU
      }
      // Zero-Mean
      caffe_gpu_set(this->blobs_[bn_param_offset_]->count(), Dtype(0),
          this->blobs_[bn_param_offset_]->mutable_gpu_data());
      // Unit-Variance
      caffe_gpu_set(this->blobs_[bn_param_offset_ + 1]->count(), Dtype(1),
          this->blobs_[bn_param_offset_ + 1]->mutable_gpu_data());
      // No Scaling
      caffe_gpu_set(this->blobs_[bn_param_offset_ + 2]->count(), Dtype(1),
          this->blobs_[bn_param_offset_ + 2]->mutable_gpu_data());
#endif
      break;
    }
  }
  // Decide which params are learnable:
  // The only learnable params are the convolution weights & possibly
  // activation params. Mask BN statistics from optimization by setting local
  // learning rates for mean, variance, and the bias correction to zero.
  float lr_mult_wts = 0.f, decay_mult_wts = 1.f;
  CHECK_LE(this->layer_param_.param_size(), 1);
  if (this->layer_param_.param_size() == 1) {
    lr_mult_wts = this->layer_param_.param(0).lr_mult();
    decay_mult_wts = this->layer_param_.param(0).decay_mult();
  } else {
    ParamSpec* fixed_param_spec = this->layer_param_.add_param();
    fixed_param_spec->set_lr_mult(0.f);
  }
  for (int i = 1; i < Nwts_; ++i) {
    ParamSpec* fixed_param_spec = this->layer_param_.add_param();
    fixed_param_spec->set_lr_mult(lr_mult_wts);
    fixed_param_spec->set_decay_mult(decay_mult_wts);
  }
  for (int i = bn_param_offset_; i <= bn_param_offset_ + 2 ; ++i) {
    ParamSpec* fixed_param_spec = this->layer_param_.add_param();
    fixed_param_spec->set_lr_mult(0.f);
  }
  // Propagate gradients to CONV weights only if LR > 0.
  this->param_propagate_down_.resize(this->blobs_.size(), false);
  if (this->layer_param_.param(0).lr_mult() > 0.f) {
    for (int i = 0; i < Nwts_; ++i) {
      this->set_param_propagate_down(i, true);
    }
  }
  // One-time set up for buffers which do not depend on N_, H_ or W_.
  const vector<int> one_wt_shape{C_, C_};
  eye_.Reshape(one_wt_shape);
  wt_buffer_.Reshape(one_wt_shape);
  A_.Reshape(one_wt_shape);
  caffe_set(eye_.count(), Dtype(0), eye_.mutable_cpu_data());
  for (int i = 0; i < C_; ++i) {
    eye_.mutable_cpu_data()[i * (C_ + 1)] = Dtype(1);
  }
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_set(wt_buffer_.count(), Dtype(0), wt_buffer_.mutable_cpu_data());
    caffe_set(A_.count(), Dtype(0), A_.mutable_cpu_data());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    eye_.mutable_gpu_data();  // CPU -> GPU
    caffe_gpu_set(wt_buffer_.count(), Dtype(0), wt_buffer_.mutable_gpu_data());
    caffe_gpu_set(A_.count(), Dtype(0), A_.mutable_gpu_data());
    // Query workspace size for QR factorization
    caffe_gpu_inverse_qr<Dtype>(C_, C_, this->blobs_[0]->mutable_gpu_data(),
        &Lwork_);
    LOG(INFO) << "Size of cusolverDn workspace for QR: " << Lwork_;
    workspace_.Reshape(vector<int>(1, Lwork_));
    caffe_gpu_set(Lwork_, Dtype(0), workspace_.mutable_gpu_data());
    tau_.Reshape(vector<int>(1, C_));
    caffe_gpu_set(tau_.count(), Dtype(0), tau_.mutable_gpu_data());
    dev_info_.Reshape(vector<int>(1, 1));
    caffe_gpu_set(dev_info_.count(), 1, dev_info_.mutable_gpu_data());
#endif
    break;
  }
  bn_mu_.ReshapeLike(*(this->blobs_[bn_param_offset_]));
  bn_sigma_.ReshapeLike(*(this->blobs_[bn_param_offset_ + 1]));
  if (use_global_stats_) {
    const Dtype bn_scale_factor =
        (this->blobs_[bn_param_offset_ + 2]->cpu_data()[0] == 0) ? Dtype(0) :
        (Dtype(1.)/ this->blobs_[bn_param_offset_ + 2]->cpu_data()[0]);
    // Share Data to save memory
    bn_mu_.ShareData(*(this->blobs_[bn_param_offset_]));
    bn_sigma_.ShareData(*(this->blobs_[bn_param_offset_ + 1]));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      // use the stored mean/variance estimates.
      caffe_cpu_scale(bn_mu_.count(), bn_scale_factor,
          this->blobs_[bn_param_offset_]->cpu_data(),
          bn_mu_.mutable_cpu_data());
      caffe_cpu_scale(bn_sigma_.count(), bn_scale_factor,
          this->blobs_[bn_param_offset_ + 1]->cpu_data(),
          bn_sigma_.mutable_cpu_data());
      // compute standard deviation over batch = sqrt(variance + epsilon)
      caffe_add_scalar(bn_sigma_.count(), eps_, bn_sigma_.mutable_cpu_data());
      caffe_powx(bn_sigma_.count(), bn_sigma_.cpu_data(), Dtype(0.5),
          bn_sigma_.mutable_cpu_data());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      // use the stored mean/variance estimates.
      caffe_gpu_scale(bn_mu_.count(), bn_scale_factor,
          this->blobs_[bn_param_offset_]->gpu_data(),
          bn_mu_.mutable_gpu_data());
      caffe_gpu_scale(bn_sigma_.count(), bn_scale_factor,
          this->blobs_[bn_param_offset_ + 1]->gpu_data(),
          bn_sigma_.mutable_gpu_data());
      // compute standard deviation over batch = sqrt(variance + epsilon)
      caffe_gpu_add_scalar(bn_sigma_.count(), eps_,
          bn_sigma_.mutable_gpu_data());
      caffe_gpu_powx(bn_sigma_.count(), bn_sigma_.gpu_data(), Dtype(0.5),
          bn_sigma_.mutable_gpu_data());
#endif
      break;
    }
  } else {
    temp_bn_sum_.Reshape(vector<int>(1, C_));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(bn_mu_.count(), Dtype(0), bn_mu_.mutable_cpu_data());
      caffe_set(bn_sigma_.count(), Dtype(1), bn_sigma_.mutable_cpu_data());
      caffe_set(C_, Dtype(0), temp_bn_sum_.mutable_cpu_data());
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(bn_mu_.count(), Dtype(0), bn_mu_.mutable_gpu_data());
      caffe_gpu_set(bn_sigma_.count(), Dtype(1), bn_sigma_.mutable_gpu_data());
      caffe_gpu_set(C_, Dtype(0), temp_bn_sum_.mutable_gpu_data());
#endif
      break;
    }
  }
  const vector<int> permute_order_shape(1, num_axes_);
  permute_order_.Reshape(permute_order_shape);
  inv_permute_order_.Reshape(permute_order_shape);
  old_steps_.Reshape(permute_order_shape);
  new_steps_.Reshape(permute_order_shape);
  // (N,C,H,W) -> (N,H,W,C)
  constexpr std::array<int, num_axes_> order_C_last = {0, 2, 3, 1};
  // (N,H,W,C) -> (N,C,H,W)
  constexpr std::array<int, num_axes_> inv_order_C_last = {0, 3, 1, 2};
  for (int i = 0; i < num_axes_; ++i) {
    permute_order_.mutable_cpu_data()[i] = order_C_last[i];
    inv_permute_order_.mutable_cpu_data()[i] = inv_order_C_last[i];
    old_steps_.mutable_cpu_data()[i] = 1;
    new_steps_.mutable_cpu_data()[i] = 1;
  }
  // We will make this true whenever any Backward pass is executed.
  requires_orth_weight_update_ = false;
  // To activate RecursiveConvLayer<Dtype>::Reshape the first time
  // Note that N_, H_, W_ can change but C_ can't change after LayerSetUp.
  N_ = 0; H_ = 0; W_ = 0;
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(C_, bottom[0]->shape(1))
      << "# of channels can't change after LayerSetUp!";
  if (N_ == bottom[0]->shape(0) && H_ == bottom[0]->shape(2)
      && W_ == bottom[0]->shape(3)) {
    return;  // No Reshape is needed.
  }
  top[0]->ReshapeLike(*bottom[0]);
  N_ = bottom[0]->shape(0);
  H_ = bottom[0]->shape(2);
  W_ = bottom[0]->shape(3);
  LOG(INFO)<< "Reshaping Blobs/Buffers: ";
  LOG(INFO)<< "Mini-Batch Size: "<< N_ << ", Channels: " << C_ << ", Height: "
      << H_ << ", Width: " << W_;
  batch_size_ = N_ * H_ * W_;  // Effective batch size : B = N*H*W
  count_ = bottom[0]->count();
  inv_batch_size_ = Dtype(1.) / batch_size_;
  bias_correction_factor_ = batch_size_ > 1 ?
      Dtype(batch_size_) / (batch_size_ - 1) : 1;
  batch_sum_multiplier_.Reshape(vector<int>(1, batch_size_));
  // Compute new shape and resize mid_ blob to it.
  old_mid_shape_.clear();
  new_mid_shape_.clear();
  for (int i = 0; i < num_axes_; ++i) {
    new_mid_shape_.push_back(
        bottom[0]->shape(permute_order_.mutable_cpu_data()[i]));
    old_mid_shape_.push_back(bottom[0]->shape(i));
  }
  mid_.Reshape(new_mid_shape_);
  for (int i = 0; i < num_axes_ - 1; ++i) {
    old_steps_.mutable_cpu_data()[i] = bottom[0]->count(i + 1);
    new_steps_.mutable_cpu_data()[i] = mid_.count(i + 1);
  }
  // Perform allocations for large buffers.
  // In the case of GPU mode, directly allocate to GPU memory
  switch (Caffe::mode()) {
  case Caffe::CPU:
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
    caffe_set(count_, Dtype(0), mid_.mutable_cpu_data());
    if (this->phase_ != TEST) {
      caffe_set(count_, Dtype(0), mid_.mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    caffe_gpu_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_gpu_data());
    caffe_gpu_set(count_, Dtype(0), mid_.mutable_gpu_data());
    if (this->phase_ != TEST) {
      caffe_gpu_set(count_, Dtype(0), mid_.mutable_gpu_diff());
    }
#endif
    break;
  }
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::orth_weight_update_cpu() {
  LOG(INFO) << "[Warning] orth_weight_update_cpu() not yet implemented."
      << " Weights may no longer be orthogonal.";
}

/* Modified from the PermutationLayer implementation in
  https://github.com/BVLC/caffe/commit/b68695db42aa79e874296071927536363fe1efbf
  by Wei Liu : https://github.com/weiliu89 */
template <typename Dtype>
void RecursiveConvLayer<Dtype>::permute_blobs_cpu(
    const vector<Blob<Dtype>*>& bottom, const bool channel_last,
    const bool permute_diffs) {
  // Called by both Forward_cpu and Backward_cpu.
  // Permute input blob (data or diff) from N*C*H*W to N*H*W*C or vice-versa.
  // The permuted blob is stored in the buffer mid_.
  vector<int> new_orders, new_steps, old_steps;
  if (channel_last) {
    mid_.Reshape(new_mid_shape_);
    for (int i = 0; i < num_axes_; ++i) {
      new_orders.push_back(permute_order_.cpu_data()[i]);
      new_steps.push_back(new_steps_.cpu_data()[i]);
      old_steps.push_back(old_steps_.cpu_data()[i]);
    }
  } else {
    mid_.Reshape(old_mid_shape_);
    for (int i = 0; i < num_axes_; ++i) {
      new_orders.push_back(inv_permute_order_.cpu_data()[i]);
      new_steps.push_back(old_steps_.cpu_data()[i]);
      old_steps.push_back(new_steps_.cpu_data()[i]);
    }
  }
  // Start permuting bottom blob data
  const Dtype* const bottom_data = bottom[0]->cpu_data();
  Dtype* const top_data = mid_.mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    int old_idx = 0, idx = i;
    for (int j = 0; j < num_axes_; ++j) {
      old_idx += (idx / new_steps[j]) * old_steps[new_orders[j]];
      idx %= new_steps[j];
    }
    top_data[i] = bottom_data[old_idx];
  }
  if (!permute_diffs) { return; }
  // Start permuting bottom blob diffs
  const Dtype* const bottom_diff = bottom[0]->cpu_diff();
  Dtype* const top_diff = mid_.mutable_cpu_diff();
  for (int i = 0; i < count_; ++i) {
    int old_idx = 0, idx = i;
    for (int j = 0; j < num_axes_; ++j) {
      old_idx += (idx / new_steps[j]) * old_steps[new_orders[j]];
      idx %= new_steps[j];
    }
    top_diff[i] = bottom_diff[old_idx];
  }
}

// ----------------------------------------------------------------------------
// ---------------------- Helpers for FORWARD PASS ----------------------------
// ----------------------------------------------------------------------------

template <typename Dtype>
void RecursiveConvLayer<Dtype>::forward_ReLU_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) const {
  const Dtype* const bottom_data = bottom[0]->cpu_data();
  Dtype* const top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0)) +
        negative_slope_ * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::forward_BN_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    const int iter) {
  const int offset = iter * C_;
  const Dtype* const bottom_data = bottom[0]->cpu_data();
  Dtype* const top_data = top[0]->mutable_cpu_data();
  if (bottom[0] != top[0]) {
    caffe_copy(count_, bottom_data, top_data);
  }
  if (!use_global_stats_) {
    // Compute Batch-Mean E(X)
    caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, C_, inv_batch_size_,
      bottom_data, batch_sum_multiplier_.cpu_data(), (Dtype)0.,
      bn_mu_.mutable_cpu_data() + offset);
  }
  // Subtract Batch-Mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)-1., batch_sum_multiplier_.cpu_data(),
      bn_mu_.cpu_data() + offset, (Dtype)1., top_data);
  if (!use_global_stats_) {
    // Compute Batch-Variance E((X-EX)^2)
    caffe_powx(top[0]->count(), top_data, Dtype(2), mid_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, C_, inv_batch_size_,
        mid_.cpu_data(), batch_sum_multiplier_.cpu_data(), (Dtype)0.,
        bn_sigma_.mutable_cpu_data() + offset);
    // Compute and save moving average
    caffe_cpu_axpby(C_, Dtype(1), bn_mu_.cpu_data() + offset,
        moving_average_fraction_,
        this->blobs_[bn_param_offset_]->mutable_cpu_data() + offset);
    caffe_cpu_axpby(C_, bias_correction_factor_, bn_sigma_.cpu_data() + offset,
        moving_average_fraction_,
        this->blobs_[bn_param_offset_ + 1]->mutable_cpu_data() + offset);
    // Compute Batch-St-dev = sqrt(Batch-Variance + epsilon)
    caffe_add_scalar(C_, eps_, bn_sigma_.mutable_cpu_data() + offset);
    caffe_powx(C_, bn_sigma_.cpu_data() + offset , Dtype(0.5),
        bn_sigma_.mutable_cpu_data() + offset);
  }
  // Replicate Batch-St-dev to input size
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.cpu_data(),
      bn_sigma_.cpu_data() + offset, (Dtype)0., mid_.mutable_cpu_data());
  // Divide by Batch-St-dev
  caffe_div(count_, top_data, mid_.cpu_data(), top_data);
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
  if (requires_orth_weight_update_) {
    orth_weight_update_cpu();
  }
  const bool channel_last = true;
  const bool permute_diffs = true;
  if (!use_global_stats_) {
    this->blobs_[bn_param_offset_ + 2]->mutable_cpu_data()[0] *=
        moving_average_fraction_;
    this->blobs_[bn_param_offset_ + 2]->mutable_cpu_data()[0] += 1;
  }
  // Permute bottom from N*C*H*W to N*H*W*C and copy to mid_
  permute_blobs_cpu(bottom, channel_last, !permute_diffs);
  top[0]->ReshapeLike(mid_);
  for (int iter = 0; iter < Nrec_; ++iter) {
    // Standard 1x1 convolution
    const int wt_offset = rand_wt_order_[iter];
    const Dtype* const weights = this->blobs_[wt_offset]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, C_,
        (Dtype)1., mid_.cpu_data(), weights, (Dtype)0.,
        top[0]->mutable_cpu_data());
    // Compute activation function in-place
    forward_activation_func_cpu(top, top);  // a_{i+1} = \sigma(a_{i+1});
    // Apply BN in-place
    forward_BN_cpu(top, top, iter);
    if (iter == Nrec_ - 1) {
      // Permute top from N*H*W*C to N*C*H*W and copy to mid_
      permute_blobs_cpu(top, !channel_last, !permute_diffs);
      top[0]->ReshapeLike(mid_);
      caffe_copy(count_, mid_.cpu_data(), top[0]->mutable_cpu_data());
    } else {
      // mid_ <- top; //a_i <- a_{i+1};
      caffe_copy(count_, top[0]->cpu_data(), mid_.mutable_cpu_data());
    }
  }
}

// ----------------------------------------------------------------------------
// --------------------- Helpers for BACKWARD PASS ----------------------------
// ----------------------------------------------------------------------------

template <typename Dtype>
void RecursiveConvLayer<Dtype>::backward_ReLU_cpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) const {
  const Dtype* const top_data = top[0]->cpu_data();
  Dtype* const bottom_data = bottom[0]->mutable_cpu_data();
  const Dtype* const top_diff = top[0]->cpu_diff();
  Dtype* const bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < count_; ++i) {
    bottom_diff[i] = top_diff[i] * ((top_data[i] > 0) +
        negative_slope_ * (top_data[i] <= 0));
    bottom_data[i] = top_data[i] * ((top_data[i] > 0) +
        inv_negative_slope_ *(top_data[i] <= 0));
  }
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::backward_BN_cpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom,
    const int iter) {
  const int offset = iter * C_;
  if (bottom[0] == top[0]) {
    caffe_copy(count_, top[0]->cpu_diff(), mid_.mutable_cpu_diff());
  }
  Dtype* const bottom_data = bottom[0]->mutable_cpu_data();
  const Dtype* const top_data = top[0]->cpu_data();
  Dtype* const bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* const top_diff =
      bottom[0] == top[0] ? mid_.cpu_diff() : top[0]->cpu_diff();

  // Replicate Batch-St-dev to input size
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.cpu_data(),
      bn_sigma_.cpu_data() + offset, (Dtype)0., mid_.mutable_cpu_data());
  if (use_global_stats_) {
    caffe_div(count_, top[0]->cpu_diff(), mid_.cpu_data(), bottom_diff);
    // Invert BN --> Multiply by Batch-St-Dev
    caffe_mul(count_, top_data, mid_.cpu_data(), bottom_data);
    // Invert BN --> Add Batch-Mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
        (Dtype)1., batch_sum_multiplier_.cpu_data(),
        bn_mu_.cpu_data() + offset, (Dtype)1., bottom_data);
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
  caffe_mul(count_, top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, C_, (Dtype)1., bottom_diff,
      batch_sum_multiplier_.cpu_data(), (Dtype)0.,
      temp_bn_sum_.mutable_cpu_data());
  // reshape (broadcast) the above
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.cpu_data(),
      temp_bn_sum_.cpu_data(), (Dtype)0., bottom_diff);
  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(count_, top_data, bottom_diff, bottom_diff);
  // sum(dE/dY)
  caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, C_, (Dtype)1., top_diff,
      batch_sum_multiplier_.cpu_data(), 0., temp_bn_sum_.mutable_cpu_data());
  // reshape (broadcast) the above: sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
  (Dtype)1., batch_sum_multiplier_.cpu_data(), temp_bn_sum_.cpu_data(),
  (Dtype)1., bottom_diff);
  // dE/dY - mean(dE/dY)- (mean(dE/dY \cdot Y) \cdot Y)
  caffe_cpu_axpby(count_, Dtype(1), top_diff, Dtype(-1. * inv_batch_size_),
      bottom_diff);
  // note: mid_.cpu_data() contains sqrt(var(X)+eps)
  caffe_div(count_, bottom_diff, mid_.cpu_data(), bottom_diff);
  // Invert BN --> Multiply by Batch-St-Dev
  caffe_mul(count_, top_data, mid_.cpu_data(), bottom_data);
  // Invert BN --> Add Batch-Mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1,
      (Dtype)1., batch_sum_multiplier_.cpu_data(), bn_mu_.cpu_data() + offset,
      (Dtype)1., bottom_data);
}

template <typename Dtype>
void RecursiveConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!this->param_propagate_down_[0] && !propagate_down[0]) {
    return;
  }
  const bool channel_last = true;
  const bool permute_diffs = true;
  // Permute top data & diffs from N*C*H*W to N*H*W*C and copy to mid_
  permute_blobs_cpu(top, channel_last, permute_diffs);
  bottom[0]->ReshapeLike(mid_);
  caffe_copy(count_, mid_.cpu_data(), bottom[0]->mutable_cpu_data());
  caffe_copy(count_, mid_.cpu_diff(), bottom[0]->mutable_cpu_diff());
  // TOP Data & Diff are now in BOTTOM, permuted in order N*H*W*C
  for (int iter = Nrec_ - 1; iter >= 0; --iter) {
    backward_BN_cpu(bottom, bottom, iter);
    backward_activation_func_cpu(bottom, bottom);
    // Invert data (bottom[0])*inv(W)->data(mid_),
    // compute diff(W) and backprop diff(bottom[0])->diff(mid_).
    const int wt_offset = rand_wt_order_[iter];
    const Dtype* const weights = this->blobs_[wt_offset]->cpu_data();
    Dtype* const weights_diff = this->blobs_[wt_offset]->mutable_cpu_diff();
    // First get BOTTOM data using the inverse of weights
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, C_, C_,
        (Dtype)1., bottom[0]->cpu_data(), weights, (Dtype)0.,
        mid_.mutable_cpu_data());
    // Note: BOTTOM Data is now in mid_, TOP Data & Diff still in bottom[0]
    // Compute diff with respect to weights if needed
    if (this->param_propagate_down_[0]) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, C_, C_, batch_size_,
          (Dtype)1., mid_.cpu_data(), bottom[0]->cpu_diff(), (Dtype)1.,
          weights_diff);
    }
    // Compute diff with respect to bottom activation.
    // We must always do this, even if propagate_down[0] is false.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, C_, C_,
        (Dtype)1., bottom[0]->cpu_diff(), weights, (Dtype)0.,
        mid_.mutable_cpu_diff());
    // Note: BOTTOM Diff is now in mid_, TOP Data & Diff are still in bottom[0]
    // Transfer Data & Diff from mid_ to bottom[0]
    caffe_copy(count_, mid_.cpu_data(), bottom[0]->mutable_cpu_data());
    caffe_copy(count_, mid_.cpu_diff(), bottom[0]->mutable_cpu_diff());
  }
  // Permute bottom data & diffs from N*H*W*C to N*C*H*W and copy to mid_
  permute_blobs_cpu(bottom, !channel_last, permute_diffs);
  bottom[0]->ReshapeLike(mid_);
  caffe_copy(count_, mid_.cpu_data(), bottom[0]->mutable_cpu_data());
  caffe_copy(count_, mid_.cpu_diff(), bottom[0]->mutable_cpu_diff());

  // The next forward pass will project the solver's regularized weight diffs
  // on to the Tangent Space in the Stiefel manifold of the weights, and
  // recompute the new weights using Cayley's transform. This will ensure that
  // the weights always remain orthogonal in a natural way while simultaneously
  // optimizing the problem at hand.
  requires_orth_weight_update_ = true;
}

#ifdef CPU_ONLY
STUB_GPU(RecursiveConvLayer);
#endif

INSTANTIATE_CLASS(RecursiveConvLayer);
REGISTER_LAYER_CLASS(RecursiveConv);

}  // namespace caffe
