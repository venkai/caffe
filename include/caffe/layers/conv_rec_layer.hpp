#ifndef CAFFE_RECURSIVE_CONV_LAYER_HPP_
#define CAFFE_RECURSIVE_CONV_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class RecursiveConvLayer : public Layer<Dtype> {
 public:
  explicit RecursiveConvLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }
  virtual inline const char* type() const { return "RecursiveConv"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void orth_weight_update_cpu();
  void forward_ReLU_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const;
  void backward_ReLU_cpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom) const;
  void permute_blobs_cpu(const vector<Blob<Dtype>*>& bottom,
      const bool channel_last, const bool permute_diffs);
  void init_param_blobs_cpu();
  void forward_BN_cpu(const vector<Blob<Dtype>*>& top,
      const int iter);
  void backward_BN_cpu(const vector<Blob<Dtype>*>& bottom,
      const int iter);

#ifndef CPU_ONLY
  void orth_weight_update_gpu();
  void forward_ReLU_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const;
  void backward_ReLU_gpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom) const;
  void permute_blobs_gpu(const vector<Blob<Dtype>*>& bottom,
      const bool channel_last, const bool permute_diffs);
  void init_param_blobs_gpu();
  void forward_BN_gpu(const vector<Blob<Dtype>*>& top,
      const int iter);
  void backward_BN_gpu(const vector<Blob<Dtype>*>& bottom,
      const int iter);
#endif

  // --- Large Buffer blobs ---
  // Roughly in decreasing order of size.
  // These should ideally be reused by layers which inherit from this class.
  Blob<Dtype> mid_;  // N*C*H*W buffer used for most intermediate computations.
  Blob<Dtype> bn_mu_;  // local BN mean: Nrec_ x C_.
  Blob<Dtype> bn_sigma_;  // local BN sigma: Nrec_ x No_.
  Blob<Dtype> batch_sum_multiplier_;  // All ones size of 1x (N_*H_*W_).

 private:
  // Wrappers for various invertible activation functions
  // (currently only ReLU with negative slope.)
  inline void forward_activation_func_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
    forward_ReLU_cpu(bottom, top);
  }
  inline void backward_activation_func_cpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom) const {
    backward_ReLU_cpu(top, bottom);
  }
#ifndef CPU_ONLY
  inline void forward_activation_func_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
    forward_ReLU_gpu(bottom, top);
  }
  inline void backward_activation_func_gpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom) const {
    backward_ReLU_gpu(top, bottom);
  }
#endif

  // --- Small Buffers/Vars ---
  // These are made private to avoid namespace clutter

  static constexpr int num_axes_ = 4;

  // Initialized "false" at LayerSetUp. Always "true" after first Backward Pass.
  // The next Forward pass will project the solver's regularized weight diffs
  // on to the Tangent Space in the Stiefel manifold of the weights, and
  // recompute the new weights using Cayley's transform. This will ensure that
  // the weights always remain orthogonal in a natural way while simultaneously
  // optimizing the problem at hand.
  bool requires_orth_weight_update_;

  // Whether initial weights neeed to be orthogonalized.
  bool requires_orth_weight_init_;

  // Mini-batch size, # channels, height, width respectively.
  int N_, C_, H_, W_;
  int count_;  // N*C*H*W
  int batch_size_;  // Effective batch size : B = N*H*W.
  bool requires_init_param_blobs_;

  // For permuting blobs from N*C*H*W to N*H*W*C or vice-versa.
  Blob<int> permute_order_, inv_permute_order_, old_steps_, new_steps_;
  vector<int> old_mid_shape_, new_mid_shape_;

  // For QR factorization
  Blob<Dtype> wt_buffer_;  // C_ x C_ buffer used in Cayley transform.
  Blob<Dtype> eye_;  // Identity matrix C_ x C_.
  Blob<Dtype> A_;  // A = trans(G)*W - trans(W)*G, where G = diff(W).
  Blob<Dtype> tau_;  // C_ x 1 buffer used for QR factorization.
  Blob<Dtype> workspace_;  // buffer on GPU memory for cusolver functions.
  int Lwork_;  // Size of workspace_.
  Blob<int> dev_info_;  // Describes if QR was successful.

  // For Recursive Convolutions
  int Nrec_;  // # of recursive convolutions.
  int Nwts_;  // # of unique weights (<= Nrec_).
  vector<int> rand_wt_order_;  // Ordering of weights.
  string orth_norm_type_;  // How to normalize A in orth_weight_update_*.

  // For activation functions
  bool apply_pre_activation_;  // Add an activation layer before conv.
  // For ReLU
  Dtype negative_slope_, inv_negative_slope_;

  // For batch-normalization
  bool reset_bn_params_;  // Reset global batch-norm params.
  bool apply_pre_bn_;  // Add a batch-norm layer before conv.
  int bn_param_offset_;  // Beginning index of global BN param blobs.
  Blob<Dtype> temp_bn_;  // Cache for BN computations: 1 x C_.
  bool use_global_stats_;
  Dtype moving_average_fraction_, eps_, bias_correction_factor_;
  Dtype inv_batch_size_;  // Precomputed inverse of batch size.

  // For Scale/Shift after batch-norm.
  bool apply_scale_, apply_bias_;

  // "BN + activation" or "activation + BN"
  bool bn_after_activation_;

  // --- Temporary (for debugging) ---
  inline void test_print(const int M, const int N, const Dtype* const A) const {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        printf("%0.4f\t", A[(i * N) + j]);
      }
      printf("\n");
    }
  }
  inline void test_print(const int N, const Dtype* const A) const {
    test_print(N, N, A);
  }
  inline void test_print(const Blob<Dtype>* const A) const {
    if (A->num_axes() > 1) {
      test_print(A->shape(0), A->shape(1), A->cpu_data());
    } else {
      test_print(1, A->shape(0), A->cpu_data());
    }
  }

};

}  // namespace caffe

#endif  // CAFFE_RECURSIVE_CONV_LAYER_HPP_
