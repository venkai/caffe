#ifndef CAFFE_XXX_LAYER_HPP_
#define CAFFE_XXX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
template <typename Ftype, typename Btype>
class ReshapeLayer : public Layer<Ftype, Btype> {
 public:
  explicit ReshapeLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Reshape"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  void dense_reshape_cpu(const vector<Blob*>& src, const vector<Blob*>& dst,
      const bool inv, const bool reshape_diffs);
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    if (r_ > 1) {
      dense_reshape_cpu(bottom, top, inv_dense_reshape_, false);
    }
  }
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
    if (r_ > 1) {
      dense_reshape_cpu(top, bottom, !inv_dense_reshape_, true);
    }
  }
#ifndef CPU_ONLY
  void dense_reshape_gpu(const vector<Blob*>& src, const vector<Blob*>& dst,
      const bool inv, const bool reshape_diffs);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    if (r_ > 1) {
      dense_reshape_gpu(bottom, top, inv_dense_reshape_, false);
    }
  }
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
    if (r_ > 1) {
      dense_reshape_gpu(top, bottom, !inv_dense_reshape_, true);
    }
  }
#endif
  /// @brief vector of axes indices whose dimensions we'll copy from the bottom
  vector<int> copy_axes_;
  /// @brief the index of the axis whose dimension we infer, or -1 if none
  int inferred_axis_;
  /// @brief the product of the "constant" output dimensions
  int constant_count_;
  /** r = dense_reshape_scale (which is provided by reshape_param).
   *  For r > 1, a blob of shape [N x (C * r * r) x H x W] is reshaped to
   *  [N x C x (H * r) x (W * r)] according to paper: "Real-Time Single Image
   *  and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
   *  Neural Network [https://arxiv.org/abs/1609.05158]."
   *  Note that for r > 1, the params: axis, num_axes, shape are ignored, and
   *  Forward/Backward passes are no longer be no-ops, since the reshape
   *  also involves a permutation of the input blob.
   */
  int r_;
  /// @brief r2 = r^2.
  int r2_;
  /** inv_dense_reshape (which is provided by reshape_param) when true,
   *  reverses the dense reshape op above, i.e. [N x C x (H * r) x (W * r)]
   *  is reshaped to [N x (C * r * r) x H x W].
   */
  bool inv_dense_reshape_;
};

}  // namespace caffe

#endif  // CAFFE_XXX_LAYER_HPP_
