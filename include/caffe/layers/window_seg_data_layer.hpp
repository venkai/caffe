#ifndef CAFFE_WINDOW_SEG_DATA_LAYER_HPP_
#define CAFFE_WINDOW_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from windows of image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowSegDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit WindowSegDataLayer(const LayerParameter& param)
  : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  Blob<Dtype> transformed_label_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  typedef struct SegItems {
    std::string imgfn;
    std::string segfn;
    int x1, y1, x2, y2;
  } SEGITEMS;

  vector<SEGITEMS> lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_WINDOW_SEG_DATA_LAYER_HPP_
