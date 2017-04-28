#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/image_superresolution_data_layer.hpp"

namespace caffe {

template <typename Dtype>
ImageSuperResolutionDataLayer<Dtype>::~ImageSuperResolutionDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageSuperResolutionDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const bool is_label_color
      = this->layer_param_.image_data_param().is_label_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  TransformationParameter transform_param =
      this->layer_param_.transform_param();
  CHECK(transform_param.has_ds_param())
      << "ImageSuperResolutionDataLayer requires ds_param";
  const int ds_scale = transform_param.ds_param().ds_scale();
  CHECK(transform_param.has_mean_file() == false)
      << "ImageSuperResolutionDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
  (new_height > 0 && new_width > 0))
      << "Current implementation requires "
      << "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    lines_.push_back(imgfn);
  }
  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
    this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  int height, width;
  const int crop_size = transform_param.crop_size();
  if (crop_size > 0) {
    height = width = crop_size;
  } else if (new_height > 0 && new_width > 0) {
    height = new_height;
    width = new_width;
  } else {
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_],
    new_height, new_width, true);
    CHECK(cv_img.data) << "Fail to load input: "
        << root_folder + lines_[lines_id_];
    height = cv_img.rows;
    CHECK(height % ds_scale == 0) << "height must be a multiple of ds_scale";
    height /= ds_scale;
    width = cv_img.cols;
    CHECK(width % ds_scale == 0) << "width must be a multiple of ds_scale";
    width /= ds_scale;
  }
  // Number of image/label channels.
  const int channels = is_color ? 3 : 1;
  const int label_channels = is_label_color ? 3 : 1;
  const int label_height = height * ds_scale;
  const int label_width = width * ds_scale;
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  // data
  const vector<int> data_shape{batch_size, channels, height, width};
  top[0]->Reshape(data_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(data_shape);
  }
  const vector<int> one_data_shape{1, channels, height, width};
  this->transformed_data_.Reshape(one_data_shape);
  // label
  const vector<int>
      label_shape{batch_size, label_channels, label_height, label_width};
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
  const vector<int>
      one_label_shape{1, label_channels, label_height, label_width};
  this->transformed_label_.Reshape(one_label_shape);
  // image dimensions, for each image, stores (img_height, img_width)
  const vector<int> data_dim_shape{batch_size, 1, 1, 2};
  top[2]->Reshape(data_dim_shape);
  this->prefetch_data_dim_.Reshape(data_dim_shape);
  //image
  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
  << top[1]->channels() << "," << top[1]->height() << ","
  << top[1]->width();
  // image_dim
  LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
  << top[2]->channels() << "," << top[2]->height() << ","
  << top[2]->width();
}

template <typename Dtype>
void ImageSuperResolutionDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
  static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageSuperResolutionDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->transformed_data_.count());
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  Dtype* top_data_dim = this->prefetch_data_dim_.mutable_cpu_data();
  const int max_height = batch->label_.height();
  const int max_width  = batch->label_.width();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  string root_folder   = image_data_param.root_folder();
  const int lines_size = lines_.size();
  int top_data_dim_offset;
  for (int item_id = 0; item_id < batch_size;) {
    // get a blob
    timer.Start();
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
    int img_row, img_col;
    const int crop_size = this->layer_param_.transform_param().crop_size(); 
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_],
    new_height, new_width, true, &img_row, &img_col);
    if (this->layer_param_.transform_param().ds_param().skip_small()
        && crop_size > 0 && (img_row < crop_size || img_col < crop_size)) {
      ++lines_id_;
      continue;
    }
    top_data_dim_offset = this->prefetch_data_dim_.offset(item_id);
    top_data_dim[top_data_dim_offset] =
        static_cast<Dtype>(std::min(max_height, img_row));
    top_data_dim[top_data_dim_offset + 1] =
        static_cast<Dtype>(std::min(max_width, img_col));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);
    // Custom Transform for SR exps
    this->data_transformer_->Transform_SR(cv_img, 
    &(this->transformed_data_), &(this->transformed_label_));
    trans_time += timer.MicroSeconds();
    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    item_id++;
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSuperResolutionDataLayer);
REGISTER_LAYER_CLASS(ImageSuperResolutionData);

}  // namespace caffe

#endif  // USE_OPENCV
