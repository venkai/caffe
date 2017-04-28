#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

boost::mt19937 gen;
const double prob_eps = 0.01;

// Venkat (6-Oct-2016): taken from
// https://github.com/ducha-aiki/caffe/blob/augmentations/src/caffe/data_transformer.cpp
inline int roll_weighted_die(const std::vector<double> probabilities) {
  std::vector<double> cumulative;
  std::partial_sum(&probabilities[0], &probabilities[0] + probabilities.size(),
          std::back_inserter(cumulative));
  boost::uniform_real<> dist(0, cumulative.back());
  boost::variate_generator<boost::mt19937&,
          boost::uniform_real<> > die(gen, dist);

  // Find the position within the sequence and add 1
  return (std::lower_bound(cumulative.begin(), cumulative.end(), die())
          - cumulative.begin());
}

namespace caffe {

#ifdef USE_OPENCV
inline int sr_to_cv_interp(int sr_interp_param) {
  int interp_mode = -1;
  switch (sr_interp_param) {
  case SuperResolutionParameter_Interp_mode_AREA:
    { interp_mode = cv::INTER_AREA; break; }
  case SuperResolutionParameter_Interp_mode_CUBIC:
    { interp_mode = cv::INTER_CUBIC; break; }
  case SuperResolutionParameter_Interp_mode_LINEAR:
    { interp_mode = cv::INTER_LINEAR; break; }
  case SuperResolutionParameter_Interp_mode_NEAREST:
    { interp_mode = cv::INTER_NEAREST; break; }
  case SuperResolutionParameter_Interp_mode_LANCZOS4:
    { interp_mode = cv::INTER_LANCZOS4; break; }
  }
  return interp_mode;
}
#endif

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decode and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

/*
notice:
this code is based on the following implementation.
https://bitbucket.org/deeplab/deeplab-public/
*/
template<typename Dtype>
void DataTransformer<Dtype>::TransformImgAndSeg(
    const std::vector<cv::Mat>& cv_img_seg,
    Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob,
    const int ignore_label) {
  CHECK(cv_img_seg.size() == 2) << "Input must contain image and seg.";
  const int img_channels = cv_img_seg[0].channels();
  // height and width may change due to pad for cropping
  int img_height = cv_img_seg[0].rows;
  int img_width = cv_img_seg[0].cols;

  const int seg_channels = cv_img_seg[1].channels();
  int seg_height = cv_img_seg[1].rows;
  int seg_width = cv_img_seg[1].cols;

  const int data_channels = transformed_data_blob->channels();
  const int data_height = transformed_data_blob->height();
  const int data_width = transformed_data_blob->width();

  const int label_channels = transformed_label_blob->channels();
  const int label_height = transformed_label_blob->height();
  const int label_width = transformed_label_blob->width();

  CHECK_EQ(img_channels, data_channels);
  CHECK_EQ(img_height, seg_height);
  CHECK_EQ(img_width, seg_width);

  CHECK_EQ(data_height, label_height);
  CHECK_EQ(data_width, label_width);
  CHECK_EQ(seg_channels, label_channels);

  CHECK(cv_img_seg[0].depth() == CV_8U)
      << "Image data type must be unsigned byte";
  CHECK(cv_img_seg[1].depth() == CV_8U)
      << "Seg data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
    << "Specify either 1 mean_value or as many as channels: "
    << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img_seg[0];
  cv::Mat cv_cropped_seg = cv_img_seg[1];

  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);
  cv_cropped_seg.convertTo(cv_cropped_seg, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  int pad_height = std::max(crop_size - img_height, 0);
  int pad_width  = std::max(crop_size - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
        0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(mean_values_[0],
        mean_values_[1], mean_values_[2]));
    cv::copyMakeBorder(cv_cropped_seg, cv_cropped_seg, 0, pad_height,
        0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(ignore_label));
    // update height/width
    img_height = cv_cropped_img.rows;
    img_width = cv_cropped_img.cols;

    seg_height = cv_cropped_seg.rows;
    seg_width = cv_cropped_seg.cols;
  }
  // crop img/seg
  if (crop_size) {
    CHECK_EQ(crop_size, data_height);
    CHECK_EQ(crop_size, data_width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_seg = cv_cropped_seg(roi);
  }

  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_seg.data);

  Dtype* transformed_data  = transformed_data_blob->mutable_cpu_data();
  Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;
  const double* label_ptr;

  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_cropped_img.ptr<double>(h);
    label_ptr = cv_cropped_seg.ptr<double>(h);

    int data_index = 0;
    int label_index = 0;

    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width
              + w_off + w;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
            (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }

      // for segmentation
      for (int c = 0; c < label_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width
              + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
        transformed_label[top_index] = pixel;
      }
    }
  }
}

// Added by Venkat 3-OCT-16
template<typename Dtype>
void DataTransformer<Dtype>::TransformInputAndLabel(
    const std::vector<cv::Mat>& cv_input_label,
    Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob) {
  CHECK(cv_input_label.size() == 2) << "Input must contain input and label.";

  const int img_channels = cv_input_label[0].channels();
  // height and width may change due to pad for cropping
  int img_height = cv_input_label[0].rows;
  int img_width = cv_input_label[0].cols;

  int label_channels = cv_input_label[1].channels();
  int label_height = cv_input_label[1].rows;
  int label_width = cv_input_label[1].cols;

  const int data_channels = transformed_data_blob->channels();
  const int data_height = transformed_data_blob->height();
  const int data_width = transformed_data_blob->width();

  const int curr_label_channels = transformed_label_blob->channels();
  const int curr_label_height = transformed_label_blob->height();
  const int curr_label_width = transformed_label_blob->width();


  CHECK_EQ(img_channels, data_channels);
  CHECK_EQ(img_height, label_height);
  CHECK_EQ(img_width, label_width);

  CHECK_EQ(data_height, curr_label_height);
  CHECK_EQ(data_width, curr_label_width);
  CHECK_EQ(curr_label_channels, label_channels);

  CHECK(cv_input_label[0].depth() == CV_8U)
      << "Image data type must be unsigned byte";
  CHECK(cv_input_label[1].depth() == CV_8U)
      << "Label data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_input_label[0];
  cv::Mat cv_cropped_label = cv_input_label[1];

  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);
  cv_cropped_label.convertTo(cv_cropped_label, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  int pad_height = std::max(crop_size - img_height, 0);
  int pad_width  = std::max(crop_size - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    if (img_channels > 1) {
      cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(mean_values_[0],
          mean_values_[1], mean_values_[2]));
    } else {
      cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(mean_values_[0]));
    }
    if (label_channels >1) {
      cv::copyMakeBorder(cv_cropped_label, cv_cropped_label, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(mean_values_[0],
          mean_values_[1], mean_values_[2]));
    } else {
      cv::copyMakeBorder(cv_cropped_label, cv_cropped_label, 0, pad_height,
          0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(mean_values_[0]));
    }
    // update height/width
    img_height = cv_cropped_img.rows;
    img_width = cv_cropped_img.cols;

    label_height = cv_cropped_label.rows;
    label_width = cv_cropped_label.cols;
  }
  // crop img/seg
  if (crop_size) {
    CHECK_EQ(crop_size, data_height);
    CHECK_EQ(crop_size, data_width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_label = cv_cropped_label(roi);
  }
  CHECK(cv_cropped_img.data);
  CHECK(cv_cropped_label.data);

  // Venkat: Insert Other Useful Image-to-Image Regression Transformations here
  // SUPER-RESOLUTION
  if (param_.has_super_res_param()) {
    const float min_sr = param_.super_res_param().min_sr_scale();
    const float max_sr = param_.super_res_param().max_sr_scale();
    CHECK(min_sr >= 1 && max_sr >= min_sr)
        << "max_sr_scale >= min_sr_scale >= 1.";
    CHECK((param_.super_res_param().has_interp_mode() &&
        !param_.super_res_param().has_down_interp_mode() &&
        !param_.super_res_param().has_up_interp_mode()) ||
        (!param_.super_res_param().has_interp_mode() &&
        param_.super_res_param().has_down_interp_mode() &&
        param_.super_res_param().has_up_interp_mode()) ||
        (!param_.super_res_param().has_interp_mode() &&
        !param_.super_res_param().has_down_interp_mode() &&
        !param_.super_res_param().has_up_interp_mode()))
    << "\n\nSpecify EITHER no interpolation (everything defaults to BILINEAR),"
    << "\nOR specify only interp_mode (used for both up/down sampling),"
    << "\nOR separately specify down_interp_mode/up_interp_mode "
    << "(without specifying interp_mode).";

    float rand_sr = 3.0;
    if (max_sr == min_sr) {
      rand_sr = max_sr;
    } else {
      // Generate a scaling factor "rand_sr" uniformly randomly
      // between min_sr and max_sr
      caffe_rng_uniform(1, min_sr, max_sr, &rand_sr);
    }
    if (rand_sr > 1.0) {
      int down_interp_mode, up_interp_mode;
      if (param_.super_res_param().has_interp_mode()) {
        down_interp_mode = up_interp_mode =
            sr_to_cv_interp(param_.super_res_param().interp_mode());
      } else {
        down_interp_mode =
            sr_to_cv_interp(param_.super_res_param().down_interp_mode());
        up_interp_mode =
            sr_to_cv_interp(param_.super_res_param().up_interp_mode());
      }
      CHECK(down_interp_mode != -1)
          << "Invalid Interpolation for downsampling";
      CHECK(up_interp_mode != -1) << "Invalid Interpolation for upsampling";
      cv::Mat cv_resized_img;
      // Downscale resized image by factor of "rand_sr" using
      // desired down-interpolation method
      cv::resize(cv_cropped_img, cv_resized_img, cv::Size(),
          (1.0 / rand_sr), (1.0 / rand_sr), down_interp_mode);
      // Convert to [0,255] range integer type for realistic evaluation
      cv_resized_img.convertTo(cv_resized_img, CV_8U);
      // Convert back to double before resizing
      cv_resized_img.convertTo(cv_resized_img, CV_64F);
      // Resize back to original size using desired up-interpolation method
      cv::resize(cv_resized_img, cv_cropped_img,
          cv::Size(cv_cropped_img.cols, cv_cropped_img.rows),
          0, 0, up_interp_mode);
    }
  }

  // DENOISING
  const int num_wgn_policies = param_.wgn_param_size();
  if (num_wgn_policies > 0) {
    int policy_num = 0;
    if (num_wgn_policies > 1) {
      std::vector<double> probabilities;
      double prob_sum = 0;
      for (unsigned int i = 0; i < num_wgn_policies; i++) {
        double prob = 0.0;
        if (param_.wgn_param(i).has_prob()) {
          prob = param_.wgn_param(i).prob();
        } else {
          prob = 1.0 / num_wgn_policies;
        }
        CHECK_GE(prob, 0);
        CHECK_LE(prob, 1);
        prob_sum += prob;
        probabilities.push_back(prob);
      }
      CHECK_NEAR(prob_sum, 1.0, prob_eps);
      policy_num = roll_weighted_die(probabilities);
    }
    const WGNParameter wgn_param = param_.wgn_param(policy_num);
    CHECK((wgn_param.has_noise_std() && !wgn_param.has_min_noise_std()
        && !wgn_param.has_max_noise_std()) ||
        (!wgn_param.has_noise_std() && wgn_param.has_min_noise_std()
        && wgn_param.has_max_noise_std()) ||
        (!wgn_param.has_noise_std() && !wgn_param.has_min_noise_std()
        && !wgn_param.has_max_noise_std()))
        << "\n\nSpecify EITHER no noise std (defaults to 25), "
        << "OR specify only noise_std,\n"
        << "OR separately specify min_noise_std/max_noise_std "
        << "(without specifying noise_std).";
    float noise_std = 0.0;
    if (wgn_param.has_noise_std() || (!wgn_param.has_min_noise_std()
        && !wgn_param.has_max_noise_std())) {
      CHECK_GE(wgn_param.noise_std(), 0);
      noise_std = wgn_param.noise_std();
    } else if (wgn_param.min_noise_std() == wgn_param.max_noise_std()) {
      CHECK_GE(wgn_param.min_noise_std(), 0);
      noise_std = wgn_param.min_noise_std();
    } else {
      CHECK_GE(wgn_param.min_noise_std(), 0);
      CHECK_GE(wgn_param.max_noise_std(), wgn_param.min_noise_std());
      caffe_rng_uniform(1, wgn_param.min_noise_std(),
          wgn_param.max_noise_std(), &noise_std);
    }
    cv::Mat cv_noisy_img = cv::Mat(cv_cropped_img.size(), CV_64F);
    // Generate White Gaussian Noise of standard deviation noise_std
    cv::randn(cv_noisy_img, 0, noise_std);
    // Add noise to original input image
    cv_cropped_img += cv_noisy_img;
    // Convert to [0,255] range integer type for realistic evaluation
    cv_cropped_img.convertTo(cv_cropped_img, CV_8U);
    // Convert back to double
    cv_cropped_img.convertTo(cv_cropped_img, CV_64F);
  }

  // JPEG DEBLOCKING
  if (param_.has_jpeg_param()) {
    const float min_quality = param_.jpeg_param().min_quality();
    const float max_quality = param_.jpeg_param().max_quality();
    CHECK(min_quality >= 1 && max_quality >= min_quality)
        << "max_quality >= min_quality >= 1.";
    float rand_quality = 20.0;
    if (max_quality == min_quality) {
      rand_quality = max_quality;
    } else {
      // Generate a scaling factor "rand_quality" uniformly randomly
      // between min_quality and max_quality
      caffe_rng_uniform(1, min_quality, max_quality, &rand_quality);
    }
    std::vector<uchar> buf;
    std::vector<int> cv_jpeg_params;
    cv_jpeg_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    cv_jpeg_params.push_back(rand_quality);
    cv::imencode(".jpg", cv_cropped_img, buf, cv_jpeg_params);
    if (img_channels > 1) {
      cv_cropped_img = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
    } else {
      cv_cropped_img = cv::imdecode(buf, CV_LOAD_IMAGE_GRAYSCALE);
    }
    // Convert back to double
    cv_cropped_img.convertTo(cv_cropped_img, CV_64F);
  }

  // --- End Image-to-Image Regression Transformations ---

  Dtype* transformed_data  = transformed_data_blob->mutable_cpu_data();
  Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;
  const double* label_ptr;

  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_cropped_img.ptr<double>(h);
    label_ptr = cv_cropped_label.ptr<double>(h);

    int data_index = 0;
    int label_index = 0;

    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width
              + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width
              + w_off + w;
          transformed_data[top_index] =
          (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
            (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }

      // for label
      for (int c = 0; c < label_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width
              + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
        transformed_label[top_index] = pixel;
      }
    }
  }
}

// Added by Venkat 20-OCT-16
template<typename Dtype>
void DataTransformer<Dtype>::Transform_SR(const cv::Mat& cv_img_full,
    Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob) {
  CHECK(cv_img_full.depth() == CV_8U)
      << "Image data type must be unsigned byte";
  const int ds_scale = param_.ds_param().ds_scale();
  const int img_channels = cv_img_full.channels();
  // height and width may change due to pad for cropping
  int img_height = cv_img_full.rows;
  int img_width = cv_img_full.cols;

  const int data_channels = transformed_data_blob->channels();
  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();

  const int label_channels = transformed_label_blob->channels();
  const int label_height   = transformed_label_blob->height();
  const int label_width    = transformed_label_blob->width();

  const int crop_size = param_.crop_size() * ds_scale;
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);

  CHECK_GT(img_channels, 0);

  int h_off = 0;
  int w_off = 0;

  // transform to double, since we will pad mean pixel values
  cv::Mat cv_img = cv_img_full;
  cv_img.convertTo(cv_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  int pad_height = std::max(crop_size - img_height, 0);
  int pad_width  = std::max(crop_size - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    if (img_channels > 1) {
      cv::copyMakeBorder(cv_img, cv_img, 0, pad_height,
      0, pad_width, cv::BORDER_CONSTANT,
      cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));
    } else {
      cv::copyMakeBorder(cv_img, cv_img, 0, pad_height,
      0, pad_width, cv::BORDER_CONSTANT,
      cv::Scalar(mean_values_[0]));
    }
    // update height/width
    img_height = cv_img.rows;
    img_width = cv_img.cols;
  }
  // crop img/seg
  if (crop_size) {
    CHECK_EQ(crop_size, label_height);
    CHECK_EQ(crop_size, label_width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      // CHECK: use middle crop
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_img = cv_img(roi);
  }
  CHECK(cv_img.data);
  cv::Mat cv_resized_img;
  const int down_interp_mode
      = sr_to_cv_interp(param_.ds_param().interp_mode());
  // Downscale resized image by factor of "ds_scale"
  // using desired down-interpolation method
  cv::resize(cv_img, cv_resized_img,
      cv::Size(cv_img.cols / ds_scale, cv_img.rows / ds_scale), 0, 0,
      down_interp_mode);
  // Convert to [0, 255] range integer type for realistic evaluation
  cv_resized_img.convertTo(cv_resized_img, CV_8U);
  // Convert back to double
  cv_resized_img.convertTo(cv_resized_img, CV_64F);

  Dtype* transformed_data  = transformed_data_blob->mutable_cpu_data();
  Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;
  const double* label_ptr;

  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_resized_img.ptr<double>(h);
    int data_index = 0;
    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < data_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width
              + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        transformed_data[top_index] = pixel * scale;
      }
    }
  }

  for (int h = 0; h < label_height; ++h) {
    label_ptr = cv_img.ptr<double>(h);
    int label_index = 0;
    for (int w = 0; w < label_width; ++w) {
      // for label
      for (int c = 0; c < label_channels; ++c) {
        if (do_mirror) {
          top_index = (c * label_height + h) * label_width
              + (label_width - 1 - w);
        } else {
          top_index = (c * label_height + h) * label_width + w;
        }
        Dtype pixel = static_cast<Dtype>(label_ptr[label_index++]);
        transformed_label[top_index] = pixel * scale;
      }
    }
  }
}

#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
