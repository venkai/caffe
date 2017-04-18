/*
notice:
this code is based on the following implementation.
https://bitbucket.org/deeplab/deeplab-public/
*/

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template <typename Dtype>
    ImageColorizationDataLayer<Dtype>::~ImageColorizationDataLayer<Dtype>() {
        this->StopInternalThread();
    }

    template <typename Dtype>
    void ImageColorizationDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        const int new_height = this->layer_param_.image_data_param().new_height();
        const int new_width  = this->layer_param_.image_data_param().new_width();
        string root_folder = this->layer_param_.image_data_param().root_folder();

        /*CHECK(label_type==ImageDataParameter_LabelType_NONE) << 
        "ImageColorizationDataLayer only supports label_type: NONE.";*/

        TransformationParameter transform_param = this->layer_param_.transform_param();
        CHECK(transform_param.has_mean_file() == false) << 
        "ImageColorizationDataLayer does not support mean file";
        CHECK((new_height == 0 && new_width == 0) ||
        (new_height > 0 && new_width > 0)) << "Current implementation requires "
        "new_height and new_width to be set at the same time.";

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
        const int crop_size = this->layer_param_.transform_param().crop_size();
        if (crop_size>0) {
            height=width=crop_size;
        }
        else if (new_height>0 && new_width>0) {
            height=new_height;
            width=new_width;
        }
        else {
            // Read an image, and use it to initialize the top blob.
            cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_],
            new_height, new_width, true);
            if (!cv_img.data) {
                DLOG(INFO) << "Fail to load input: " << root_folder + lines_[lines_id_];
                CHECK(false) << "Fail to load input: " << root_folder + lines_[lines_id_];
            }
            height = cv_img.rows;
            width = cv_img.cols;
        }
        // Number of image/label channels.
        const int channels=1;
        const int label_channels=2;
        
        const int batch_size = this->layer_param_.image_data_param().batch_size();
        //image
        top[0]->Reshape(batch_size, channels, height, width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
        }
        this->transformed_data_.Reshape(1, channels, height, width);

        //label
        top[1]->Reshape(batch_size, label_channels, height, width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(batch_size, label_channels, height, width);
        }
        this->transformed_label_.Reshape(1, label_channels, height, width);     


        // image dimensions, for each image, stores (img_height, img_width)
        top[2]->Reshape(batch_size, 1, 1, 2);
        this->prefetch_data_dim_.Reshape(batch_size, 1, 1, 2);
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
    void ImageColorizationDataLayer<Dtype>::ShuffleImages() {
        caffe::rng_t* prefetch_rng =
        static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        shuffle(lines_.begin(), lines_.end(), prefetch_rng);
    }

    // This function is called on prefetch thread
    template <typename Dtype>
    void ImageColorizationDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
        
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK(this->transformed_data_.count());
        Dtype* top_data = batch->data_.mutable_cpu_data();
        Dtype* top_label = batch->label_.mutable_cpu_data();
        Dtype* top_data_dim = this->prefetch_data_dim_.mutable_cpu_data();

        const int max_height = batch->data_.height();
        const int max_width  = batch->data_.width();

        ImageDataParameter image_data_param = this->layer_param_.image_data_param();
        const int batch_size = image_data_param.batch_size();
        const int new_height = image_data_param.new_height();
        const int new_width  = image_data_param.new_width();
        string root_folder   = image_data_param.root_folder();

        const int lines_size = lines_.size();
        int top_data_dim_offset;

        for (int item_id = 0; item_id < batch_size; ++item_id) {
            top_data_dim_offset = this->prefetch_data_dim_.offset(item_id);
            // get a blob
            timer.Start();
            CHECK_GT(lines_size, lines_id_);
            int img_row, img_col;
            cv::Mat cv_img = ReadImageToCVMatDeconvnet(root_folder + lines_[lines_id_],
                    new_height, new_width, true, &img_row, &img_col);
            top_data_dim[top_data_dim_offset] = static_cast<Dtype>(std::min(max_height, img_row));
            top_data_dim[top_data_dim_offset + 1] = static_cast<Dtype>(std::min(max_width, img_col));

            read_time += timer.MicroSeconds();
            std::vector<cv::Mat> cv_input_label;
            timer.Start();
            /** Venkat: convert color image to YCbCr, and split as input: Y channel (H x W x 1), label: CbCr (H x W x 2)**/
            cv::cvtColor(cv_img, cv_img, CV_BGR2YCrCb);
            // Extract the L channel
            std::vector<cv::Mat> ycrcb_planes(3);
            cv::split(cv_img, ycrcb_planes);
            //L image is in ycrcb_planes[0]
            cv_input_label.push_back(ycrcb_planes[0]);
            //get rid of L channel from label
            ycrcb_planes.erase(ycrcb_planes.begin());
            cv::Mat crcb_img;
            cv::merge(ycrcb_planes,crcb_img);
            //Assign CrCb image to label
            cv_input_label.push_back(crcb_img);
            
            // Apply transformations (mirror, crop...) to the image
            int offset;
            offset = batch->data_.offset(item_id);
            this->transformed_data_.set_cpu_data(top_data + offset);

            offset = batch->label_.offset(item_id);
            this->transformed_label_.set_cpu_data(top_label + offset);

            /////Custom TransformInputAndLabel  *//////////
            this->data_transformer_->TransformInputAndLabel(cv_input_label, 
            &(this->transformed_data_), &(this->transformed_label_));
            trans_time += timer.MicroSeconds();

            // go to the next std::vector<int>::iterator iter;
            lines_id_++;
            if (lines_id_ >= lines_size) {
                // We have reached the end. Restart from the first.
                DLOG(INFO) << "Restarting data prefetching from start.";
                lines_id_ = 0;
                if (this->layer_param_.image_data_param().shuffle()) {
                    ShuffleImages();
                }
            }
        }
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    INSTANTIATE_CLASS(ImageColorizationDataLayer);
    REGISTER_LAYER_CLASS(ImageColorizationData);
}  // namespace caffe
