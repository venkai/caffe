#include <algorithm>
#include <vector>
#include <stdio.h>
#include "caffe/layers/conv_rec_layer.hpp"

namespace caffe {
    
    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::test_print(const int M, const int N, Dtype* A) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          printf("%0.3f\t",A[(i*N) + j]);
        }
        printf("\n");
      }
    }
    
    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::test_inverse_QR_case1() {
      // Case1: AX=B
      const int M = 3, N = 4;
      
      /*       | 1 2 3 |          | 9.860  6.000  3.400  3.250 |
       *   A = | 4 5 6 |      B = | 22.04  15.00  9.370  7.480 |
       *       | 2 1 1 |          | 4.290  4.000  2.840  1.620 |
       *
       *   Solving X for AX=B. 
       *   Expected X = | 0.23  1.00  0.85  0.21 |
       *                | 1.86  1.00  0.87  0.56 |
       *                | 1.97  1.00  0.27  0.64 |
       */
       
      /* In row major, */
      float Aorig[M*M] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 1.0};
      float Xorig[M*N] = { 0.23, 1, 0.85, 0.21, 1.86, 1, 0.87, 0.56, 1.97, 1, 0.27, 0.64  }; // exact solution
      float Borig[M*N] = { 9.86, 6.0, 3.4, 3.25, 22.04, 15.0, 9.37, 7.48, 4.29, 4.0, 2.84, 1.62};
      
      for (int i = 0; i < M*M; ++i) { eye_.mutable_cpu_data()[i] = Dtype(Aorig[i]); }
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Borig[i]); }
      int* devInfo = dev_info_.mutable_gpu_data();
      LOG(INFO) << "Solving for X in A*X = B (in rowmajor), or X*A=B in column major, where";
      printf("A = \n"); test_print(M,M,eye_.mutable_cpu_data());
      printf("B = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      caffe_gpu_inverse_qr(CblasLeft, CblasNoTrans, M, N, Dtype(1.0), eye_.mutable_gpu_data(), tau_.mutable_gpu_data(),
          wt_buffer_.mutable_gpu_data(), Lwork_, workspace_.mutable_gpu_data(), devInfo);
      LOG(INFO)<<"Results: ";
      printf("Anew = \n"); test_print(M,M,eye_.mutable_cpu_data());
      printf("TAU = \n"); test_print(1,M,tau_.mutable_cpu_data());
      printf("Xhat = B*inv(R)*Q' = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data());
      printf("Expected Solution X = \n");
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Xorig[i]); }
      test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      printf("--------------------------------------------------------------------------\n");
    }
    
    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::test_inverse_QR_case2() {
      // Case2: trans(A)*X=B
      const int M = 3, N = 4;
      
      /*       | 1 4 2 |          | 9.860  6.000  3.400  3.250 |
       *   A = | 2 5 1 |      B = | 22.04  15.00  9.370  7.480 |
       *       | 3 6 1 |          | 4.290  4.000  2.840  1.620 |
       *
       *   Solving X for trans(A)*X=B. 
       *   Expected X = | 0.23  1.00  0.85  0.21 |
       *                | 1.86  1.00  0.87  0.56 |
       *                | 1.97  1.00  0.27  0.64 |
       */
       
      /* In row major, */
      float Aorig[M*M] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
      float Xorig[M*N] = { 0.23, 1, 0.85, 0.21, 1.86, 1, 0.87, 0.56, 1.97, 1, 0.27, 0.64  }; // exact solution
      float Borig[M*N] = { 9.86, 6.0, 3.4, 3.25, 22.04, 15.0, 9.37, 7.48, 4.29, 4.0, 2.84, 1.62};
      
      for (int i = 0; i < M*M; ++i) { eye_.mutable_cpu_data()[i] = Dtype(Aorig[i]); }
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Borig[i]); }
      int* devInfo = dev_info_.mutable_gpu_data();
      LOG(INFO) << "Solving for X in trans(A)*X = B (in rowmajor), or X*trans(A)=B in column major, where";
      printf("A = \n"); test_print(M,M,eye_.mutable_cpu_data());
      printf("B = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      caffe_gpu_inverse_qr(CblasLeft, CblasTrans, M, N, Dtype(1.0), eye_.mutable_gpu_data(), tau_.mutable_gpu_data(),
          wt_buffer_.mutable_gpu_data(), Lwork_, workspace_.mutable_gpu_data(), devInfo);
      LOG(INFO)<<"Results: ";
      printf("Anew = \n"); test_print(M,M,eye_.mutable_cpu_data());
      printf("TAU = \n"); test_print(1,M,tau_.mutable_cpu_data());
      printf("Xhat = B*Q*inv(R') = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data());
      printf("Expected Solution X = \n");
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Xorig[i]); }
      test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      printf("--------------------------------------------------------------------------\n");
    }
    
    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::test_inverse_QR_case3() {
      // Case3: X*trans(A)=B
      const int M = 4, N = 3;
      
      /*       | 1 2 3 |                | 9.860  6.000  3.400  3.250 |
       *   A = | 4 5 6 |     trans(B) = | 22.04  15.00  9.370  7.480 |
       *       | 2 1 1 |                | 4.290  4.000  2.840  1.620 |
       *
       *   
       *   Solving X for X*trans(A)=B. 
       *   Expected trans(X) = | 0.23  1.00  0.85  0.21 |
       *                       | 1.86  1.00  0.87  0.56 |
       *                       | 1.97  1.00  0.27  0.64 |
       */
       
      /* In row major, */
      float Aorig[N*N] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 1.0};
      float Xorig[M*N] = { 0.23, 1.86, 1.97, 1.0, 1.0, 1.0, 0.85, 0.87, 0.27, 0.21, 0.56, 0.64}; // exact solution
      float Borig[M*N] = { 9.86, 22.04, 4.29, 6.0, 15.0, 4.0, 3.4, 9.37, 2.84, 3.25, 7.48, 1.62};
      
      for (int i = 0; i < N*N; ++i) { eye_.mutable_cpu_data()[i] = Dtype(Aorig[i]); }
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Borig[i]); }
      int* devInfo = dev_info_.mutable_gpu_data();
      LOG(INFO) << "Solving for X in X*trans(A) = B (in rowmajor), or trans(A)*X=B in column major, where";
      printf("A = \n"); test_print(N,N,eye_.mutable_cpu_data());
      printf("B = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      caffe_gpu_inverse_qr(CblasRight, CblasTrans, M, N, Dtype(1.0), eye_.mutable_gpu_data(), tau_.mutable_gpu_data(),
          wt_buffer_.mutable_gpu_data(), Lwork_, workspace_.mutable_gpu_data(), devInfo);
      LOG(INFO)<<"Results: ";
      printf("Anew = \n"); test_print(N,N,eye_.mutable_cpu_data());
      printf("TAU = \n"); test_print(1,N,tau_.mutable_cpu_data());
      printf("Xhat = Q*inv(R')*B = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data());
      printf("Expected Solution X = \n");
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Xorig[i]); }
      test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      printf("--------------------------------------------------------------------------\n");
    }
    
    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::test_inverse_QR_case4() {
      // Case4: X*A=B
      const int M = 4, N = 3;
      
      /*       | 1 4 2 |                | 9.860  6.000  3.400  3.250 |
       *   A = | 2 5 1 |     trans(B) = | 22.04  15.00  9.370  7.480 |
       *       | 3 6 1 |                | 4.290  4.000  2.840  1.620 |
       *
       *   Solving X for X*A=B. 
       *   Expected trans(X) = | 0.23  1.00  0.85  0.21 |
       *                       | 1.86  1.00  0.87  0.56 |
       *                       | 1.97  1.00  0.27  0.64 |
       */
       
      /* In row major, */
      float Aorig[M*M] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
      float Xorig[M*N] = { 0.23, 1.86, 1.97, 1.0, 1.0, 1.0, 0.85, 0.87, 0.27, 0.21, 0.56, 0.64}; // exact solution
      float Borig[M*N] = { 9.86, 22.04, 4.29, 6.0, 15.0, 4.0, 3.4, 9.37, 2.84, 3.25, 7.48, 1.62};
      
      for (int i = 0; i < N*N; ++i) { eye_.mutable_cpu_data()[i] = Dtype(Aorig[i]); }
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Borig[i]); }
      int* devInfo = dev_info_.mutable_gpu_data();
      LOG(INFO) << "Solving for X in X*A = B (in rowmajor), or A*X=B in column major, where";
      printf("A = \n"); test_print(N,N,eye_.mutable_cpu_data());
      printf("B = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      caffe_gpu_inverse_qr(CblasRight, CblasNoTrans, M, N, Dtype(1.0), eye_.mutable_gpu_data(), tau_.mutable_gpu_data(),
          wt_buffer_.mutable_gpu_data(), Lwork_, workspace_.mutable_gpu_data(), devInfo);
      LOG(INFO)<<"Results: ";
      printf("Anew = \n"); test_print(N,N,eye_.mutable_cpu_data());
      printf("TAU = \n"); test_print(1,N,tau_.mutable_cpu_data());
      printf("Xhat = inv(R)*Q'*B = \n"); test_print(M,N,wt_buffer_.mutable_cpu_data());
      printf("Expected Solution X = \n");
      for (int i = 0; i < M*N; ++i) { wt_buffer_.mutable_cpu_data()[i] = Dtype(Xorig[i]); }
      test_print(M,N,wt_buffer_.mutable_cpu_data()); printf("\n");
      printf("--------------------------------------------------------------------------\n");
    }
    
    template <typename Dtype>
    __global__ void PermuteKernel(const int nthreads, const int num_axes, const int* old_steps,
    const int* new_steps, const int* new_orders, const Dtype* bottom_data, Dtype* top_data) {
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
    void RecursiveConvLayer<Dtype>::permute_blobs_gpu(const vector<Blob<Dtype>*>& bottom, const bool channel_last, const bool permute_diffs) {
        const int num_axes = bottom[0]->num_axes();
        const int count = bottom[0]->count();
        if (channel_last) {
            mid_.Reshape(new_mid_shape_);
        } else {
            mid_.Reshape(old_mid_shape_);
        }
        /*caffe_gpu_set(count, Dtype(0), mid_.mutable_gpu_data());
        if (permute_diffs) {
            caffe_gpu_set(count, Dtype(0), mid_.mutable_gpu_diff());
        }*/
        //Start Permuting bottom blob data
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = mid_.mutable_gpu_data();
        if (channel_last) {
            // NOLINT_NEXT_LINE(whitespace/operators)
            PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, num_axes, old_steps_.gpu_data(), new_steps_.gpu_data(), permute_order_.gpu_data(), bottom_data, top_data);
            CUDA_POST_KERNEL_CHECK;
        } else {
            // NOLINT_NEXT_LINE(whitespace/operators)
            PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, num_axes, new_steps_.gpu_data(), old_steps_.gpu_data(), inv_permute_order_.gpu_data(), bottom_data, top_data);
            CUDA_POST_KERNEL_CHECK;
        }
        if (!permute_diffs) { return; }
        //Start Permuting bottom blob diffs
        const Dtype* bottom_diff = bottom[0]->gpu_diff();
        Dtype* top_diff = mid_.mutable_gpu_diff();
        if (channel_last) {
            // NOLINT_NEXT_LINE(whitespace/operators)
            PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, num_axes, old_steps_.gpu_data(), new_steps_.gpu_data(), permute_order_.gpu_data(), bottom_diff, top_diff);
            CUDA_POST_KERNEL_CHECK;
        } else {
            // NOLINT_NEXT_LINE(whitespace/operators)
            PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, num_axes, new_steps_.gpu_data(), old_steps_.gpu_data(), inv_permute_order_.gpu_data(), bottom_diff, top_diff);
            CUDA_POST_KERNEL_CHECK;
        }
    }


    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        LOG(INFO) << "---- CASE 1 ----"; test_inverse_QR_case1();
        LOG(INFO) << "---- CASE 2 ----"; test_inverse_QR_case2();
        LOG(INFO) << "---- CASE 3 ----"; test_inverse_QR_case3();
        LOG(INFO) << "---- CASE 4 ----"; test_inverse_QR_case4();
        
        const Dtype* weights = this->blobs_[0]->mutable_gpu_data(); // Ni X No
        //caffe_copy(top[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
        const bool order_C_last = true;
        const bool inv_order_C_last = false;
        permute_blobs_gpu(bottom,order_C_last,false); // Permute bottom from NxCxHxW to (N*H*W) x C and copy to mid_
        top[0]->ReshapeLike(mid_);
        // caffe_copy(mid_.count(), mid_.gpu_data(), top[0]->mutable_gpu_data());
        if (!use_global_stats_) {
            //this->blobs_[3]->mutable_cpu_data()[0] *= moving_average_fraction_;
            //this->blobs_[3]->mutable_cpu_data()[0] += 1;
            caffe_gpu_scal<Dtype>(this->blobs_[3]->count(),moving_average_fraction_,this->blobs_[3]->mutable_gpu_data());
            caffe_gpu_add_scalar(this->blobs_[3]->count(), Dtype(1), this->blobs_[3]->mutable_gpu_data());
        }
        for (int iter = 0; iter < Nrec_; ++iter) {
            // Standard 1x1 convolution
            const int wt_offset = rand_wt_order_[iter] * C_ * C_;
            // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, num_outputs, num_inputs, (Dtype)1., bottom_data, wt_trans, (Dtype)0., top_data);
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, C_, (Dtype)1., mid_.gpu_data(), weights + wt_offset, (Dtype)0., top[0]->mutable_gpu_data());
            
            /*[Optional] Insert permutations, activation functions, batch-norm, etc here*/
            
            //Compute activation function in-place
            forward_activation_func_gpu(top,top); //a_{i+1} = \sigma(a_{i+1});
            
            //Apply BN
            forward_BN_gpu(top,top,iter);
            
            if (iter == Nrec_ - 1) {
                permute_blobs_gpu(top,inv_order_C_last,false); // Permute top from (N*H*W) x C to NxCxHxW and copy to mid_
                top[0]->ReshapeLike(mid_);
                caffe_copy(mid_.count(), mid_.gpu_data(), top[0]->mutable_gpu_data());
            } else {
                //mid_ <- top; //a_i <- a_{i+1};
                caffe_copy(mid_.count(), top[0]->gpu_data(), mid_.mutable_gpu_data());
            }
        }
        
    }

    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        if (!this->param_propagate_down_[0] && !propagate_down[0]) {
            return;
        }
        const bool order_C_last = true;
        const bool inv_order_C_last = false;
        const Dtype* wt_inv_data = wt_inv_.gpu_data();
        const Dtype* weights = this->blobs_[0]->gpu_data();
        Dtype* weights_diff = this->blobs_[0]->mutable_gpu_diff();
        // mid_ <- top
        permute_blobs_gpu(top,order_C_last,true);
        bottom[0]->ReshapeLike(mid_);
        caffe_copy(bottom[0]->count(), mid_.gpu_data(), bottom[0]->mutable_gpu_data());
        caffe_copy(bottom[0]->count(), mid_.gpu_diff(), bottom[0]->mutable_gpu_diff());
        // TOP Data & Diff are now in BOTTOM, permuted in order (N*H*W) x C
        for (int iter = Nrec_-1; iter >=0; --iter) {
            backward_BN_gpu(bottom,bottom,iter);
            backward_activation_func_gpu(bottom,bottom);
            /* Invert data(bottom[0])*inv(W)->data(mid_), compute diff(W) and backprop diff(bottom[0])->diff(mid_)  */
            const int wt_offset = rand_wt_order_[iter] * C_ * C_;
            // LOG(INFO)<< "Iter = " << iter << ", wt_offset = " << wt_offset;
            // First get BOTTOM data using the inverse of weights 
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, C_, (Dtype)1., bottom[0]->gpu_data(), wt_inv_data + wt_offset, (Dtype)0., mid_.mutable_gpu_data());
            // Note: BOTTOM Data is now in mid_, TOP Data & Diff are still in bottom[0] 
            if (this->param_propagate_down_[0]) { // compute diff with respect to weights if needed
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, C_, C_, batch_size_, (Dtype)1., mid_.gpu_data(), bottom[0]->gpu_diff(), (Dtype)1., weights_diff + wt_offset);
            }
            // Compute diff with respect to bottom activation (we must always do this, even if propagate_down[0] is false)
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size_, C_, C_, (Dtype)1., bottom[0]->gpu_diff(), weights + wt_offset, (Dtype)0., mid_.mutable_gpu_diff());
            // Note: BOTTOM Diff is now in mid_, TOP Data & Diff are still in bottom[0] 
            // Transfer Data & Diff from mid_ to bottom[0]
            caffe_copy(bottom[0]->count(), mid_.gpu_data(), bottom[0]->mutable_gpu_data());
            caffe_copy(bottom[0]->count(), mid_.gpu_diff(), bottom[0]->mutable_gpu_diff());
        }
        permute_blobs_gpu(bottom,inv_order_C_last,true); // Permute bottom from (N*H*W) x C to NxCxHxW and copy to mid_
        bottom[0]->ReshapeLike(mid_);
        caffe_copy(bottom[0]->count(), mid_.gpu_data(), bottom[0]->mutable_gpu_data());
        caffe_copy(bottom[0]->count(), mid_.gpu_diff(), bottom[0]->mutable_gpu_diff());
    }


    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::forward_activation_func_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        RecursiveConvLayer<Dtype>::forward_ReLU_gpu(bottom,top);
    }

    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::backward_activation_func_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
        RecursiveConvLayer<Dtype>::backward_ReLU_gpu(top,bottom);
    }

    template <typename Dtype>
    __global__ void ReLUForward(const int nthreads, const Dtype negative_slope, const Dtype* bottom_data, Dtype* top_data) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            top_data[index] = bottom_data[index] > 0 ? bottom_data[index] : bottom_data[index] * negative_slope;
        }
    }
    
    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::forward_ReLU_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        // NOLINT_NEXT_LINE(whitespace/operators)
        ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), negative_slope_, bottom_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }

    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::forward_BN_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top, int iter) {
        const int offset = iter*C_;
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        if (bottom[0] != top[0]) { caffe_copy(bottom[0]->count(), bottom_data, top_data); }
        if (!use_global_stats_) {
            // Compute Batch-Mean E(X)
            caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, inv_batch_size_, bottom_data, batch_sum_multiplier_.gpu_data(), 0., bn_mu_.mutable_gpu_data() + offset);
        }
        //Subtract Batch-Mean
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1, -1., batch_sum_multiplier_.gpu_data(), bn_mu_.gpu_data() + offset, 1., top_data);
        if (!use_global_stats_) {
            // Compute Batch-Variance E((X-EX)^2)
            //caffe_gpu_powx(top[0]->count(), top_data, Dtype(2), mid_.mutable_gpu_data() );  // (X-EX)^2
            caffe_gpu_mul(top[0]->count(), top_data, top_data, mid_.mutable_gpu_data() );  // (X-EX)^2
            caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, inv_batch_size_, mid_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0., bn_sigma_.mutable_gpu_data() + offset); // E((X-EX)^2)
            
            // Compute and save moving average
            caffe_gpu_axpby(C_, Dtype(1), bn_mu_.gpu_data() + offset, moving_average_fraction_, this->blobs_[1]->mutable_gpu_data() + offset);
            caffe_gpu_axpby(C_, bias_correction_factor_, bn_sigma_.gpu_data() + offset, moving_average_fraction_, this->blobs_[2]->mutable_gpu_data() + offset);
            
            // Compute Batch-St-dev = sqrt(Batch-Variance + epsilon)
            caffe_gpu_add_scalar(C_, eps_, bn_sigma_.mutable_gpu_data() + offset);
            caffe_gpu_powx(C_, bn_sigma_.gpu_data() + offset , Dtype(0.5), bn_sigma_.mutable_gpu_data() + offset);
        }
        // Replicate Batch-St-dev to input size
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1, 1., batch_sum_multiplier_.gpu_data(), bn_sigma_.gpu_data() + offset , 0., mid_.mutable_gpu_data() );
        // Divide by Batch-St-dev
        caffe_gpu_div(mid_.count(), top_data, mid_.gpu_data(), top_data);
    }

    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::backward_BN_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom, int iter) {
        const int offset = iter*C_;
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* top_data = top[0]->gpu_data();
        Dtype* bottom_data = bottom[0]->mutable_gpu_data();
        // Replicate Batch-St-dev to input size
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1, 1., batch_sum_multiplier_.gpu_data(), bn_sigma_.gpu_data() + offset , 0., mid_.mutable_gpu_data() );
        if (use_global_stats_) {
            caffe_gpu_div(mid_.count(), top[0]->gpu_diff(), mid_.gpu_data(), bottom_diff);
            
            // Invert BN
            // Multiply by Batch-St-Dev
            caffe_gpu_mul(mid_.count(), top_data, mid_.gpu_data(), bottom_data);
            // Add Batch-Mean
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1, 1., batch_sum_multiplier_.gpu_data(), bn_mu_.gpu_data() + offset, 1., bottom_data);
            return;
        }
        const Dtype* top_diff;
        if (bottom[0] != top[0]) {
            top_diff = top[0]->gpu_diff();
        } else {
            caffe_copy(mid_.count(), top[0]->gpu_diff(), mid_.mutable_gpu_diff());
            top_diff = mid_.gpu_diff();
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
        caffe_gpu_mul(mid_.count(), top_data, top_diff, bottom_diff);
        caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, 1., bottom_diff, batch_sum_multiplier_.gpu_data(), 0., temp_bn_sum_.mutable_gpu_data());
        // reshape (broadcast) the above
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1, 1., batch_sum_multiplier_.gpu_data(), temp_bn_sum_.gpu_data(), 0., bottom_diff);
        // sum(dE/dY \cdot Y) \cdot Y
        caffe_gpu_mul(mid_.count(), top_data, bottom_diff, bottom_diff);
        // sum(dE/dY)
        caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, C_, 1., top_diff, batch_sum_multiplier_.gpu_data(), 0., temp_bn_sum_.mutable_gpu_data());
        // reshape (broadcast) the above to make sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1, 1., batch_sum_multiplier_.gpu_data(), temp_bn_sum_.gpu_data(), 1., bottom_diff);
        // dE/dY - mean(dE/dY)- (mean(dE/dY \cdot Y) \cdot Y)
        caffe_gpu_axpby(mid_.count(), Dtype(1), top_diff, (Dtype(-1.) * inv_batch_size_), bottom_diff);
        // note: mid_.gpu_data() contains sqrt(var(X)+eps)
        caffe_gpu_div(mid_.count(), bottom_diff, mid_.gpu_data(), bottom_diff);
        
        // Invert BN
        // Multiply by Batch-St-Dev
        caffe_gpu_mul(mid_.count(), top_data, mid_.gpu_data(), bottom_data);
        // Add Batch-Mean
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, C_, 1, 1., batch_sum_multiplier_.gpu_data(), bn_mu_.gpu_data() + offset, 1., bottom_data);
    }

    template <typename Dtype>
    __global__ void ReLUBackward(const int nthreads, const Dtype negative_slope, const Dtype inv_negative_slope, 
    const Dtype* top_data, const Dtype* top_diff, Dtype* bottom_data, Dtype* bottom_diff) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            bottom_diff[index] = top_data[index] > 0 ? top_diff[index] : top_diff[index] * negative_slope;
            bottom_data[index] = top_data[index] > 0 ? top_data[index] : top_data[index] * inv_negative_slope;
        }
    }
    
    template <typename Dtype>
    void RecursiveConvLayer<Dtype>::backward_ReLU_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
        const Dtype* top_data = top[0]->gpu_data();
        Dtype* bottom_data = bottom[0]->mutable_gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        // NOLINT_NEXT_LINE(whitespace/operators)
        ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), negative_slope_, inv_negative_slope_, top_data, top_diff, bottom_data, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
    }


    INSTANTIATE_LAYER_GPU_FUNCS(RecursiveConvLayer);
}  // namespace caffe
