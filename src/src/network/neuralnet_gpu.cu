#include "neuralnet_gpu.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 1024




__global__ void k_rmsprop_update_params(cu_tensor params, cu_tensor grad, cu_tensor mean_square, float learning_rate, float decay, float eps, float grad_clip) {
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(glob_ix < params.shape.len) {
        float grad_val = grad.array[glob_ix];
        float ms = mean_square.array[glob_ix]*decay + grad_val*grad_val*(1-decay);
        mean_square.array[glob_ix] = ms;

        grad_val *= powf(ms + eps, -0.5);

        grad_val = (grad_val > grad_clip) ? grad_clip : ((grad_val < -grad_clip) ? -grad_clip : grad_val);

        params.array[glob_ix] = params.array[glob_ix]-learning_rate*grad_val;
    }
}


void w_rmsprop_update_params(cu_tensor params, cu_tensor grad, cu_tensor mean_square, float learning_rate, float decay, float eps, float grad_clip) {
    int gridDim = (params.shape.len + BLOCK_SIZE - 1) / BLOCK_SIZE; //ceil

    k_rmsprop_update_params<<<gridDim,BLOCK_SIZE>>>(params, grad, mean_square, learning_rate, decay, eps, grad_clip);
}









__global__ void k_sgd_update_params(cu_tensor params, cu_tensor grad, float learning_rate, float grad_clip) {
    int th_ix = threadIdx.x;
    int glob_ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(glob_ix < params.shape.len) {
        float grad_val = grad.array[glob_ix];
        grad_val = (grad_val > grad_clip) ? grad_clip : ((grad_val < -grad_clip) ? -grad_clip : grad_val);
        params.array[glob_ix] -= grad_val*learning_rate;
    }
}


void w_sgd_update_params(cu_tensor params, cu_tensor grad, float learning_rate, float grad_clip) {
    int blockDim = 1024;
    int gridDim = (params.shape.len+blockDim-1) / blockDim; //ceil

    // run the kernel
    k_sgd_update_params<<<gridDim, blockDim>>>(params, grad, learning_rate, grad_clip);
}